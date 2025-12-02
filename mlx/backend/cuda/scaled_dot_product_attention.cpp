// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

array prepare_sdpa_input(const array& x, Stream s) {
  // SDPA kernel's requirements on inputs:
  // 1. last dim's stride be 1;
  // 2. pointer be aligned.
  if (x.strides(-1) != 1 || get_alignment(x) < 16) {
    array x_copy = contiguous_copy_gpu(x, s);
    auto& encoder = cu::get_command_encoder(s);
    encoder.add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

// Return argsort(x.strides()).
Strides argsort_strides(const array& x) {
  Strides argsort(x.ndim());
  std::iota(argsort.rbegin(), argsort.rend(), 0);
  if (!x.flags().row_contiguous) {
    std::stable_sort(argsort.begin(), argsort.end(), [&x](int idx1, int idx2) {
      auto s1 = x.strides(idx1) > 0 ? x.strides(idx1) : 1;
      auto s2 = x.strides(idx2) > 0 ? x.strides(idx2) : 1;
      return s1 < s2;
    });
  }
  return argsort;
}

// Compute strides of from |shape|, and transpose it in order of |argsort|.
template <typename T, typename U>
Strides strides_from_shape(const T& shape, const U& argsort) {
  Strides strides(shape.size());
  int64_t stride = 1;
  for (int i : argsort) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

void malloc_with_same_layout(
    cu::CommandEncoder& encoder,
    array& o,
    const array& q) {
  auto buffer = cu::malloc_async(o.nbytes(), encoder);
  auto argsort = argsort_strides(q);
  if (std::is_sorted(argsort.rbegin(), argsort.rend())) {
    o.set_data(buffer);
  } else {
    o.set_data(
        buffer,
        o.size(),
        strides_from_shape(o.shape(), argsort),
        {true, false, false});
  }
}

template <typename T>
void change_seq_length(
    std::shared_ptr<fe::graph::Tensor_attributes>& attrs,
    int64_t seq_length,
    const T& argsort) {
  auto shape = attrs->get_dim();
  shape[2] = seq_length;
  auto strides = strides_from_shape(shape, argsort);
  attrs->set_dim(shape).set_stride(convert_vector<int64_t>(strides));
}

constexpr int MAX_SEQ_LENGTH_Q = 12'000;
constexpr int MAX_SEQ_LENGTH_KV = 12'000;

bool uses_variable_seq_length(
    const array& q,
    const array& k,
    const array& v,
    const std::optional<array>& mask_arr) {
  if (!q.flags().contiguous || !k.flags().contiguous || !v.flags().contiguous) {
    return false;
  }
  if (mask_arr && !mask_arr->flags().contiguous) {
    return false;
  }
  return true;
}

constexpr int QKV_NDIM = 4;

struct SDPACacheKey {
  int device_id;
  fe::DataType_t cudnn_dtype;
  bool do_causal;
  bool output_logsumexp;
  bool variable_seq_length;
  std::array<int, QKV_NDIM> q_shape;
  std::array<int, QKV_NDIM> k_shape;
  std::array<int, QKV_NDIM> v_shape;
  std::array<int, QKV_NDIM> mask_shape;
  std::array<int64_t, QKV_NDIM> q_strides;
  std::array<int64_t, QKV_NDIM> k_strides;
  std::array<int64_t, QKV_NDIM> v_strides;
  std::array<int64_t, QKV_NDIM> mask_strides;
};

BytesKey<SDPACacheKey> build_sdpa_cache_key(
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    bool output_logsumexp = true,
    bool variable_seq_length = false) {
  BytesKey<SDPACacheKey> cache_key;
  cache_key.pod = {
      encoder.device().cuda_device(),
      dtype_to_cudnn_type(q.dtype()),
      do_causal,
      output_logsumexp,
      variable_seq_length,
      vector_key<QKV_NDIM>(q.shape()),
      vector_key<QKV_NDIM>(k.shape()),
      vector_key<QKV_NDIM>(v.shape()),
  };
  if (mask_arr) {
    cache_key.pod.mask_shape = vector_key<QKV_NDIM>(mask_arr->shape());
  }
  if (variable_seq_length) {
    cache_key.pod.q_shape[2] = MAX_SEQ_LENGTH_Q;
    cache_key.pod.k_shape[2] = MAX_SEQ_LENGTH_KV;
    cache_key.pod.v_shape[2] = MAX_SEQ_LENGTH_KV;
    cache_key.pod.mask_shape[2] = MAX_SEQ_LENGTH_Q;
    cache_key.pod.mask_shape[3] = MAX_SEQ_LENGTH_KV;
    cache_key.pod.q_strides = vector_key<QKV_NDIM>(argsort_strides(q));
    cache_key.pod.k_strides = vector_key<QKV_NDIM>(argsort_strides(k));
    cache_key.pod.v_strides = vector_key<QKV_NDIM>(argsort_strides(v));
    if (mask_arr) {
      cache_key.pod.mask_strides =
          vector_key<QKV_NDIM>(argsort_strides(*mask_arr));
    }
  } else {
    cache_key.pod.q_strides = vector_key<QKV_NDIM>(q.strides());
    cache_key.pod.k_strides = vector_key<QKV_NDIM>(k.strides());
    cache_key.pod.v_strides = vector_key<QKV_NDIM>(v.strides());
    if (mask_arr) {
      cache_key.pod.mask_strides = vector_key<QKV_NDIM>(mask_arr->strides());
    }
  }
  return cache_key;
}

auto& sdpa_cache() {
  static LRUBytesKeyCache<SDPACacheKey, DnnGraph> cache(
      "MLX_CUDA_SDPA_CACHE_SIZE", /* default_capacity */ 64);
  return cache;
}

auto& sdpa_backward_cache() {
  static LRUBytesKeyCache<SDPACacheKey, DnnGraph> cache(
      "MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE", /* default_capacity */ 64);
  return cache;
}

enum UIDS {
  Q,
  K,
  V,
  SCALE,
  BIAS,
  O,
  STATS,
  // Variable seq length:
  SEQ_LEN_Q,
  SEQ_LEN_KV,
  // Backward graph:
  D_Q,
  D_K,
  D_V,
  D_O,
};

DnnGraph build_sdpa_graph(
    cudnnHandle_t handle,
    const SDPACacheKey& cache_key,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    bool output_logsumexp,
    const array& o,
    const array& stats) {
  DnnGraph graph(handle, q.dtype());

  auto q_ = graph.tensor("Q", Q, q);
  auto k_ = graph.tensor("K", K, k);
  auto v_ = graph.tensor("V", V, v);
  if (cache_key.variable_seq_length) {
    change_seq_length(q_, MAX_SEQ_LENGTH_Q, cache_key.q_strides);
    change_seq_length(k_, MAX_SEQ_LENGTH_KV, cache_key.k_strides);
    change_seq_length(v_, MAX_SEQ_LENGTH_KV, cache_key.v_strides);
  }

  auto options = fe::graph::SDPA_attributes()
                     .set_name("sdpa_cudnn")
                     .set_attn_scale(graph.scalar("Scale", SCALE, float32))
                     .set_generate_stats(output_logsumexp);
  if (do_causal) {
    if (q.shape(2) > k.shape(2)) {
      options.set_causal_mask(do_causal);
    } else {
      options.set_causal_mask_bottom_right(do_causal);
    }
  }
  if (mask_arr) {
    auto bias_ = graph.tensor("BIAS", BIAS, *mask_arr);
    if (cache_key.variable_seq_length) {
      auto shape = bias_->get_dim();
      shape[2] = MAX_SEQ_LENGTH_Q;
      shape[3] = MAX_SEQ_LENGTH_KV;
      auto strides = bias_->get_stride();
      for (int i = shape.size() - 1, stride = 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
      }
      bias_->set_dim(shape).set_stride(strides);
    }
    options.set_bias(bias_);
  }
  if (cache_key.variable_seq_length) {
    auto seq_len_q_ = graph.scalar("SEQ_LEN_Q", SEQ_LEN_Q, int32);
    seq_len_q_->set_dim({q.shape(0), 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_is_pass_by_value(false);
    auto seq_len_kv_ = graph.scalar("SEQ_LEN_KV", SEQ_LEN_KV, int32);
    seq_len_kv_->set_dim({q.shape(0), 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_is_pass_by_value(false);
    options.set_padding_mask(true)
        .set_seq_len_q(seq_len_q_)
        .set_seq_len_kv(seq_len_kv_);
  }

  auto [o_, stats_] = graph.sdpa(q_, k_, v_, options);
  graph.tensor(o_, O, o)->set_output(true);
  if (output_logsumexp) {
    graph.tensor(stats_, STATS, stats)->set_output(true);
  }
  if (cache_key.variable_seq_length) {
    change_seq_length(o_, MAX_SEQ_LENGTH_Q, cache_key.q_strides);
    if (output_logsumexp) {
      change_seq_length(stats_, MAX_SEQ_LENGTH_Q, cache_key.q_strides);
    }
  }

  CHECK_CUDNN_FE_ERROR(graph.prepare());
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_FE_ERROR(graph.build());
  return graph;
}

DnnGraph build_sdpa_backward_graph(
    cudnnHandle_t handle,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const array& o,
    const array& d_o,
    const array& stats,
    array& d_q,
    array& d_k,
    array& d_v) {
  DnnGraph graph(handle, q.dtype());

  auto q_ = graph.tensor("Q", Q, q);
  auto k_ = graph.tensor("K", K, k);
  auto v_ = graph.tensor("V", V, v);
  auto o_ = graph.tensor("O", O, o);
  auto d_o_ = graph.tensor("D_O", D_O, d_o);
  auto stats_ = graph.tensor("STATS", STATS, stats);

  auto options = fe::graph::SDPA_backward_attributes()
                     .set_name("sdpa_backward_cudnn")
                     .set_attn_scale(graph.scalar("Scale", SCALE, float32));
  if (do_causal) {
    if (q.shape(2) > k.shape(2)) {
      options.set_causal_mask(do_causal);
    } else {
      options.set_causal_mask_bottom_right(do_causal);
    }
  }
  if (mask_arr) {
    options.set_bias(graph.tensor("BIAS", BIAS, *mask_arr));
  }

  auto [d_q_, d_k_, d_v_] =
      graph.sdpa_backward(q_, k_, v_, o_, d_o_, stats_, options);
  graph.tensor(d_q_, D_Q, d_q)->set_output(true);
  graph.tensor(d_k_, D_K, d_k)->set_output(true);
  graph.tensor(d_v_, D_V, d_v)->set_output(true);

  CHECK_CUDNN_FE_ERROR(graph.prepare());
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_FE_ERROR(graph.build());
  return graph;
}

} // namespace

bool supports_sdpa_cudnn(
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    Stream s) {
  static bool enabled = env::get_var("MLX_CUDA_USE_CUDNN_SPDA", 1);
  if (!enabled) {
    return false;
  }

  // cuDNN SDPA requires Ampere and later.
  if (cu::device(s.device).compute_capability_major() < 8) {
    return false;
  }

  // Only use cuDNN for prefilling (T_q > 1) and training (T_q == T_kv).
  if ((q.shape(2) == 1) && (q.shape(2) != k.shape(2))) {
    return false;
  }

  // D_qk and D_v must be a multiple of 8 with maximum value 128.
  if ((q.shape(-1) % 8 != 0) || (q.shape(-1) > 128) || (v.shape(-1) % 8 != 0) ||
      (v.shape(-1) > 128)) {
    return false;
  }

  Dtype dtype = q.dtype();
  return dtype == float16 || dtype == bfloat16;
}

void sdpa_cudnn(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    array& stats,
    bool do_causal,
    const std::optional<array>& mask_arr,
    bool output_logsumexp,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  auto handle = encoder.device().cudnn_handle();

  malloc_with_same_layout(encoder, o, q);

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_output_array(o);
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }
  if (output_logsumexp) {
    stats.set_data(cu::malloc_async(stats.nbytes(), encoder));
    encoder.set_output_array(stats);
  }

  bool variable_seq_length = uses_variable_seq_length(q, k, v, mask_arr);

  // Search cache.
  auto cache_key = build_sdpa_cache_key(
      encoder,
      q,
      k,
      v,
      do_causal,
      mask_arr,
      output_logsumexp,
      variable_seq_length);
  auto it = sdpa_cache().find(cache_key);
  if (it == sdpa_cache().end()) {
    auto graph = build_sdpa_graph(
        handle,
        cache_key.pod,
        q,
        k,
        v,
        do_causal,
        mask_arr,
        output_logsumexp,
        o,
        stats);
    it = sdpa_cache().emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {Q, gpu_ptr<void>(q)},
      {K, gpu_ptr<void>(k)},
      {V, gpu_ptr<void>(v)},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)}};
  if (mask_arr) {
    variant_pack[BIAS] = gpu_ptr<void>(*mask_arr);
  }
  if (output_logsumexp) {
    variant_pack[STATS] = gpu_ptr<void>(stats);
  }
  if (variable_seq_length) {
    int B = q.shape(0);
    std::vector<int32_t> filled_q(B, q.shape(2));
    array seq_len_q(filled_q.begin(), {B}, int32);
    encoder.add_temporary(seq_len_q);
    std::vector<int32_t> filled_kv(B, k.shape(2));
    array seq_len_kv(filled_kv.begin(), {B}, int32);
    encoder.add_temporary(seq_len_kv);
    variant_pack[SEQ_LEN_Q] = gpu_ptr<void>(seq_len_q);
    variant_pack[SEQ_LEN_KV] = gpu_ptr<void>(seq_len_kv);
  }

  CHECK_CUDNN_FE_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

void sdpa_backward_cudnn(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    const array& o,
    const array& stats,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const array& d_o,
    array& d_q,
    array& d_k,
    array& d_v,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  auto handle = encoder.device().cudnn_handle();

  malloc_with_same_layout(encoder, d_q, q);
  malloc_with_same_layout(encoder, d_k, k);
  malloc_with_same_layout(encoder, d_v, v);

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_input_array(o);
  encoder.set_input_array(stats);
  encoder.set_input_array(d_o);
  encoder.set_output_array(d_q);
  encoder.set_output_array(d_k);
  encoder.set_output_array(d_v);
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }

  // Search cache.
  auto cache_key = build_sdpa_cache_key(encoder, q, k, v, do_causal, mask_arr);
  auto it = sdpa_backward_cache().find(cache_key);
  if (it == sdpa_backward_cache().end()) {
    auto graph = build_sdpa_backward_graph(
        handle, q, k, v, do_causal, mask_arr, o, d_o, stats, d_q, d_k, d_v);
    it = sdpa_backward_cache().emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {Q, gpu_ptr<void>(q)},
      {K, gpu_ptr<void>(k)},
      {V, gpu_ptr<void>(v)},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)},
      {STATS, gpu_ptr<void>(stats)},
      {D_O, gpu_ptr<void>(d_o)},
      {D_Q, gpu_ptr<void>(d_q)},
      {D_K, gpu_ptr<void>(d_k)},
      {D_V, gpu_ptr<void>(d_v)}};
  if (mask_arr) {
    variant_pack[BIAS] = gpu_ptr<void>(*mask_arr);
  }

  CHECK_CUDNN_FE_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

// Defined in scaled_dot_product_attention.cu file.
bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp);
void sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks,
    Stream s);

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }

  return !supports_sdpa_vector(
             q, k, v, has_mask, has_arr_mask, do_causal, output_logsumexp) &&
      !supports_sdpa_cudnn(q, k, v, do_causal, s);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");

  auto& s = stream();

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  auto& out = outputs[0];
  auto& stats = outputs[1];
  bool has_mask = inputs.size() - has_sinks_ > 3;
  bool has_arr_mask = has_mask && !do_causal_;

  std::optional<array> mask_arr;
  if (has_arr_mask) {
    mask_arr = prepare_sdpa_input(inputs[3], s);
  }

  if (supports_sdpa_vector(
          q, k, v, has_mask, has_arr_mask, do_causal_, output_logsumexp_)) {
    if (has_sinks_) {
      sdpa_vector(q, k, v, scale_, out, do_causal_, inputs.back(), s);
    } else {
      sdpa_vector(q, k, v, scale_, out, do_causal_, std::nullopt, s);
    }
  } else {
    sdpa_cudnn(
        q,
        k,
        v,
        scale_,
        out,
        stats,
        do_causal_,
        mask_arr,
        output_logsumexp_,
        s);
  }
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  // The frontend adds a padding mask when sequence length is not a multiple of
  // tile size.
  if (q.shape(2) % 128 != 0) {
    return true;
  }
  return s.device == Device::cpu;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ScaledDotProductAttentionVJP::eval_gpu");

  auto& s = stream();

  assert(inputs.size() >= 6);
  int primals_size = inputs.size() - 3;
  bool has_arr_mask = primals_size > 3 + has_sinks_;

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  array o = prepare_sdpa_input(inputs[primals_size], s);
  array stats = prepare_sdpa_input(inputs[primals_size + 1], s);
  array d_o = prepare_sdpa_input(inputs[primals_size + 2], s);

  std::optional<array> mask_arr;
  if (has_arr_mask) {
    mask_arr = prepare_sdpa_input(inputs[3], s);
  }

  assert(outputs.size() == 3);
  auto& d_q = outputs[0];
  auto& d_k = outputs[1];
  auto& d_v = outputs[2];

  sdpa_backward_cudnn(
      q, k, v, scale_, o, stats, do_causal_, mask_arr, d_o, d_q, d_k, d_v, s);
}

} // namespace fast

} // namespace mlx::core
