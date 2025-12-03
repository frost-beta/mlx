// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

array prepare_norm_input(const array& x, Stream s) {
  if (x.strides(-1) != 1 || !x.flags().contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    auto& encoder = cu::get_command_encoder(s);
    encoder.add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

struct RMSNormCacheKey {
  int device_id;
  fe::DataType_t cudnn_dtype;
  int dim_last;
  int dim_batch;
  bool has_weight;
  bool is_training;
};

auto& rms_norm_cache() {
  static LRUBytesKeyCache<RMSNormCacheKey, DnnGraph> cache(
      "MLX_CUDA_RMS_NORM_CACHE_SIZE", /* default_capacity */ 64);
  return cache;
}

enum UIDS {
  X,
  SCALE,
  EPS,
  Y,
};

DnnGraph build_rms_norm_graph(
    cudnnHandle_t handle,
    const array& x,
    const std::optional<array>& scale,
    array& y) {
  DnnGraph graph(handle, x.dtype());

  auto x_ = graph.tensor_2d("X", X, x);
  auto eps_ = graph.scalar("EPS", EPS, float32);

  DnnGraph::Tensor scale_;
  if (scale) {
    scale_ = graph.tensor_2d("SCALE", SCALE, *scale);
  }

  auto options = fe::graph::Rmsnorm_attributes()
                     .set_name("rms_norm_cudnn")
                     .set_forward_phase(fe::NormFwdPhase_t::INFERENCE)
                     .set_epsilon(eps_);

  auto [y_, _] = graph.rmsnorm(x_, scale_, options);
  graph.tensor_2d(y_, Y, y)->set_output(true);

  CHECK_CUDNN_FE_ERROR(graph.prepare());
  CHECK_CUDNN_FE_ERROR(graph.build());
  return graph;
}

} // namespace

bool uses_cudnn_norms() {
  static bool enabled = env::get_var("MLX_CUDA_USE_CUDNN_NORMS", 0);
  return enabled;
}

void rms_norm_cudnn(
    const array& x_,
    const array& scale_,
    array& y,
    float eps,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  auto handle = encoder.device().cudnn_handle();

  array x = prepare_norm_input(x_, s);
  y.set_data(cu::malloc_async(y.nbytes(), encoder));
  encoder.set_input_array(x);
  encoder.set_output_array(y);

  std::optional<array> scale;
  if (scale_.ndim() > 0) {
    scale = prepare_norm_input(scale_, s);
    encoder.set_input_array(*scale);
  }

  int dim_last = x.shape(-1);
  int dim_batch = x.data_size() / dim_last;

  // Search cache.
  BytesKey<RMSNormCacheKey> cache_key;
  cache_key.pod = {
      encoder.device().cuda_device(),
      dtype_to_cudnn_type(y.dtype()),
      dim_last,
      dim_batch,
      scale.has_value(),
  };
  auto it = rms_norm_cache().find(cache_key);
  if (it == rms_norm_cache().end()) {
    auto graph = build_rms_norm_graph(handle, x, scale, y);
    it = rms_norm_cache().emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {X, gpu_ptr<void>(x)}, {EPS, &eps}, {Y, gpu_ptr<void>(y)}};
  if (scale) {
    variant_pack[SCALE] = gpu_ptr<void>(*scale);
  }

  CHECK_CUDNN_FE_ERROR(
      graph.encode_capturing(encoder, std::move(variant_pack)));
}

// Defined in layer_norm.cu file.
void layer_norm_vector(
    const array& x,
    const array& w,
    const array& b,
    array& out,
    float eps,
    Stream s);
void layer_norm_backward_vector(
    const array& x,
    const array& w,
    const array& b,
    const array& g,
    array& gx,
    array& gw,
    array& gb,
    float eps,
    Stream s);
// Defined in rms_norm.cu file.
void rms_norm_vector(
    const array& x,
    const array& w,
    array& out,
    float eps,
    Stream s);
void rms_norm_backward_vector(
    const array& x,
    const array& w,
    const array& g,
    array& gx,
    array& gw,
    float eps,
    Stream s);

namespace fast {

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("LayerNorm::eval_gpu");
  const array& x = inputs[0];
  const array& w = inputs[1];
  const array& b = inputs[2];
  array& out = outputs[0];
  layer_norm_vector(x, w, b, out, eps_, stream());
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("LayerNormVJP::eval_gpu");
  const array& x = inputs[0];
  const array& w = inputs[1];
  const array& b = inputs[2];
  const array& g = inputs[3];
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];
  layer_norm_backward_vector(x, w, b, g, gx, gw, gb, eps_, stream());
}

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("RMSNorm::eval_gpu");
  const array& x = inputs[0];
  const array& w = inputs[1];
  array& out = outputs[0];

  if (uses_cudnn_norms()) {
    rms_norm_cudnn(x, w, out, eps_, stream());
  } else {
    rms_norm_vector(x, w, out, eps_, stream());
  }
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("RMSNormVJP::eval_gpu");
  const array& x = inputs[0];
  const array& w = inputs[1];
  const array& g = inputs[2];
  array& gx = outputs[0];
  array& gw = outputs[1];
  rms_norm_backward_vector(x, w, g, gx, gw, eps_, stream());
}

} // namespace fast

} // namespace mlx::core
