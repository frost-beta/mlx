// Copyright © 2025 Apple Inc.

#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

// Defined in layer_norm.cu file.
void dispatch_layer_norm(
    const array& x,
    const array& w,
    const array& b,
    array& out,
    float eps,
    Stream s);
void dispatch_layer_norm_backward(
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
void dispatch_rms_norm(
    const array& x,
    const array& w,
    array& out,
    float eps,
    Stream s);
void dispatch_rms_norm_backward(
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
  dispatch_layer_norm(x, w, b, out, eps_, stream());
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
  dispatch_layer_norm_backward(x, w, b, g, gx, gw, gb, eps_, stream());
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
  dispatch_rms_norm(x, w, out, eps_, stream());
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
  dispatch_rms_norm_backward(x, w, g, gx, gw, eps_, stream());
}

} // namespace fast

} // namespace mlx::core
