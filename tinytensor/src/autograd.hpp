#pragma once

#include "tinytensor/tensor.hpp"

#include "tensor_impl.hpp"

#include <initializer_list>

namespace tinytensor {

/// Called by every operation once its result is computed. If any input
/// requires a gradient, the result requires one too and gets a node that
/// knows how to differentiate it; otherwise nothing is recorded, and a
/// computation on constants costs no graph at all.
void record(const Tensor& result, std::initializer_list<Tensor> inputs, BackwardFn backward);

}  // namespace tinytensor
