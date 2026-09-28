#pragma once

#include "tinytensor/tensor.hpp"

namespace tinytensor {

// Every operation returns a new tensor with storage of its own; the inputs are
// never modified. A shape the operation does not accept throws
// std::invalid_argument naming both shapes.

/// Element-wise sum of two tensors of the same shape. The one broadcast
/// supported is the bias of a linear layer: an [n, m] matrix and an [m]
/// vector, in either order, add the vector to every row.
[[nodiscard]] Tensor add(const Tensor& lhs, const Tensor& rhs);

/// Element-wise product. Shapes must match exactly.
[[nodiscard]] Tensor mul(const Tensor& lhs, const Tensor& rhs);

/// Matrix product of an [n, k] and a [k, m] matrix; both operands must be 2-D.
[[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs);

[[nodiscard]] Tensor operator+(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator*(const Tensor& lhs, const Tensor& rhs);

}  // namespace tinytensor
