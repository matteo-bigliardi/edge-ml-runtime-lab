#include "tinytensor/ops.hpp"

#include <cstddef>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace tinytensor {
namespace {

[[noreturn]] void throw_incompatible(std::string_view op, const Tensor& lhs, const Tensor& rhs,
                                     std::string_view accepted) {
    std::ostringstream message;
    message << op << ": shapes " << lhs.shape().to_string() << " and "
            << rhs.shape().to_string() << " are not compatible (" << accepted << ")";
    throw std::invalid_argument(message.str());
}

template <typename Op>
Tensor elementwise(const Tensor& lhs, const Tensor& rhs, Op op) {
    const auto a = lhs.data();
    const auto b = rhs.data();
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = op(a[i], b[i]);
    }
    return Tensor::from_vector(lhs.shape(), std::move(out));
}

/// True for an [n, m] matrix paired with an [m] row vector.
bool is_row_broadcast(const Shape& matrix, const Shape& row) {
    return matrix.ndim() == 2 && row.ndim() == 1 && row.dim(0) == matrix.dim(1);
}

Tensor add_to_every_row(const Tensor& matrix, const Tensor& row) {
    const auto m = matrix.data();
    const auto v = row.data();
    const std::size_t cols = v.size();
    std::vector<float> out(m.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = m[i] + v[i % cols];
    }
    return Tensor::from_vector(matrix.shape(), std::move(out));
}

}  // namespace

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() == rhs.shape()) {
        return elementwise(lhs, rhs, std::plus<>{});
    }
    if (is_row_broadcast(lhs.shape(), rhs.shape())) {
        return add_to_every_row(lhs, rhs);
    }
    if (is_row_broadcast(rhs.shape(), lhs.shape())) {
        // Float addition is commutative, so swapping the operands gives the
        // same bits as adding in the order written.
        return add_to_every_row(rhs, lhs);
    }
    throw_incompatible("add", lhs, rhs, "expected equal shapes, or [n, m] with [m]");
}

Tensor mul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw_incompatible("mul", lhs, rhs, "expected equal shapes");
    }
    return elementwise(lhs, rhs, std::multiplies<>{});
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return add(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return mul(lhs, rhs);
}

}  // namespace tinytensor
