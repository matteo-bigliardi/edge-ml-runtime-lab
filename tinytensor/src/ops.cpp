#include "tinytensor/ops.hpp"

#include "autograd.hpp"
#include "tensor_impl.hpp"

#include <cstddef>
#include <functional>
#include <span>
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

/// The gradient of a vector that was added to every row: each row's gradient
/// flows back to the same vector, so they sum.
std::vector<float> sum_over_rows(std::span<const float> grad, std::size_t cols) {
    std::vector<float> out(cols, 0.0F);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        out[i % cols] += grad[i];
    }
    return out;
}

Tensor add_values(const Tensor& lhs, const Tensor& rhs) {
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

}  // namespace

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    Tensor result = add_values(lhs, rhs);
    // d(a + b)/da = 1, so an input shaped like the output gets the output's
    // gradient unchanged. The broadcast bias needs it summed over the rows.
    record(result, {lhs, rhs},
           [out_shape = result.shape()](std::span<const float> grad, const TensorInputs& inputs) {
               InputGrads grads(inputs.size());
               for (std::size_t i = 0; i < inputs.size(); ++i) {
                   if (!inputs[i]->requires_grad) {
                       continue;
                   }
                   grads[i] = inputs[i]->shape == out_shape
                                  ? std::vector<float>(grad.begin(), grad.end())
                                  : sum_over_rows(grad, inputs[i]->storage.size());
               }
               return grads;
           });
    return result;
}

Tensor mul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw_incompatible("mul", lhs, rhs, "expected equal shapes");
    }
    Tensor result = elementwise(lhs, rhs, std::multiplies<>{});
    // d(a * b)/da = b and d(a * b)/db = a: each input gets the output's
    // gradient scaled by the other input. For x * x both terms land on x.
    record(result, {lhs, rhs}, [](std::span<const float> grad, const TensorInputs& inputs) {
        InputGrads grads(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i]->requires_grad) {
                continue;
            }
            const std::vector<float>& other = inputs[1 - i]->storage;
            grads[i].resize(grad.size());
            for (std::size_t j = 0; j < grad.size(); ++j) {
                grads[i][j] = grad[j] * other[j];
            }
        }
        return grads;
    });
    return result;
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    const Shape& a_shape = lhs.shape();
    const Shape& b_shape = rhs.shape();
    if (a_shape.ndim() != 2 || b_shape.ndim() != 2 || a_shape.dim(1) != b_shape.dim(0)) {
        throw_incompatible("matmul", lhs, rhs, "expected [n, k] and [k, m]");
    }

    const std::size_t n = to_index(a_shape.dim(0));
    const std::size_t k = to_index(a_shape.dim(1));
    const std::size_t m = to_index(b_shape.dim(1));
    const auto a = lhs.data();
    const auto b = rhs.data();
    std::vector<float> out(n * m);

    // The textbook loop order: one dot product per output element, walking
    // rhs down a column with stride m. It is naive on purpose, as the baseline
    // the optimised variant in the benchmarks is measured against. It
    // accumulates in float, like the float kernels it is compared with.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            float acc = 0.0F;
            for (std::size_t p = 0; p < k; ++p) {
                acc += a[i * k + p] * b[p * m + j];
            }
            out[i * m + j] = acc;
        }
    }
    return Tensor::from_vector({a_shape.dim(0), b_shape.dim(1)}, std::move(out));
}

Tensor relu(const Tensor& input) {
    const auto x = input.data();
    std::vector<float> out(x.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        // Written as "negative becomes zero" rather than std::max(x, 0):
        // a NaN compares false, so it passes through instead of turning
        // into a silent zero.
        out[i] = x[i] < 0.0F ? 0.0F : x[i];
    }
    Tensor result = Tensor::from_vector(input.shape(), std::move(out));
    // The slope is 1 where the input was positive and 0 where it was not. At
    // exactly zero relu has no derivative; 0 is the convention PyTorch uses.
    record(result, {input}, [](std::span<const float> grad, const TensorInputs& inputs) {
        const std::vector<float>& x = inputs[0]->storage;
        std::vector<float> dx(grad.size());
        for (std::size_t i = 0; i < dx.size(); ++i) {
            dx[i] = x[i] > 0.0F ? grad[i] : 0.0F;
        }
        return InputGrads{std::move(dx)};
    });
    return result;
}

Tensor sum(const Tensor& input) {
    // Accumulated in double: a float accumulator stops absorbing small terms
    // once the running total is large, and this sum is what a loss is made
    // of. The gradient checks difference two nearby losses, so an error here
    // would show up as a wrong gradient.
    double total = 0.0;
    for (const float value : input.data()) {
        total += value;
    }
    Tensor result = Tensor::from_vector({}, {static_cast<float>(total)});
    // Every element enters the total with weight one, so each receives the
    // total's gradient as it is.
    record(result, {input}, [](std::span<const float> grad, const TensorInputs& inputs) {
        return InputGrads{std::vector<float>(inputs[0]->storage.size(), grad[0])};
    });
    return result;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return add(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return mul(lhs, rhs);
}

}  // namespace tinytensor
