#pragma once

#include "tinytensor/shape.hpp"
#include "tinytensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace tinytensor {

struct TensorImpl;

using TensorInputs = std::vector<std::shared_ptr<TensorImpl>>;

/// One gradient per input, in the order the inputs were recorded. An input
/// that does not require a gradient gets an empty vector, so the backward
/// function can skip work nobody will read.
using InputGrads = std::vector<std::vector<float>>;

/// Turns the gradient of an operation's output into the gradients of its
/// inputs. It receives the inputs instead of capturing them: a closure that
/// captured tensors could capture the output it is attached to, and that
/// would be a reference cycle. Capturing nothing but values rules it out.
using BackwardFn =
    std::function<InputGrads(std::span<const float> grad_output, const TensorInputs& inputs)>;

/// What an operation leaves on its result: where it came from and how to
/// differentiate it.
struct Node {
    TensorInputs inputs;
    BackwardFn backward;
};

/// What a Tensor handle points to. Private to the library: the public header
/// only forward-declares it, so the autograd state does not leak into the API.
struct TensorImpl {
    TensorImpl(Shape shape_, std::vector<float> storage_)
        : shape(std::move(shape_)), storage(std::move(storage_)) {}

    Shape shape;
    std::vector<float> storage;

    bool requires_grad = false;

    /// Leaves only. Null until a backward reaches this tensor.
    std::shared_ptr<TensorImpl> grad;

    /// Null for leaves. Holds the inputs, so a result keeps its whole history
    /// alive for as long as it lives itself.
    std::shared_ptr<Node> grad_fn;
};

/// The library's way past Tensor's private members. Tensor befriends it, so
/// code inside the library reaches the implementation while the public API
/// keeps it hidden.
struct TensorAccess {
    [[nodiscard]] static const std::shared_ptr<TensorImpl>& impl(const Tensor& tensor) noexcept {
        return tensor.impl_;
    }

    [[nodiscard]] static Tensor wrap(std::shared_ptr<TensorImpl> impl) noexcept {
        return Tensor(std::move(impl));
    }
};

/// Shape guarantees every extent is non-negative, so this conversion cannot
/// wrap. It exists so that the one place where int64 extents become size_t
/// indices is named, instead of being a cast scattered through every loop.
[[nodiscard]] inline std::size_t to_index(std::int64_t extent) noexcept {
    return static_cast<std::size_t>(extent);
}

}  // namespace tinytensor
