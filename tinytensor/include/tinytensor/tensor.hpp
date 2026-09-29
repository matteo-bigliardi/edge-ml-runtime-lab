#pragma once

#include "tinytensor/shape.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace tinytensor {

struct TensorImpl;
struct TensorAccess;

/// A handle to a contiguous row-major block of floats and its shape.
///
/// Copying a Tensor copies the handle, not the data: both copies see the same
/// storage, as in PyTorch. `clone()` is the way to get independent data. The
/// handle is also what the autograd graph holds on to: a result keeps its
/// inputs alive for as long as its gradient may still be needed, with no
/// ownership to manage by hand.
///
/// There is no empty state: every Tensor refers to valid storage whose size
/// matches its shape.
class Tensor {
public:
    [[nodiscard]] static Tensor zeros(Shape shape);
    [[nodiscard]] static Tensor ones(Shape shape);

    /// Takes `values` in row-major order. Throws std::invalid_argument when
    /// their count does not match the shape.
    [[nodiscard]] static Tensor from_vector(Shape shape, std::vector<float> values);

    /// Standard normal samples. The same seed gives the same values with any
    /// standard library, so a seeded test means the same thing on every CI job.
    [[nodiscard]] static Tensor randn(Shape shape, std::uint64_t seed);

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] std::int64_t numel() const noexcept;

    [[nodiscard]] std::span<const float> data() const noexcept;

    /// Writes through to every handle that shares this storage.
    [[nodiscard]] std::span<float> mutable_data() noexcept;

    /// The single value of a one-element tensor. Throws std::invalid_argument
    /// for any other size.
    [[nodiscard]] float item() const;

    /// A deep copy of the values, detached from any graph: a new leaf that
    /// does not require a gradient.
    [[nodiscard]] Tensor clone() const;

    // Autograd. A leaf is a tensor made by a factory; a tensor made by an
    // operation on inputs that require a gradient records how to
    // differentiate itself, and requires one too.

    [[nodiscard]] bool requires_grad() const noexcept;

    /// Marks a leaf as something backward() should compute a gradient for.
    /// Throws std::logic_error on an operation's result, whose need for a
    /// gradient follows from its inputs.
    void set_requires_grad(bool requires_grad);

    /// What backward() has accumulated for this tensor so far, or nothing if
    /// no backward has reached it. Only leaves keep a gradient, as in PyTorch.
    [[nodiscard]] std::optional<Tensor> grad() const;

    /// Drops the accumulated gradient, as PyTorch's zero_grad() does by default.
    void zero_grad() noexcept;

    /// Back-propagates from this scalar through the graph that produced it and
    /// adds d(this)/d(leaf) to the gradient of every leaf that requires one.
    /// Gradients accumulate across calls until zero_grad(). Throws
    /// std::logic_error if this tensor does not require a gradient, and
    /// std::invalid_argument if it holds more than one element.
    void backward() const;

private:
    friend struct TensorAccess;

    explicit Tensor(std::shared_ptr<TensorImpl> impl) noexcept;

    std::shared_ptr<TensorImpl> impl_;
};

}  // namespace tinytensor
