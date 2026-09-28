#pragma once

#include "tinytensor/shape.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace tinytensor {

struct TensorImpl;

/// A handle to a contiguous row-major block of floats and its shape.
///
/// Copying a Tensor copies the handle, not the data: both copies see the same
/// storage, as in PyTorch. `clone()` is the way to get independent data. The
/// handle is what the autograd graph will hold on to, so a tensor stays alive
/// for as long as anything still needs it, with no ownership to manage by hand.
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

    /// A deep copy: same shape and values, storage of its own.
    [[nodiscard]] Tensor clone() const;

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) noexcept;

    std::shared_ptr<TensorImpl> impl_;
};

}  // namespace tinytensor
