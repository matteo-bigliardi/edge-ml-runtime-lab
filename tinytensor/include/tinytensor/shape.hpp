#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace tinytensor {

/// The extents of a tensor, and the strides of the row-major contiguous layout
/// that TinyTensor always uses.
///
/// Strides are derived, never stored: every tensor in this library is
/// contiguous, so a stored stride vector could only ever disagree with the
/// dimensions. Non-contiguous views (transpose, slice) are out of scope, and
/// the day they are not, this is the class that has to change first.
class Shape {
public:
    /// The empty shape, which describes a scalar: zero dimensions, one element.
    Shape() = default;

    Shape(std::initializer_list<std::int64_t> dims);
    explicit Shape(std::vector<std::int64_t> dims);

    [[nodiscard]] std::size_t ndim() const noexcept { return dims_.size(); }

    /// The number of elements. A scalar holds one; any zero-sized dimension
    /// makes the whole tensor empty.
    [[nodiscard]] std::int64_t numel() const noexcept;

    [[nodiscard]] const std::vector<std::int64_t>& dims() const noexcept { return dims_; }

    /// Throws std::out_of_range past the last dimension.
    [[nodiscard]] std::int64_t dim(std::size_t index) const;

    /// Element counts to skip per dimension, outermost first.
    [[nodiscard]] std::vector<std::int64_t> strides() const;

    /// Readable form, e.g. "(2, 3)". For assertion messages and debugging.
    [[nodiscard]] std::string to_string() const;

    friend bool operator==(const Shape& lhs, const Shape& rhs) noexcept {
        return lhs.dims_ == rhs.dims_;
    }

private:
    std::vector<std::int64_t> dims_;
};

}  // namespace tinytensor
