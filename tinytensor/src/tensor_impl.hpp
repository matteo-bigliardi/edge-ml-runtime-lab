#pragma once

#include "tinytensor/shape.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinytensor {

/// What a Tensor handle points to. Private to the library: the public header
/// only forward-declares it, so the autograd state that will live here later
/// does not leak into the API.
struct TensorImpl {
    Shape shape;
    std::vector<float> storage;
};

/// Shape guarantees every extent is non-negative, so this conversion cannot
/// wrap. It exists so that the one place where int64 extents become size_t
/// indices is named, instead of being a cast scattered through every loop.
[[nodiscard]] inline std::size_t to_index(std::int64_t extent) noexcept {
    return static_cast<std::size_t>(extent);
}

}  // namespace tinytensor
