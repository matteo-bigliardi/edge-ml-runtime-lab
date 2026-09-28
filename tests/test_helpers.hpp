#pragma once

#include "tinytensor/tensor.hpp"

#include <vector>

namespace tinytensor::testing {

/// The tensor's values as a vector, so a test can compare them with `==` and
/// Catch2 prints both sides when they differ.
inline std::vector<float> values(const Tensor& tensor) {
    const auto data = tensor.data();
    return {data.begin(), data.end()};
}

}  // namespace tinytensor::testing
