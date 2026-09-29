#pragma once

#include "tinytensor/no_grad.hpp"
#include "tinytensor/ops.hpp"
#include "tinytensor/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace tinytensor::testing {

using Loss = std::function<Tensor(const std::vector<Tensor>&)>;

/// sum(t * w) for fixed random weights w. A plain sum would send a gradient of
/// exactly one to every element, and a backward that routes gradients to the
/// wrong positions would still pass. Random weights make every position count.
inline Tensor weighted_sum(const Tensor& tensor, std::uint64_t seed = 1234) {
    return sum(tensor * Tensor::randn(tensor.shape(), seed));
}

/// Checks the gradients backward() computes against central differences,
/// (L(x + h) - L(x - h)) / 2h, for every element of every input.
///
/// `loss` maps the inputs to a scalar. The inputs are cloned first, so the
/// caller's tensors are left alone. The differences are taken under no_grad:
/// they only need values, and recording would build one graph per perturbed
/// element for nothing.
///
/// The step is large for float on purpose. Every operation here is at most
/// quadratic in any single element, and central differences are exact for
/// quadratics, so a large step costs no accuracy and keeps the loss's
/// rounding error small next to the difference being measured.
inline void check_gradients(const Loss& loss, const std::vector<Tensor>& inputs) {
    using Catch::Matchers::WithinAbs;
    using Catch::Matchers::WithinRel;

    constexpr float step = 1e-2F;
    constexpr double tolerance = 1e-3;

    std::vector<Tensor> params;
    for (const Tensor& input : inputs) {
        Tensor param = input.clone();
        param.set_requires_grad(true);
        params.push_back(param);
    }
    loss(params).backward();

    const NoGradGuard no_grad;
    for (std::size_t t = 0; t < params.size(); ++t) {
        const std::optional<Tensor> analytic = params[t].grad();
        REQUIRE(analytic.has_value());

        const auto data = params[t].mutable_data();
        for (std::size_t i = 0; i < data.size(); ++i) {
            const float original = data[i];
            // Rounded to float, the two points are not exactly 2h apart;
            // dividing by their actual distance removes that error.
            const float above = original + step;
            const float below = original - step;
            data[i] = above;
            const double loss_above = loss(params).item();
            data[i] = below;
            const double loss_below = loss(params).item();
            data[i] = original;

            const double numeric =
                (loss_above - loss_below) / (static_cast<double>(above) - static_cast<double>(below));
            INFO("input " << t << ", element " << i);
            CHECK_THAT(static_cast<double>(analytic->data()[i]),
                       WithinAbs(numeric, tolerance) || WithinRel(numeric, tolerance));
        }
    }
}

}  // namespace tinytensor::testing
