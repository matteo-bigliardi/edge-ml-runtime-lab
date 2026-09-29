#include "tinytensor/ops.hpp"
#include "tinytensor/tensor.hpp"

#include "gradcheck.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using tinytensor::Tensor;
using tinytensor::testing::check_gradients;
using tinytensor::testing::weighted_sum;

using Inputs = std::vector<Tensor>;

TEST_CASE("gradient check: add", "[gradcheck][add]") {
    check_gradients([](const Inputs& in) { return weighted_sum(in[0] + in[1]); },
                    {Tensor::randn({3, 4}, 1), Tensor::randn({3, 4}, 2)});
}

TEST_CASE("gradient check: add with a broadcast bias, in both orders", "[gradcheck][add]") {
    const Inputs inputs{Tensor::randn({3, 4}, 3), Tensor::randn({4}, 4)};

    check_gradients([](const Inputs& in) { return weighted_sum(in[0] + in[1]); }, inputs);
    check_gradients([](const Inputs& in) { return weighted_sum(in[1] + in[0]); }, inputs);
}

TEST_CASE("gradient check: mul", "[gradcheck][mul]") {
    check_gradients([](const Inputs& in) { return weighted_sum(in[0] * in[1]); },
                    {Tensor::randn({2, 3}, 5), Tensor::randn({2, 3}, 6)});
}

TEST_CASE("gradient check: matmul", "[gradcheck][matmul]") {
    check_gradients([](const Inputs& in) { return weighted_sum(tinytensor::matmul(in[0], in[1])); },
                    {Tensor::randn({2, 3}, 9), Tensor::randn({3, 4}, 10)});
}

TEST_CASE("gradient check: relu", "[gradcheck][relu]") {
    // Every value is further from zero than the finite-difference step: at
    // the kink the two sides of the difference would straddle it, and the
    // numerical slope would be meaningless there.
    check_gradients([](const Inputs& in) { return weighted_sum(tinytensor::relu(in[0])); },
                    {Tensor::from_vector({2, 3}, {-1.5F, 0.3F, 2.0F, -0.2F, 0.7F, -0.9F})});
}

TEST_CASE("gradient check: sum", "[gradcheck][sum]") {
    check_gradients([](const Inputs& in) { return tinytensor::sum(in[0]); },
                    {Tensor::randn({2, 3}, 7)});
}

TEST_CASE("gradient check: a tensor reused along several paths", "[gradcheck]") {
    check_gradients([](const Inputs& in) { return weighted_sum(in[0] * in[0] + in[0]); },
                    {Tensor::randn({5}, 8)});
}

TEST_CASE("gradient check: a two-layer MLP, inputs included", "[gradcheck][mlp]") {
    const Tensor x = Tensor::randn({4, 3}, 11);
    const Tensor w1 = Tensor::randn({3, 5}, 12);
    const Tensor b1 = Tensor::randn({5}, 13);
    const Tensor w2 = Tensor::randn({5, 2}, 14);
    const Tensor b2 = Tensor::randn({2}, 15);

    // The check is only valid if no hidden pre-activation crosses relu's kink
    // when one element moves by the step. Moving x[i, p] or w1[p, j] shifts a
    // pre-activation by at most step * max(|x|, |w1|); moving b1 by step.
    // So every pre-activation must be further from zero than that.
    const auto largest = [](const Tensor& t) {
        float m = 0.0F;
        for (const float v : t.data()) {
            m = std::max(m, std::abs(v));
        }
        return m;
    };
    const float reach = 1e-2F * std::max({largest(x), largest(w1), 1.0F});
    for (const float h : (tinytensor::matmul(x, w1) + b1).data()) {
        REQUIRE(std::abs(h) > reach);
    }

    check_gradients(
        [](const Inputs& in) {
            const Tensor hidden = tinytensor::relu(tinytensor::matmul(in[0], in[1]) + in[2]);
            return weighted_sum(tinytensor::matmul(hidden, in[3]) + in[4]);
        },
        {x, w1, b1, w2, b2});
}
