#include "tinytensor/ops.hpp"
#include "tinytensor/tensor.hpp"

#include "gradcheck.hpp"

#include <catch2/catch_test_macros.hpp>

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
