#include "tinytensor/tensor.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using tinytensor::Shape;
using tinytensor::Tensor;
using tinytensor::testing::values;

TEST_CASE("zeros and ones fill every element", "[tensor]") {
    const Tensor zeros = Tensor::zeros({2, 3});
    const Tensor ones = Tensor::ones({4});

    CHECK(zeros.shape() == Shape{2, 3});
    CHECK(zeros.numel() == 6);
    CHECK(values(zeros) == std::vector<float>(6, 0.0F));
    CHECK(values(ones) == std::vector<float>(4, 1.0F));
}

TEST_CASE("from_vector keeps values in row-major order", "[tensor]") {
    const Tensor tensor = Tensor::from_vector({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

    CHECK(tensor.shape() == Shape{2, 2});
    CHECK(values(tensor) == std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
}

TEST_CASE("from_vector rejects a value count that does not match the shape", "[tensor]") {
    CHECK_THROWS_AS(Tensor::from_vector({2, 2}, {1.0F, 2.0F, 3.0F}), std::invalid_argument);
    CHECK_THROWS_AS(Tensor::from_vector({2}, {}), std::invalid_argument);
}

TEST_CASE("an empty shape is a scalar and a zero extent is an empty tensor", "[tensor]") {
    const Tensor scalar = Tensor::from_vector({}, {7.0F});
    const Tensor empty = Tensor::zeros({3, 0});

    CHECK(scalar.numel() == 1);
    CHECK(scalar.item() == 7.0F);
    CHECK(empty.numel() == 0);
    CHECK(empty.data().empty());
}

TEST_CASE("item needs exactly one element", "[tensor]") {
    CHECK(Tensor::ones({1, 1}).item() == 1.0F);
    CHECK_THROWS_AS(Tensor::ones({2}).item(), std::invalid_argument);
    CHECK_THROWS_AS(Tensor::zeros({0}).item(), std::invalid_argument);
}

TEST_CASE("copying a tensor shares its storage", "[tensor]") {
    Tensor original = Tensor::zeros({3});
    Tensor alias = original;

    alias.mutable_data()[1] = 5.0F;

    CHECK(values(original) == std::vector<float>{0.0F, 5.0F, 0.0F});
}

TEST_CASE("randn reproduces a reference computed outside C++", "[tensor][randn]") {
    // Produced by an independent Python implementation of mt19937_64 and the
    // same Box-Muller transform. Matching it on both CI jobs is what shows the
    // sequence does not depend on the standard library. The tolerance only
    // absorbs last-bit differences between the platforms' log/sin/cos.
    const std::vector<float> expected{-0.48121771F, -0.57453686F, 0.49458385F, 0.57012153F,
                                      0.37455428F};

    const Tensor sample = Tensor::randn({5}, 42);

    REQUIRE(sample.numel() == 5);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK_THAT(sample.data()[i], WithinAbs(expected[i], 1e-6));
    }
}

TEST_CASE("randn is a function of its seed", "[tensor][randn]") {
    CHECK(values(Tensor::randn({2, 3}, 7)) == values(Tensor::randn({2, 3}, 7)));
    CHECK(values(Tensor::randn({2, 3}, 7)) != values(Tensor::randn({2, 3}, 8)));
}

TEST_CASE("randn samples have zero mean and unit variance", "[tensor][randn]") {
    const Tensor sample = Tensor::randn({100'000}, 1);

    double sum = 0.0;
    double sum_of_squares = 0.0;
    for (const float value : sample.data()) {
        sum += value;
        sum_of_squares += static_cast<double>(value) * value;
    }
    const double count = static_cast<double>(sample.numel());
    const double mean = sum / count;

    // Both bounds sit between four and five standard errors: loose enough never
    // to flake on a correct generator, tight enough to catch a wrong scale.
    CHECK_THAT(mean, WithinAbs(0.0, 0.015));
    CHECK_THAT(sum_of_squares / count - mean * mean, WithinAbs(1.0, 0.02));
}

TEST_CASE("clone copies the data, not the handle", "[tensor]") {
    Tensor original = Tensor::from_vector({2}, {1.0F, 2.0F});
    Tensor copy = original.clone();

    copy.mutable_data()[0] = 9.0F;

    CHECK(copy.shape() == original.shape());
    CHECK(values(original) == std::vector<float>{1.0F, 2.0F});
    CHECK(values(copy) == std::vector<float>{9.0F, 2.0F});
}
