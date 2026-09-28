#include "tinytensor/tensor.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

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

TEST_CASE("clone copies the data, not the handle", "[tensor]") {
    Tensor original = Tensor::from_vector({2}, {1.0F, 2.0F});
    Tensor copy = original.clone();

    copy.mutable_data()[0] = 9.0F;

    CHECK(copy.shape() == original.shape());
    CHECK(values(original) == std::vector<float>{1.0F, 2.0F});
    CHECK(values(copy) == std::vector<float>{9.0F, 2.0F});
}
