#include "tinytensor/ops.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

using tinytensor::Shape;
using tinytensor::Tensor;
using tinytensor::testing::values;

TEST_CASE("add sums tensors of the same shape", "[ops][add]") {
    const Tensor a = Tensor::from_vector({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    const Tensor b = Tensor::from_vector({2, 2}, {10.0F, 20.0F, 30.0F, 40.0F});

    const Tensor sum = tinytensor::add(a, b);

    CHECK(sum.shape() == Shape{2, 2});
    CHECK(values(sum) == std::vector<float>{11.0F, 22.0F, 33.0F, 44.0F});
    CHECK(values(a + b) == values(sum));
}

TEST_CASE("add broadcasts a bias vector over every row", "[ops][add]") {
    const Tensor matrix = Tensor::from_vector({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const Tensor bias = Tensor::from_vector({3}, {10.0F, 20.0F, 30.0F});
    const std::vector<float> expected{11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F};

    CHECK((matrix + bias).shape() == Shape{2, 3});
    CHECK(values(matrix + bias) == expected);
    CHECK(values(bias + matrix) == expected);
}

TEST_CASE("add rejects shapes outside the supported broadcast", "[ops][add]") {
    const Tensor matrix = Tensor::zeros({2, 3});

    // A vector matching the rows instead of the columns.
    CHECK_THROWS_AS(matrix + Tensor::zeros({2}), std::invalid_argument);
    // General broadcasting is out of scope, even where PyTorch would accept it.
    CHECK_THROWS_AS(matrix + Tensor::zeros({1, 3}), std::invalid_argument);
    CHECK_THROWS_AS(matrix + Tensor::zeros({}), std::invalid_argument);
    CHECK_THROWS_AS(Tensor::zeros({2, 3, 4}) + Tensor::zeros({4}), std::invalid_argument);
}

TEST_CASE("mul multiplies tensors of the same shape", "[ops][mul]") {
    const Tensor a = Tensor::from_vector({3}, {1.0F, -2.0F, 3.0F});
    const Tensor b = Tensor::from_vector({3}, {4.0F, 5.0F, -6.0F});

    CHECK(values(tinytensor::mul(a, b)) == std::vector<float>{4.0F, -10.0F, -18.0F});
    CHECK(values(a * b) == values(tinytensor::mul(a, b)));
}

TEST_CASE("mul does not broadcast", "[ops][mul]") {
    CHECK_THROWS_AS(Tensor::zeros({2, 3}) * Tensor::zeros({3}), std::invalid_argument);
    CHECK_THROWS_AS(Tensor::zeros({2, 3}) * Tensor::zeros({3, 2}), std::invalid_argument);
}

TEST_CASE("the result of an operation owns new storage", "[ops]") {
    const Tensor a = Tensor::ones({2});
    const Tensor b = Tensor::ones({2});

    Tensor sum = a + b;
    sum.mutable_data()[0] = 100.0F;

    CHECK(values(a) == std::vector<float>{1.0F, 1.0F});
    CHECK(values(b) == std::vector<float>{1.0F, 1.0F});
}

TEST_CASE("element-wise operations accept empty tensors", "[ops]") {
    CHECK((Tensor::zeros({0}) + Tensor::zeros({0})).numel() == 0);
    CHECK((Tensor::zeros({0, 3}) + Tensor::zeros({3})).shape() == Shape{0, 3});
    CHECK((Tensor::zeros({2, 0}) + Tensor::zeros({0})).shape() == Shape{2, 0});
    CHECK((Tensor::zeros({2, 0}) * Tensor::zeros({2, 0})).numel() == 0);
}
