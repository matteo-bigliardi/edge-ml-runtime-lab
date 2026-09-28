#include "tinytensor/ops.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
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

TEST_CASE("matmul of non-square matrices", "[ops][matmul]") {
    // Non-square on every side, so a swapped index or a transposed operand
    // cannot produce the right numbers by accident.
    const Tensor a = Tensor::from_vector({2, 3}, {1.0F, 2.0F, 3.0F,
                                                  4.0F, 5.0F, 6.0F});
    const Tensor b = Tensor::from_vector({3, 4}, {1.0F, 0.0F, 2.0F, -1.0F,
                                                  0.0F, 1.0F, 1.0F, 2.0F,
                                                  3.0F, -1.0F, 0.0F, 1.0F});

    const Tensor product = tinytensor::matmul(a, b);

    CHECK(product.shape() == Shape{2, 4});
    CHECK(values(product) == std::vector<float>{10.0F, -1.0F, 4.0F, 6.0F,
                                                22.0F, -1.0F, 13.0F, 12.0F});
}

TEST_CASE("matmul by the identity leaves a matrix unchanged", "[ops][matmul]") {
    const Tensor a = Tensor::randn({3, 3}, 11);
    const Tensor identity = Tensor::from_vector({3, 3}, {1.0F, 0.0F, 0.0F,
                                                         0.0F, 1.0F, 0.0F,
                                                         0.0F, 0.0F, 1.0F});

    CHECK(values(tinytensor::matmul(a, identity)) == values(a));
    CHECK(values(tinytensor::matmul(identity, a)) == values(a));
}

TEST_CASE("matmul over an empty inner dimension gives zeros", "[ops][matmul]") {
    const Tensor product = tinytensor::matmul(Tensor::zeros({2, 0}), Tensor::zeros({0, 3}));

    CHECK(product.shape() == Shape{2, 3});
    CHECK(values(product) == std::vector<float>(6, 0.0F));
}

TEST_CASE("matmul rejects mismatched or non-2-D operands", "[ops][matmul]") {
    CHECK_THROWS_AS(tinytensor::matmul(Tensor::zeros({2, 3}), Tensor::zeros({2, 3})),
                    std::invalid_argument);
    CHECK_THROWS_AS(tinytensor::matmul(Tensor::zeros({3}), Tensor::zeros({3, 2})),
                    std::invalid_argument);
    CHECK_THROWS_AS(tinytensor::matmul(Tensor::zeros({2, 2, 2}), Tensor::zeros({2, 2})),
                    std::invalid_argument);
}

TEST_CASE("relu zeroes negatives and keeps the rest", "[ops][relu]") {
    const Tensor x = Tensor::from_vector({2, 2}, {-1.5F, 0.0F, 2.0F, -0.25F});

    const Tensor y = tinytensor::relu(x);

    CHECK(y.shape() == Shape{2, 2});
    CHECK(values(y) == std::vector<float>{0.0F, 0.0F, 2.0F, 0.0F});
}

TEST_CASE("relu lets NaN through", "[ops][relu]") {
    const Tensor x = Tensor::from_vector({1}, {std::numeric_limits<float>::quiet_NaN()});

    CHECK(std::isnan(tinytensor::relu(x).item()));
}

TEST_CASE("sum reduces every element to a scalar", "[ops][sum]") {
    const Tensor total = tinytensor::sum(Tensor::from_vector({2, 3}, {1.0F, 2.0F, 3.0F,
                                                                      4.0F, 5.0F, 6.0F}));

    CHECK(total.shape() == Shape{});
    CHECK(total.item() == 21.0F);
}

TEST_CASE("sum of an empty tensor is zero", "[ops][sum]") {
    CHECK(tinytensor::sum(Tensor::zeros({0, 4})).item() == 0.0F);
}

TEST_CASE("sum keeps small terms next to large ones", "[ops][sum]") {
    // In float, 1e8 + 1 rounds back to 1e8 (the spacing there is 8), so a
    // float accumulator returns 0. The double accumulator returns 1.
    const Tensor x = Tensor::from_vector({3}, {1e8F, 1.0F, -1e8F});

    CHECK(tinytensor::sum(x).item() == 1.0F);
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
