#include "tinytensor/shape.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using tinytensor::Shape;

TEST_CASE("a default shape describes a scalar", "[shape]") {
    const Shape shape;

    CHECK(shape.ndim() == 0);
    CHECK(shape.numel() == 1);
    CHECK(shape.strides().empty());
    CHECK(shape.to_string() == "()");
}

TEST_CASE("dimensions and element count", "[shape]") {
    const Shape shape{2, 3, 4};

    CHECK(shape.ndim() == 3);
    CHECK(shape.numel() == 24);
    CHECK(shape.dim(0) == 2);
    CHECK(shape.dim(2) == 4);
    CHECK(shape.to_string() == "(2, 3, 4)");
}

TEST_CASE("a zero-sized dimension empties the tensor", "[shape]") {
    const Shape shape{2, 0, 4};

    CHECK(shape.numel() == 0);
}

TEST_CASE("strides describe a row-major contiguous layout", "[shape]") {
    // The last dimension is the one that moves by a single element; every
    // outer stride is the product of the dimensions inside it.
    CHECK(Shape{2, 3, 4}.strides() == std::vector<std::int64_t>{12, 4, 1});
    CHECK(Shape{5}.strides() == std::vector<std::int64_t>{1});
}

TEST_CASE("shapes compare by dimensions", "[shape]") {
    CHECK(Shape{2, 3} == Shape{2, 3});
    CHECK(Shape{2, 3} != Shape{3, 2});
    CHECK(Shape{2, 3} != Shape{2, 3, 1});
}

TEST_CASE("a negative dimension is rejected at construction", "[shape]") {
    CHECK_THROWS_AS(Shape{-1}, std::invalid_argument);
    CHECK_THROWS_AS((Shape{2, -3}), std::invalid_argument);
}

TEST_CASE("indexing past the last dimension throws", "[shape]") {
    const Shape shape{2, 3};

    CHECK_THROWS_AS(shape.dim(2), std::out_of_range);
    CHECK_THROWS_AS(Shape{}.dim(0), std::out_of_range);
}
