#include "tinytensor/no_grad.hpp"
#include "tinytensor/ops.hpp"
#include "tinytensor/tensor.hpp"

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

using tinytensor::NoGradGuard;
using tinytensor::Tensor;
using tinytensor::testing::values;

namespace {

Tensor parameter(Tensor tensor) {
    tensor.set_requires_grad(true);
    return tensor;
}

}  // namespace

TEST_CASE("factories make leaves that do not require a gradient", "[autograd]") {
    const Tensor x = Tensor::ones({2});

    CHECK_FALSE(x.requires_grad());
    CHECK_FALSE(x.grad().has_value());
}

TEST_CASE("backward through sum gives every element a gradient of one", "[autograd]") {
    const Tensor x = parameter(Tensor::from_vector({2, 2}, {1.0F, -2.0F, 3.0F, 4.0F}));

    tinytensor::sum(x).backward();

    REQUIRE(x.grad().has_value());
    CHECK(x.grad()->shape() == x.shape());
    CHECK(values(*x.grad()) == std::vector<float>(4, 1.0F));
}

TEST_CASE("a scalar leaf is its own loss", "[autograd]") {
    const Tensor x = parameter(Tensor::from_vector({}, {3.0F}));

    x.backward();

    CHECK(x.grad()->item() == 1.0F);
}

TEST_CASE("a tensor used twice receives both contributions", "[autograd]") {
    const Tensor x = parameter(Tensor::from_vector({3}, {1.0F, 2.0F, 3.0F}));

    tinytensor::sum(x + x).backward();

    CHECK(values(*x.grad()) == std::vector<float>(3, 2.0F));
}

TEST_CASE("a shared intermediate is propagated once, after all of its uses", "[autograd]") {
    // y feeds the loss along two paths. If backward reached y before both
    // contributions had arrived, x would see only part of the gradient.
    const Tensor x = parameter(Tensor::ones({2}));
    const Tensor y = x + x;

    tinytensor::sum(y + y).backward();

    CHECK(values(*x.grad()) == std::vector<float>(2, 4.0F));
}

TEST_CASE("mul sends each input the other one", "[autograd][mul]") {
    const Tensor x = parameter(Tensor::from_vector({3}, {1.0F, 2.0F, 3.0F}));
    const Tensor y = parameter(Tensor::from_vector({3}, {4.0F, 5.0F, 6.0F}));

    tinytensor::sum(x * y).backward();

    CHECK(values(*x.grad()) == values(y));
    CHECK(values(*y.grad()) == values(x));
}

TEST_CASE("the square of a tensor has gradient 2x", "[autograd][mul]") {
    const Tensor x = parameter(Tensor::from_vector({3}, {1.0F, -2.0F, 0.5F}));

    tinytensor::sum(x * x).backward();

    CHECK(values(*x.grad()) == std::vector<float>{2.0F, -4.0F, 1.0F});
}

TEST_CASE("the broadcast bias receives the gradient summed over rows", "[autograd]") {
    const Tensor matrix = parameter(Tensor::zeros({3, 2}));
    const Tensor bias = parameter(Tensor::zeros({2}));

    tinytensor::sum(matrix + bias).backward();

    CHECK(values(*matrix.grad()) == std::vector<float>(6, 1.0F));
    CHECK(values(*bias.grad()) == std::vector<float>(2, 3.0F));
}

TEST_CASE("only tensors that require a gradient receive one", "[autograd]") {
    const Tensor x = parameter(Tensor::ones({2}));
    const Tensor constant = Tensor::ones({2});

    const Tensor y = x + constant;
    tinytensor::sum(y).backward();

    CHECK(x.grad().has_value());
    CHECK_FALSE(constant.grad().has_value());
    // An operation's result requires a gradient but does not keep one.
    CHECK(y.requires_grad());
    CHECK_FALSE(y.grad().has_value());
}

TEST_CASE("operations on constants record no graph", "[autograd]") {
    const Tensor y = Tensor::ones({2}) + Tensor::ones({2});

    CHECK_FALSE(y.requires_grad());
    CHECK_THROWS_AS(tinytensor::sum(y).backward(), std::logic_error);
}

TEST_CASE("gradients accumulate across backward calls until zero_grad", "[autograd]") {
    Tensor x = parameter(Tensor::ones({2}));

    tinytensor::sum(x).backward();
    tinytensor::sum(x).backward();
    CHECK(values(*x.grad()) == std::vector<float>(2, 2.0F));

    x.zero_grad();
    CHECK_FALSE(x.grad().has_value());

    tinytensor::sum(x).backward();
    CHECK(values(*x.grad()) == std::vector<float>(2, 1.0F));
}

TEST_CASE("backward starts from a scalar", "[autograd]") {
    const Tensor x = parameter(Tensor::ones({2}));

    CHECK_THROWS_AS((x + x).backward(), std::invalid_argument);
}

TEST_CASE("only leaves can change whether they require a gradient", "[autograd]") {
    const Tensor x = parameter(Tensor::ones({2}));
    Tensor y = x + x;

    CHECK_THROWS_AS(y.set_requires_grad(false), std::logic_error);
}

TEST_CASE("clone is a detached leaf", "[autograd]") {
    const Tensor x = parameter(Tensor::ones({2}));
    tinytensor::sum(x).backward();
    const Tensor y = x + x;

    Tensor copy = y.clone();

    CHECK_FALSE(copy.requires_grad());
    CHECK_FALSE(x.clone().grad().has_value());
    // A leaf, unlike y: it may be made a parameter of its own.
    CHECK_NOTHROW(copy.set_requires_grad(true));
}

TEST_CASE("no_grad records nothing, and recording resumes after it", "[autograd][no_grad]") {
    const Tensor x = parameter(Tensor::ones({2}));

    {
        const NoGradGuard no_grad;
        const Tensor y = x + x;
        CHECK_FALSE(y.requires_grad());
    }

    CHECK((x + x).requires_grad());
}

TEST_CASE("no_grad guards nest", "[autograd][no_grad]") {
    CHECK(tinytensor::is_grad_enabled());
    {
        const NoGradGuard outer;
        {
            const NoGradGuard inner;
            CHECK_FALSE(tinytensor::is_grad_enabled());
        }
        // Leaving the inner guard restores the outer one's state, not "on".
        CHECK_FALSE(tinytensor::is_grad_enabled());
    }
    CHECK(tinytensor::is_grad_enabled());
}
