#include "tinytensor/tensor.hpp"

#include "tensor_impl.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinytensor {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) noexcept : impl_(std::move(impl)) {}

Tensor Tensor::zeros(Shape shape) {
    std::vector<float> values(to_index(shape.numel()), 0.0F);
    return from_vector(std::move(shape), std::move(values));
}

Tensor Tensor::ones(Shape shape) {
    std::vector<float> values(to_index(shape.numel()), 1.0F);
    return from_vector(std::move(shape), std::move(values));
}

Tensor Tensor::from_vector(Shape shape, std::vector<float> values) {
    // Every other factory goes through here, so this is the one check that
    // makes "storage size matches shape" true for every tensor that exists.
    if (values.size() != to_index(shape.numel())) {
        std::ostringstream message;
        message << "Shape " << shape.to_string() << " holds " << shape.numel()
                << " elements, got " << values.size() << " values";
        throw std::invalid_argument(message.str());
    }
    return Tensor(std::make_shared<TensorImpl>(std::move(shape), std::move(values)));
}

Tensor Tensor::randn(Shape shape, std::uint64_t seed) {
    // std::normal_distribution would be shorter, but the standard does not fix
    // its algorithm: libstdc++ and MSVC turn the same engine output into
    // different numbers. The engine itself is specified bit for bit, so the
    // transform on top of it is written here (Box-Muller).
    std::mt19937_64 engine(seed);
    const auto uniform = [&engine] {
        // The top 53 bits, shifted into (0, 1]: zero is excluded so that the
        // logarithm below stays finite.
        return (static_cast<double>(engine() >> 11U) + 1.0) * 0x1.0p-53;
    };

    std::vector<float> values(to_index(shape.numel()));
    for (std::size_t i = 0; i < values.size(); i += 2) {
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle = 2.0 * std::numbers::pi * uniform();
        values[i] = static_cast<float>(radius * std::cos(angle));
        if (i + 1 < values.size()) {
            values[i + 1] = static_cast<float>(radius * std::sin(angle));
        }
    }
    return from_vector(std::move(shape), std::move(values));
}

const Shape& Tensor::shape() const noexcept {
    return impl_->shape;
}

std::int64_t Tensor::numel() const noexcept {
    return impl_->shape.numel();
}

std::span<const float> Tensor::data() const noexcept {
    return impl_->storage;
}

std::span<float> Tensor::mutable_data() noexcept {
    return impl_->storage;
}

float Tensor::item() const {
    if (impl_->storage.size() != 1) {
        std::ostringstream message;
        message << "item() needs a tensor with exactly one element, got shape "
                << impl_->shape.to_string();
        throw std::invalid_argument(message.str());
    }
    return impl_->storage.front();
}

Tensor Tensor::clone() const {
    // Copies the values only: copying the whole impl would also copy the
    // graph node, and the clone would claim a history it does not have.
    return Tensor(std::make_shared<TensorImpl>(impl_->shape, impl_->storage));
}

}  // namespace tinytensor
