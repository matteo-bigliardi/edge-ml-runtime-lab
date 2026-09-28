#include "tinytensor/tensor.hpp"

#include "tensor_impl.hpp"

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
    return Tensor(std::make_shared<TensorImpl>(TensorImpl{std::move(shape), std::move(values)}));
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
    return Tensor(std::make_shared<TensorImpl>(*impl_));
}

}  // namespace tinytensor
