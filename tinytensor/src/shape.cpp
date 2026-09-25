#include "tinytensor/shape.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinytensor {
namespace {

void validate(const std::vector<std::int64_t>& dims) {
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] < 0) {
            std::ostringstream message;
            message << "Shape dimension " << i << " is negative (" << dims[i] << ")";
            throw std::invalid_argument(message.str());
        }
    }
}

}  // namespace

Shape::Shape(std::initializer_list<std::int64_t> dims)
    : Shape(std::vector<std::int64_t>(dims)) {}

Shape::Shape(std::vector<std::int64_t> dims) : dims_(std::move(dims)) {
    validate(dims_);
}

std::int64_t Shape::numel() const noexcept {
    // Unchecked on purpose: a shape whose product overflows int64 describes a
    // tensor that cannot be allocated anyway. Allocation is where the failure
    // belongs, and where it will be reported.
    std::int64_t count = 1;
    for (const std::int64_t extent : dims_) {
        count *= extent;
    }
    return count;
}

std::int64_t Shape::dim(std::size_t index) const {
    if (index >= dims_.size()) {
        std::ostringstream message;
        message << "Dimension " << index << " is out of range for shape " << to_string();
        throw std::out_of_range(message.str());
    }
    return dims_[index];
}

std::vector<std::int64_t> Shape::strides() const {
    std::vector<std::int64_t> result(dims_.size());
    std::int64_t stride = 1;
    for (std::size_t i = dims_.size(); i-- > 0;) {
        result[i] = stride;
        stride *= dims_[i];
    }
    return result;
}

std::string Shape::to_string() const {
    std::ostringstream out;
    out << '(';
    for (std::size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << dims_[i];
    }
    out << ')';
    return out.str();
}

}  // namespace tinytensor
