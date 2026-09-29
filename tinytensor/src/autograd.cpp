#include "autograd.hpp"

#include "tinytensor/no_grad.hpp"

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tinytensor {
namespace {

// Per thread, as in PyTorch: a guard on one thread must not silently stop
// another thread from recording its graph.
thread_local bool grad_enabled = true;

/// Every tensor that backward() has to visit from `root`, each one after all
/// of its inputs. Only paths through tensors that require a gradient are
/// followed: the rest of the graph cannot receive one.
///
/// Iterative on purpose. A recursive walk is shorter, but its depth is the
/// depth of the graph, and a long chain of operations would overflow the stack.
std::vector<TensorImpl*> topological_order(TensorImpl* root) {
    std::vector<TensorImpl*> order;
    std::unordered_set<TensorImpl*> visited{root};
    // Each frame is a tensor and the index of the next input to look at.
    std::vector<std::pair<TensorImpl*, std::size_t>> stack{{root, 0}};

    while (!stack.empty()) {
        auto& [tensor, next_input] = stack.back();
        const Node* node = tensor->grad_fn.get();
        if (node != nullptr && next_input < node->inputs.size()) {
            TensorImpl* input = node->inputs[next_input].get();
            ++next_input;
            // Pushing may reallocate the stack, so the references above must
            // not be used after this line.
            if (input->requires_grad && visited.insert(input).second) {
                stack.emplace_back(input, 0);
            }
        } else {
            order.push_back(tensor);
            stack.pop_back();
        }
    }
    return order;
}

void accumulate(std::vector<float>& into, const std::vector<float>& addend) {
    for (std::size_t i = 0; i < into.size(); ++i) {
        into[i] += addend[i];
    }
}

void run_backward(TensorImpl* root) {
    const std::vector<TensorImpl*> order = topological_order(root);

    // Gradients still on their way, keyed by the tensor they belong to. A
    // tensor is processed only after every tensor that consumed it, so by the
    // time it is reached its entry holds the sum over all of its uses.
    std::unordered_map<TensorImpl*, std::vector<float>> pending;
    pending.emplace(root, std::vector<float>{1.0F});

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        TensorImpl* tensor = *it;
        auto entry = pending.find(tensor);
        std::vector<float> grad = std::move(entry->second);
        pending.erase(entry);

        if (tensor->grad_fn == nullptr) {
            if (tensor->grad == nullptr) {
                tensor->grad = std::make_shared<TensorImpl>(tensor->shape, std::move(grad));
            } else {
                accumulate(tensor->grad->storage, grad);
            }
            continue;
        }

        const Node& node = *tensor->grad_fn;
        InputGrads input_grads = node.backward(grad, node.inputs);
        for (std::size_t i = 0; i < node.inputs.size(); ++i) {
            TensorImpl* input = node.inputs[i].get();
            if (!input->requires_grad) {
                continue;
            }
            // The same tensor can be an input more than once (x * x): each
            // use contributes, and the contributions add up.
            auto [slot, inserted] = pending.try_emplace(input, std::move(input_grads[i]));
            if (!inserted) {
                accumulate(slot->second, input_grads[i]);
            }
        }
    }
}

}  // namespace

bool is_grad_enabled() noexcept {
    return grad_enabled;
}

NoGradGuard::NoGradGuard() noexcept : previous_(grad_enabled) {
    grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    grad_enabled = previous_;
}

void record(const Tensor& result, std::initializer_list<Tensor> inputs, BackwardFn backward) {
    if (!grad_enabled) {
        return;
    }
    bool any_requires_grad = false;
    for (const Tensor& input : inputs) {
        any_requires_grad = any_requires_grad || input.requires_grad();
    }
    if (!any_requires_grad) {
        return;
    }

    auto node = std::make_shared<Node>();
    for (const Tensor& input : inputs) {
        node->inputs.push_back(TensorAccess::impl(input));
    }
    node->backward = std::move(backward);

    const auto& impl = TensorAccess::impl(result);
    impl->requires_grad = true;
    impl->grad_fn = std::move(node);
}

bool Tensor::requires_grad() const noexcept {
    return impl_->requires_grad;
}

void Tensor::set_requires_grad(bool requires_grad) {
    if (impl_->grad_fn != nullptr) {
        throw std::logic_error(
            "set_requires_grad() only applies to leaves; an operation's result requires a "
            "gradient exactly when one of its inputs does");
    }
    impl_->requires_grad = requires_grad;
}

std::optional<Tensor> Tensor::grad() const {
    if (impl_->grad == nullptr) {
        return std::nullopt;
    }
    return Tensor(impl_->grad);
}

void Tensor::zero_grad() noexcept {
    impl_->grad.reset();
}

void Tensor::backward() const {
    if (!impl_->requires_grad) {
        throw std::logic_error(
            "backward() on a tensor that does not require a gradient: nothing it depends on "
            "was marked with set_requires_grad(true)");
    }
    if (impl_->storage.size() != 1) {
        throw std::invalid_argument("backward() starts from a scalar loss, got shape " +
                                    impl_->shape.to_string());
    }
    run_backward(impl_.get());
}

}  // namespace tinytensor
