#pragma once

namespace tinytensor {

/// Whether operations currently record the graph. On by default, per thread.
[[nodiscard]] bool is_grad_enabled() noexcept;

/// Stops operations from recording the graph for as long as it lives: for
/// inference, and for updating parameters without making the update part of
/// their history. Restores the previous state when destroyed, so guards nest.
class NoGradGuard {
public:
    NoGradGuard() noexcept;
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

}  // namespace tinytensor
