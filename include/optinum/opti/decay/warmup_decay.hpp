#pragma once

#include <cstddef>

namespace optinum::opti {

    /**
     * Warmup decay - linearly increases learning rate, then applies another decay
     *
     * For t < warmup_steps:
     *   lr(t) = initial_lr * t / warmup_steps
     *
     * For t >= warmup_steps:
     *   lr(t) = initial_lr (constant, or can be combined with other decay)
     *
     * Properties:
     * - Prevents large gradients early in training
     * - Helps stabilize training with large batch sizes
     * - Common in transformer training
     *
     * Reference: Goyal et al. (2017)
     *   "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
     */
    struct WarmupDecay {
        std::size_t warmup_steps = 100; ///< Number of warmup iterations
        double initial_lr = 0.0;        ///< Target learning rate after warmup
        bool initialized = false;       ///< Whether initial_lr has been captured

        /// Constructor with parameters
        explicit WarmupDecay(std::size_t warmup = 100) : warmup_steps(warmup) {}

        /// Update step size based on iteration
        template <typename T> void update(T &step_size, std::size_t iteration) noexcept {
            if (!initialized) {
                initial_lr = static_cast<double>(step_size);
                initialized = true;
            }

            if (iteration < warmup_steps && warmup_steps > 0) {
                // Linear warmup
                double progress = static_cast<double>(iteration + 1) / static_cast<double>(warmup_steps);
                step_size = static_cast<T>(initial_lr * progress);
            } else {
                // After warmup, keep constant (or combine with another decay)
                step_size = static_cast<T>(initial_lr);
            }
        }

        /// Reset state
        void reset() noexcept {
            initialized = false;
            initial_lr = 0.0;
        }

        /// Initialize
        void initialize() noexcept { reset(); }
    };

} // namespace optinum::opti
