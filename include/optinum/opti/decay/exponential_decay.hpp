#pragma once

#include <cmath>
#include <cstddef>

namespace optinum::opti {

    /**
     * Exponential decay - smoothly reduces learning rate exponentially
     *
     * lr(t) = initial_lr * decay_rate^t
     *
     * Or equivalently: lr(t) = initial_lr * exp(-decay_rate * t)
     * (depending on how decay_rate is interpreted)
     *
     * This implementation uses: lr(t) = initial_lr * decay_rate^(t / decay_steps)
     *
     * Properties:
     * - Smooth, continuous decay
     * - Never reaches zero (asymptotic)
     * - Good for fine-tuning as training progresses
     */
    struct ExponentialDecay {
        double decay_rate = 0.96;      ///< Decay factor per decay_steps iterations
        std::size_t decay_steps = 100; ///< Number of iterations for one decay cycle
        double initial_lr = 0.0;       ///< Initial learning rate (captured on first call)
        bool initialized = false;      ///< Whether initial_lr has been captured

        /// Constructor with parameters
        explicit ExponentialDecay(double rate = 0.96, std::size_t steps = 100) : decay_rate(rate), decay_steps(steps) {}

        /// Update step size based on iteration
        template <typename T> void update(T &step_size, std::size_t iteration) noexcept {
            if (!initialized) {
                initial_lr = static_cast<double>(step_size);
                initialized = true;
            }

            double exponent = static_cast<double>(iteration) / static_cast<double>(decay_steps);
            step_size = static_cast<T>(initial_lr * std::pow(decay_rate, exponent));
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
