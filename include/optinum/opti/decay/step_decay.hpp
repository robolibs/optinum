#pragma once

#include <cmath>
#include <cstddef>

namespace optinum::opti {

    /**
     * Step decay - reduces learning rate by a factor at fixed intervals
     *
     * lr(t) = initial_lr * drop_rate^floor(t / step_size)
     *
     * Example: With drop_rate=0.5 and step_size=10:
     *   - Iterations 0-9:   lr = initial_lr
     *   - Iterations 10-19: lr = initial_lr * 0.5
     *   - Iterations 20-29: lr = initial_lr * 0.25
     *   - ...
     *
     * Common in training neural networks where you want discrete drops.
     */
    struct StepDecay {
        double drop_rate = 0.5;     ///< Factor to multiply lr by at each step
        std::size_t step_size = 10; ///< Number of iterations between drops
        double initial_lr = 0.0;    ///< Initial learning rate (captured on first call)
        bool initialized = false;   ///< Whether initial_lr has been captured

        /// Constructor with parameters
        explicit StepDecay(double drop = 0.5, std::size_t step = 10) : drop_rate(drop), step_size(step) {}

        /// Update step size based on iteration
        template <typename T> void update(T &step_size_val, std::size_t iteration) noexcept {
            if (!initialized) {
                initial_lr = static_cast<double>(step_size_val);
                initialized = true;
            }

            std::size_t num_drops = iteration / step_size;
            step_size_val = static_cast<T>(initial_lr * std::pow(drop_rate, static_cast<double>(num_drops)));
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
