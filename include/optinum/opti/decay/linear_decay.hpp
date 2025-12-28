#pragma once

#include <cstddef>

namespace optinum::opti {

    /**
     * Linear decay - reduces learning rate linearly over time
     *
     * lr(t) = initial_lr * (1 - t / total_iterations)
     *
     * Or with min_lr: lr(t) = initial_lr - (initial_lr - min_lr) * t / total_iterations
     *
     * Properties:
     * - Simple, predictable decay
     * - Reaches min_lr exactly at total_iterations
     * - Good when you know the total training budget
     */
    struct LinearDecay {
        std::size_t total_iterations = 1000; ///< Total number of iterations
        double min_lr = 0.0;                 ///< Minimum learning rate at end
        double initial_lr = 0.0;             ///< Initial learning rate (captured on first call)
        bool initialized = false;            ///< Whether initial_lr has been captured

        /// Constructor with parameters
        explicit LinearDecay(std::size_t total = 1000, double min = 0.0) : total_iterations(total), min_lr(min) {}

        /// Update step size based on iteration
        template <typename T> void update(T &step_size, std::size_t iteration) noexcept {
            if (!initialized) {
                initial_lr = static_cast<double>(step_size);
                initialized = true;
            }

            if (iteration >= total_iterations) {
                step_size = static_cast<T>(min_lr);
                return;
            }

            double progress = static_cast<double>(iteration) / static_cast<double>(total_iterations);
            step_size = static_cast<T>(initial_lr - (initial_lr - min_lr) * progress);
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
