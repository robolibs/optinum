#pragma once

#include <cmath>
#include <cstddef>

namespace optinum::opti {

    /**
     * Polynomial decay - reduces learning rate following a polynomial curve
     *
     * lr(t) = (initial_lr - end_lr) * (1 - t / total_steps)^power + end_lr
     *
     * Properties:
     * - Flexible decay shape controlled by power parameter
     * - power = 1: linear decay
     * - power = 2: quadratic decay (slower start, faster end)
     * - power = 0.5: square root decay (faster start, slower end)
     * - Reaches end_lr exactly at total_steps
     *
     * Reference: Used in TensorFlow and many deep learning frameworks
     */
    struct PolynomialDecay {
        std::size_t total_steps = 1000; ///< Total number of iterations
        double end_lr = 0.0001;         ///< Final learning rate
        double power = 1.0;             ///< Polynomial power (1.0 = linear)
        bool cycle = false;             ///< If true, restart from initial_lr after total_steps
        double initial_lr = 0.0;        ///< Initial learning rate (captured on first call)
        bool initialized = false;       ///< Whether initial_lr has been captured

        /// Constructor with parameters
        explicit PolynomialDecay(std::size_t total = 1000, double end = 0.0001, double p = 1.0, bool cyc = false)
            : total_steps(total), end_lr(end), power(p), cycle(cyc) {}

        /// Update step size based on iteration
        template <typename T> void update(T &step_size, std::size_t iteration) noexcept {
            if (!initialized) {
                initial_lr = static_cast<double>(step_size);
                initialized = true;
            }

            std::size_t step = iteration;
            if (cycle) {
                step = iteration % total_steps;
            } else if (iteration >= total_steps) {
                step_size = static_cast<T>(end_lr);
                return;
            }

            double progress = static_cast<double>(step) / static_cast<double>(total_steps);
            double decay_factor = std::pow(1.0 - progress, power);
            step_size = static_cast<T>((initial_lr - end_lr) * decay_factor + end_lr);
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
