#pragma once

#include <cmath>
#include <cstddef>

namespace optinum::opti {

    /**
     * Cosine annealing - smoothly reduces learning rate following a cosine curve
     *
     * lr(t) = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * t / T_max))
     *
     * Properties:
     * - Smooth decay following cosine curve
     * - Starts slow, accelerates in middle, slows at end
     * - Reaches min_lr exactly at T_max iterations
     * - Popular in deep learning (SGDR paper)
     *
     * Reference: Loshchilov & Hutter (2016)
     *   "SGDR: Stochastic Gradient Descent with Warm Restarts"
     */
    struct CosineAnnealing {
        std::size_t T_max = 1000; ///< Maximum number of iterations (one cycle)
        double min_lr = 0.0;      ///< Minimum learning rate at end of cycle
        double initial_lr = 0.0;  ///< Initial learning rate (captured on first call)
        bool initialized = false; ///< Whether initial_lr has been captured

        static constexpr double PI = 3.14159265358979323;

        /// Constructor with parameters
        explicit CosineAnnealing(std::size_t t_max = 1000, double min = 0.0) : T_max(t_max), min_lr(min) {}

        /// Update step size based on iteration
        template <typename T> void update(T &step_size, std::size_t iteration) noexcept {
            if (!initialized) {
                initial_lr = static_cast<double>(step_size);
                initialized = true;
            }

            // Clamp iteration to T_max
            std::size_t t = (iteration < T_max) ? iteration : T_max;

            double cos_term = std::cos(PI * static_cast<double>(t) / static_cast<double>(T_max));
            step_size = static_cast<T>(min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + cos_term));
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
