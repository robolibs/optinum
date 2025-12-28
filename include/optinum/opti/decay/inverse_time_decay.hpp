#pragma once

#include <cstddef>

namespace optinum::opti {

    /**
     * Inverse time decay - reduces learning rate inversely with time
     *
     * lr(t) = initial_lr / (1 + decay_rate * t)
     *
     * Or with staircase: lr(t) = initial_lr / (1 + decay_rate * floor(t / decay_steps))
     *
     * Properties:
     * - Rapid initial decay, then slows down
     * - Never reaches zero
     * - Classic schedule from optimization theory
     * - Satisfies Robbins-Monro conditions for convergence
     */
    struct InverseTimeDecay {
        double decay_rate = 0.01;    ///< Decay rate coefficient
        std::size_t decay_steps = 1; ///< Steps between decay updates (1 = continuous)
        bool staircase = false;      ///< If true, decay in discrete steps
        double initial_lr = 0.0;     ///< Initial learning rate (captured on first call)
        bool initialized = false;    ///< Whether initial_lr has been captured

        /// Constructor with parameters
        explicit InverseTimeDecay(double rate = 0.01, std::size_t steps = 1, bool stairs = false)
            : decay_rate(rate), decay_steps(steps), staircase(stairs) {}

        /// Update step size based on iteration
        template <typename T> void update(T &step_size, std::size_t iteration) noexcept {
            if (!initialized) {
                initial_lr = static_cast<double>(step_size);
                initialized = true;
            }

            double t;
            if (staircase) {
                std::size_t steps = iteration / decay_steps;
                t = static_cast<double>(steps);
            } else {
                t = static_cast<double>(iteration) / static_cast<double>(decay_steps);
            }

            step_size = static_cast<T>(initial_lr / (1.0 + decay_rate * t));
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
