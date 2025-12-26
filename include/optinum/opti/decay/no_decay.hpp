#pragma once

#include <cstddef>

namespace optinum::opti {

    /**
     * No decay policy - keeps step size constant
     *
     * The step size (learning rate) remains fixed throughout optimization.
     * This is the simplest decay policy.
     */
    struct NoDecay {
        /**
         * Update step size (does nothing for NoDecay)
         *
         * @param step_size Current step size (unchanged)
         * @param iteration Current iteration number (unused)
         */
        template <typename T> void update(T &step_size, std::size_t iteration) const noexcept {
            // Do nothing - step size remains constant
            (void)step_size;
            (void)iteration;
        }

        /// Reset state (no state to reset)
        void reset() noexcept {}

        /// Initialize (no initialization needed)
        void initialize() noexcept {}
    };

} // namespace optinum::opti
