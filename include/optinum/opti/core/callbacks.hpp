#pragma once

#include <optinum/opti/core/types.hpp>

#include <datapod/matrix/vector.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Base callback interface for optimization monitoring
     *
     * Callbacks can be used to:
     * - Monitor optimization progress
     * - Log iteration information
     * - Implement early stopping
     * - Save checkpoints
     *
     * Example:
     * @code
     * struct MyCallback {
     *     template <typename T, std::size_t N>
     *     bool on_iteration(const IterationInfo<T>& info,
     *                       const dp::mat::vector<T, N>& x) {
     *         std::cout << "Iter " << info.iteration
     *                   << ", obj = " << info.objective << "\n";
     *         return false; // continue optimization
     *     }
     * };
     * @endcode
     */

    /**
     * Default callback that does nothing
     */
    struct NoCallback {
        /// Called at the beginning of optimization
        template <typename T, std::size_t N> void on_begin(const dp::mat::vector<T, N> &x0) const noexcept { (void)x0; }

        /// Called after each iteration
        /// @return true to stop optimization, false to continue
        template <typename T, std::size_t N>
        bool on_iteration(const IterationInfo<T> &info, const dp::mat::vector<T, N> &x) const noexcept {
            (void)info;
            (void)x;
            return false; // Continue
        }

        /// Called when optimization completes
        template <typename T, std::size_t N> void on_end(const OptimizationResult<T, N> &result) const noexcept {
            (void)result;
        }
    };

    /**
     * Simple logging callback that prints iteration info
     */
    struct LogCallback {
        std::size_t print_every = 1; ///< Print every N iterations

        explicit LogCallback(std::size_t interval = 1) : print_every(interval) {}

        template <typename T, std::size_t N> void on_begin(const dp::mat::vector<T, N> &x0) const noexcept {
            (void)x0;
            // Could print initial state
        }

        template <typename T, std::size_t N>
        bool on_iteration(const IterationInfo<T> &info, const dp::mat::vector<T, N> &x) const noexcept {
            (void)x;
            if (info.iteration % print_every == 0) {
                // Would print to stdout/logging system
                // For now, do nothing (real implementation would use iostream)
            }
            return false;
        }

        template <typename T, std::size_t N> void on_end(const OptimizationResult<T, N> &result) const noexcept {
            (void)result;
            // Could print final result
        }
    };

    /**
     * Early stopping callback based on objective threshold
     */
    template <typename T> struct EarlyStoppingCallback {
        T objective_threshold; ///< Stop if objective falls below this

        explicit EarlyStoppingCallback(T threshold) : objective_threshold(threshold) {}

        template <std::size_t N> void on_begin(const dp::mat::vector<T, N> &x0) const noexcept { (void)x0; }

        template <std::size_t N>
        bool on_iteration(const IterationInfo<T> &info, const dp::mat::vector<T, N> &x) const noexcept {
            (void)x;
            // Stop if objective is good enough
            return info.objective < objective_threshold;
        }

        template <std::size_t N> void on_end(const OptimizationResult<T, N> &result) const noexcept { (void)result; }
    };

} // namespace optinum::opti
