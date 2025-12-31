#pragma once

/**
 * Lookahead Optimizer Wrapper
 *
 * Lookahead wraps any base update policy (Adam, SGD, etc.) and maintains
 * slow and fast weights. The fast weights are updated by the base optimizer,
 * while the slow weights are updated periodically by interpolating toward
 * the fast weights.
 *
 * Algorithm:
 *   1. Maintain slow weights (phi) and fast weights (theta = x)
 *   2. For k inner steps: update theta using base optimizer
 *   3. Every k steps: phi = phi + alpha * (theta - phi)
 *   4. Reset theta = phi
 *
 * This creates a "lookahead" effect where the optimizer explores with fast
 * weights but commits to updates more conservatively via slow weights.
 *
 * Benefits:
 *   - Reduces variance in optimization
 *   - Improves convergence stability
 *   - Works with any base optimizer
 *
 * References:
 *   - Zhang et al. (2019) "Lookahead Optimizer: k steps forward, 1 step back"
 *     https://arxiv.org/abs/1907.08610
 *
 * @file lookahead.hpp
 */

#include <cstddef>

#include <datapod/matrix.hpp>
#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/bridge.hpp>

namespace optinum::meta {

    namespace dp = ::datapod;

    /**
     * Lookahead optimizer wrapper
     *
     * Wraps any update policy to add lookahead behavior. The wrapped policy
     * updates "fast weights" while Lookahead maintains "slow weights" that
     * are updated every k steps.
     *
     * Usage:
     * @code
     * // Create Lookahead with Adam as base
     * Lookahead<AdamUpdate> lookahead_adam;
     * lookahead_adam.config.k = 5;
     * lookahead_adam.config.alpha = 0.5;
     *
     * // Use with GradientDescent
     * GradientDescent<Lookahead<AdamUpdate>> gd;
     * gd.step_size = 0.001;
     *
     * // Or use standalone
     * dp::mat::vector<double, 3> x{1.0, 2.0, 3.0};
     * dp::mat::vector<double, 3> grad{0.1, 0.2, 0.3};
     * lookahead_adam.update(x, 0.001, grad);
     * @endcode
     *
     * @tparam BasePolicy The base update policy (e.g., AdamUpdate, VanillaUpdate)
     */
    template <typename BasePolicy> class Lookahead {
      public:
        /**
         * Lookahead configuration
         */
        struct Config {
            std::size_t k = 5;  ///< Number of inner steps before slow weight update
            double alpha = 0.5; ///< Slow weight interpolation factor (0-1)
        };

        Config config;   ///< Configuration parameters
        BasePolicy base; ///< The wrapped base update policy

        /// Default constructor
        Lookahead() = default;

        /// Constructor with config
        explicit Lookahead(const Config &cfg) : config(cfg) {}

        /// Constructor with base policy
        explicit Lookahead(BasePolicy base_policy) : base(std::move(base_policy)) {}

        /// Constructor with base policy and config
        Lookahead(BasePolicy base_policy, const Config &cfg) : config(cfg), base(std::move(base_policy)) {}

        /**
         * Update the iterate using Lookahead
         *
         * Applies the base policy update to fast weights (x), and every k steps
         * updates slow weights and resets fast weights to slow weights.
         *
         * @param x Current iterate (fast weights, modified in-place)
         * @param step_size Learning rate
         * @param gradient Current gradient
         */
        template <typename T, std::size_t N>
        void update(dp::mat::vector<T, N> &x, T step_size, const dp::mat::vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            // Lazy initialization of slow weights
            if (slow_weights_.size() != n) {
                slow_weights_.resize(n);
                // Initialize slow weights to current x
                for (std::size_t i = 0; i < n; ++i) {
                    slow_weights_[i] = static_cast<double>(x[i]);
                }
                step_count_ = 0;
            }

            // Apply base policy update to fast weights (x)
            base.update(x, step_size, gradient);

            // Increment step counter
            ++step_count_;

            // Every k steps, update slow weights and reset fast weights
            if (step_count_ >= config.k) {
                T *x_ptr = x.data();
                double *slow_ptr = slow_weights_.data();

                // SIMD-accelerated slow weight update and reset
                // slow = slow + alpha * (fast - slow) = (1-alpha)*slow + alpha*fast
                constexpr std::size_t W = simd::backend::default_pack_width<double>();
                using pack_t = simd::pack<double, W>;

                const pack_t alpha_pack(config.alpha);
                const pack_t one_minus_alpha(1.0 - config.alpha);

                const std::size_t main = simd::backend::main_loop_count_runtime(n, W);

                // SIMD loop
                for (std::size_t i = 0; i < main; i += W) {
                    // Load fast weights (convert to double if needed)
                    pack_t fast;
                    if constexpr (std::is_same_v<T, double>) {
                        fast = pack_t::loadu(x_ptr + i);
                    } else {
                        alignas(32) double fast_tmp[W];
                        for (std::size_t j = 0; j < W; ++j)
                            fast_tmp[j] = static_cast<double>(x_ptr[i + j]);
                        fast = pack_t::loadu(fast_tmp);
                    }

                    // Load slow weights
                    auto slow = pack_t::loadu(slow_ptr + i);

                    // Interpolate: slow = (1-alpha)*slow + alpha*fast
                    slow = pack_t::fma(one_minus_alpha, slow, alpha_pack * fast);

                    // Store updated slow weights
                    slow.storeu(slow_ptr + i);

                    // Reset fast weights to slow weights
                    if constexpr (std::is_same_v<T, double>) {
                        slow.storeu(x_ptr + i);
                    } else {
                        for (std::size_t j = 0; j < W; ++j) {
                            x_ptr[i + j] = static_cast<T>(slow[j]);
                        }
                    }
                }

                // Scalar tail
                for (std::size_t i = main; i < n; ++i) {
                    double fast_i = static_cast<double>(x_ptr[i]);
                    double slow_i = slow_ptr[i];

                    // Interpolate slow weights toward fast weights
                    slow_i = slow_i + config.alpha * (fast_i - slow_i);
                    slow_ptr[i] = slow_i;

                    // Reset fast weights to slow weights
                    x_ptr[i] = static_cast<T>(slow_i);
                }

                // Reset step counter
                step_count_ = 0;
            }
        }

        /**
         * Reset the optimizer state
         *
         * Clears slow weights and resets the base policy.
         */
        void reset() noexcept {
            slow_weights_.resize(0);
            step_count_ = 0;
            base.reset();
        }

        /**
         * Initialize for GradientDescent compatibility
         *
         * @param size Number of parameters
         */
        template <typename T, std::size_t N> void initialize(std::size_t size) noexcept {
            slow_weights_.resize(0); // Will be initialized on first update
            step_count_ = 0;
            base.template initialize<T, N>(size);
        }

        /// Get current step count within the k-step cycle
        std::size_t step_count() const noexcept { return step_count_; }

        /// Get reference to slow weights (for inspection)
        const dp::mat::vector<double, dp::mat::Dynamic> &slow_weights() const noexcept { return slow_weights_; }

        /// Get reference to base policy
        BasePolicy &get_base() noexcept { return base; }
        const BasePolicy &get_base() const noexcept { return base; }

      private:
        dp::mat::vector<double, dp::mat::Dynamic> slow_weights_; ///< Slow weights (phi)
        std::size_t step_count_ = 0;                             ///< Steps since last slow weight update
    };

    // Convenience type aliases
    // Note: These require the update policies to be included separately

} // namespace optinum::meta
