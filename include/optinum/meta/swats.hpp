#pragma once

/**
 * SWATS (Switching from Adam to SGD)
 *
 * SWATS is an adaptive optimizer that starts with Adam and automatically
 * switches to SGD when beneficial. The switch is triggered when the
 * projected gradient becomes stable, indicating that Adam's adaptive
 * learning rate has converged.
 *
 * Algorithm:
 *   1. Start with Adam optimizer
 *   2. Track exponential moving average of learning rate: lr_avg
 *   3. Compute projected gradient for switching criterion
 *   4. When criterion met (lr variance is low): switch to SGD
 *   5. Continue with SGD using the learned learning rate
 *
 * The switching criterion is based on the observation that Adam's effective
 * learning rate stabilizes as training progresses. When this happens,
 * switching to SGD with the learned rate often improves generalization.
 *
 * References:
 *   - Keskar & Socher (2017) "Improving Generalization Performance by
 *     Switching from Adam to SGD"
 *     https://arxiv.org/abs/1712.07628
 *
 * @file swats.hpp
 */

#include <cmath>
#include <cstddef>
#include <limits>

#include <datapod/matrix.hpp>
#include <optinum/simd/bridge.hpp>

namespace optinum::meta {

    namespace dp = ::datapod;

    /**
     * SWATS (Switching from Adam to SGD) Update Policy
     *
     * An adaptive optimizer that automatically switches from Adam to SGD
     * when the learning rate stabilizes.
     *
     * Usage:
     * @code
     * SWATS<double> swats;
     * swats.config.beta1 = 0.9;
     * swats.config.beta2 = 0.999;
     * swats.config.min_adam_steps = 100;
     *
     * // Use with GradientDescent
     * GradientDescent<SWATS<double>> gd;
     * gd.step_size = 0.001;
     *
     * // Or use standalone
     * dp::mat::vector<double, 3> x{1.0, 2.0, 3.0};
     * dp::mat::vector<double, 3> grad{0.1, 0.2, 0.3};
     * swats.update(x, 0.001, grad);
     *
     * // Check if switched
     * if (swats.has_switched()) {
     *     std::cout << "Switched at iteration " << swats.switch_iteration() << std::endl;
     * }
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class SWATS {
      public:
        /**
         * SWATS configuration
         */
        struct Config {
            // Adam parameters
            T beta1 = T{0.9};    ///< First moment decay rate
            T beta2 = T{0.999};  ///< Second moment decay rate
            T epsilon = T{1e-8}; ///< Numerical stability constant

            // Switching parameters
            T switch_threshold = T{1e-9};     ///< Threshold for switching criterion
            std::size_t min_adam_steps = 100; ///< Minimum steps before switching allowed
        };

        Config config; ///< Configuration parameters

        /// Default constructor
        SWATS() = default;

        /// Constructor with config
        explicit SWATS(const Config &cfg) : config(cfg) {}

        /**
         * Update the iterate using SWATS
         *
         * Uses Adam until the switching criterion is met, then switches to SGD.
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate
         * @param gradient Current gradient
         */
        template <std::size_t N>
        void update(dp::mat::vector<T, N> &x, T step_size, const dp::mat::vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            // Lazy initialization
            if (m_.size() != n) {
                m_.resize(n);
                v_.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    m_[i] = T{0};
                    v_[i] = T{0};
                }
                iteration_ = 0;
                switched_ = false;
                switch_iter_ = 0;
                sgd_lr_ = step_size;
                lr_avg_ = T{0};
            }

            ++iteration_;

            if (!switched_) {
                // Adam phase
                update_adam(x, step_size, gradient);

                // Check switching criterion after minimum steps
                if (iteration_ >= config.min_adam_steps) {
                    check_switch_criterion(step_size, gradient);
                }
            } else {
                // SGD phase
                update_sgd(x, gradient);
            }
        }

        /**
         * Reset the optimizer state
         */
        void reset() noexcept {
            m_.resize(0);
            v_.resize(0);
            iteration_ = 0;
            switched_ = false;
            switch_iter_ = 0;
            sgd_lr_ = T{0};
            lr_avg_ = T{0};
        }

        /**
         * Initialize for GradientDescent compatibility
         */
        template <typename U, std::size_t N> void initialize(std::size_t) noexcept {
            m_.resize(0);
            v_.resize(0);
            iteration_ = 0;
            switched_ = false;
            switch_iter_ = 0;
            sgd_lr_ = T{0};
            lr_avg_ = T{0};
        }

        /// Check if optimizer has switched from Adam to SGD
        bool has_switched() const noexcept { return switched_; }

        /// Get the iteration at which the switch occurred (0 if not switched)
        std::size_t switch_iteration() const noexcept { return switch_iter_; }

        /// Get current iteration count
        std::size_t iteration() const noexcept { return iteration_; }

        /// Get the learned SGD learning rate
        T sgd_learning_rate() const noexcept { return sgd_lr_; }

      private:
        /**
         * Adam update step
         */
        template <std::size_t N>
        void update_adam(dp::mat::vector<T, N> &x, T step_size, const dp::mat::vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            T one_minus_beta1 = T{1} - config.beta1;
            T one_minus_beta2 = T{1} - config.beta2;

            // Bias correction
            T bias_correction1 = T{1} - std::pow(config.beta1, T(iteration_));
            T bias_correction2 = T{1} - std::pow(config.beta2, T(iteration_));
            T step_correction = step_size * std::sqrt(bias_correction2) / bias_correction1;

            T *m_ptr = m_.data();
            T *v_ptr = v_.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // Compute effective learning rate for switching criterion
            T lr_sum = T{0};

            for (std::size_t i = 0; i < n; ++i) {
                T g_i = g_ptr[i];

                // Update biased first moment: m = beta1 * m + (1 - beta1) * g
                m_ptr[i] = config.beta1 * m_ptr[i] + one_minus_beta1 * g_i;

                // Update biased second moment: v = beta2 * v + (1 - beta2) * g²
                v_ptr[i] = config.beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;

                // Compute bias-corrected estimates
                T m_hat = m_ptr[i] / bias_correction1;
                T v_hat = v_ptr[i] / bias_correction2;

                // Effective per-parameter learning rate
                T effective_lr = step_size / (std::sqrt(v_hat) + config.epsilon);
                lr_sum += effective_lr;

                // Update: x = x - step_size * m_hat / (sqrt(v_hat) + eps)
                x_ptr[i] -= step_correction * m_ptr[i] / (std::sqrt(v_ptr[i]) + config.epsilon);
            }

            // Update exponential moving average of learning rate
            T avg_lr = lr_sum / T(n);
            if (iteration_ == 1) {
                lr_avg_ = avg_lr;
            } else {
                // Exponential moving average
                lr_avg_ = T{0.9} * lr_avg_ + T{0.1} * avg_lr;
            }

            // Store for potential SGD switch
            sgd_lr_ = lr_avg_;
        }

        /**
         * SGD update step (after switching)
         */
        template <std::size_t N>
        void update_sgd(dp::mat::vector<T, N> &x, const dp::mat::vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();
            T *x_ptr = x.data();
            const T *g_ptr = gradient.data();

            for (std::size_t i = 0; i < n; ++i) {
                x_ptr[i] -= sgd_lr_ * g_ptr[i];
            }
        }

        /**
         * Check if we should switch from Adam to SGD
         *
         * The switching criterion is based on the stability of the effective
         * learning rate. When the variance of the learning rate becomes small,
         * we switch to SGD.
         */
        template <std::size_t N>
        void check_switch_criterion(T step_size, const dp::mat::vector<T, N> &gradient) noexcept {
            const std::size_t n = gradient.size();

            // Compute current effective learning rate
            T bias_correction2 = T{1} - std::pow(config.beta2, T(iteration_));
            T lr_sum = T{0};
            T lr_sq_sum = T{0};

            const T *v_ptr = v_.data();

            for (std::size_t i = 0; i < n; ++i) {
                T v_hat = v_ptr[i] / bias_correction2;
                T effective_lr = step_size / (std::sqrt(v_hat) + config.epsilon);
                lr_sum += effective_lr;
                lr_sq_sum += effective_lr * effective_lr;
            }

            T avg_lr = lr_sum / T(n);
            T var_lr = lr_sq_sum / T(n) - avg_lr * avg_lr;

            // Switch when variance is below threshold
            if (var_lr < config.switch_threshold && var_lr >= T{0}) {
                switched_ = true;
                switch_iter_ = iteration_;
                sgd_lr_ = avg_lr; // Use average effective learning rate for SGD
            }
        }

        dp::mat::vector<T, dp::mat::Dynamic> m_; ///< First moment estimate
        dp::mat::vector<T, dp::mat::Dynamic> v_; ///< Second moment estimate
        std::size_t iteration_ = 0;              ///< Current iteration
        bool switched_ = false;                  ///< Whether we've switched to SGD
        std::size_t switch_iter_ = 0;            ///< Iteration at which we switched
        T sgd_lr_ = T{0};                        ///< Learning rate for SGD phase
        T lr_avg_ = T{0};                        ///< Exponential moving average of learning rate
    };

} // namespace optinum::meta
