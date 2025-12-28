#pragma once

// Core types and interfaces
#include <optinum/opti/core/callbacks.hpp>
#include <optinum/opti/core/function.hpp>
#include <optinum/opti/core/types.hpp>

// Decay policies (learning rate schedulers)
#include <optinum/opti/decay/cosine_annealing.hpp>
#include <optinum/opti/decay/exponential_decay.hpp>
#include <optinum/opti/decay/inverse_time_decay.hpp>
#include <optinum/opti/decay/linear_decay.hpp>
#include <optinum/opti/decay/no_decay.hpp>
#include <optinum/opti/decay/polynomial_decay.hpp>
#include <optinum/opti/decay/step_decay.hpp>
#include <optinum/opti/decay/warmup_decay.hpp>

// Line search algorithms
#include <optinum/opti/line_search/line_search.hpp>

// Gradient descent and update policies
#include <optinum/opti/gradient/gradient_descent.hpp>
#include <optinum/opti/gradient/update_policies/adabound_update.hpp>
#include <optinum/opti/gradient/update_policies/adadelta_update.hpp>
#include <optinum/opti/gradient/update_policies/adagrad_update.hpp>
#include <optinum/opti/gradient/update_policies/adam_update.hpp>
#include <optinum/opti/gradient/update_policies/amsgrad_update.hpp>
#include <optinum/opti/gradient/update_policies/momentum_update.hpp>
#include <optinum/opti/gradient/update_policies/nadam_update.hpp>
#include <optinum/opti/gradient/update_policies/nesterov_update.hpp>
#include <optinum/opti/gradient/update_policies/rmsprop_update.hpp>
#include <optinum/opti/gradient/update_policies/vanilla_update.hpp>
#include <optinum/opti/gradient/update_policies/yogi_update.hpp>

// Quasi-Newton methods
#include <optinum/opti/quasi_newton/gauss_newton.hpp>
#include <optinum/opti/quasi_newton/lbfgs.hpp>
#include <optinum/opti/quasi_newton/levenberg_marquardt.hpp>

// Test problems
#include <optinum/opti/problem/ackley.hpp>
#include <optinum/opti/problem/rastrigin.hpp>
#include <optinum/opti/problem/rosenbrock.hpp>
#include <optinum/opti/problem/sphere.hpp>

namespace optinum::opti {

    // =========================================================================
    // Convenient type aliases for common optimizer configurations
    // =========================================================================

    /// Gradient descent with momentum
    using Momentum = GradientDescent<MomentumUpdate>;

    /// RMSprop optimizer
    using RMSprop = GradientDescent<RMSPropUpdate>;

    /// Adam optimizer
    using Adam = GradientDescent<AdamUpdate>;

    /// AMSGrad optimizer (Adam with guaranteed convergence)
    using AMSGrad = GradientDescent<AMSGradUpdate>;

    /// Nesterov accelerated gradient
    using Nesterov = GradientDescent<NesterovUpdate>;

    /// AdaGrad optimizer
    using AdaGrad = GradientDescent<AdaGradUpdate>;

    /// AdaDelta optimizer
    using AdaDelta = GradientDescent<AdaDeltaUpdate>;

    /// AdaBound optimizer
    using AdaBound = GradientDescent<AdaBoundUpdate>;

    /// NAdam optimizer (Nesterov + Adam)
    using NAdam = GradientDescent<NAdamUpdate>;

    /// Yogi optimizer
    using Yogi = GradientDescent<YogiUpdate>;

} // namespace optinum::opti
