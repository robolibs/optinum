#pragma once

// Core types and interfaces
#include <optinum/opti/core/callbacks.hpp>
#include <optinum/opti/core/function.hpp>
#include <optinum/opti/core/types.hpp>

// Decay policies
#include <optinum/opti/decay/no_decay.hpp>

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

namespace optinum::opti {} // namespace optinum::opti
