#pragma once

// Core types and interfaces
#include <optinum/opti/core/callbacks.hpp>
#include <optinum/opti/core/function.hpp>
#include <optinum/opti/core/types.hpp>

// Decay policies
#include <optinum/opti/decay/no_decay.hpp>

// Gradient descent and update policies
#include <optinum/opti/gradient/gradient_descent.hpp>
#include <optinum/opti/gradient/update_policies/vanilla_update.hpp>

// Test problems
#include <optinum/opti/problem/sphere.hpp>

namespace optinum::opti {} // namespace optinum::opti
