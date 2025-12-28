#pragma once

/**
 * @file meta.hpp
 * @brief Metaheuristic optimization module
 *
 * Population-based and sampling-based optimization methods for non-convex,
 * black-box, and stochastic optimization problems.
 *
 * All methods use SIMD acceleration for population operations.
 */

#include <cstddef>
#include <vector>

#include "../simd/simd.hpp"

// Population-based optimizers
#include "de.hpp"
#include "pso.hpp"

// Sampling-based optimizers
#include "cem.hpp"
#include "mppi.hpp"
#include "sa.hpp"

// TODO: Add includes as methods are implemented
// #include "cmaes.hpp"
// #include "ga.hpp"
// #include "lookahead.hpp"
// #include "swats.hpp"

namespace optinum::meta {

    /// Common result type for metaheuristic optimizers
    template <typename T> struct MetaResult {
        simd::Vector<T, simd::Dynamic> best_position;
        T best_value;
        std::size_t iterations;
        std::size_t function_evaluations;
        bool converged;
        std::vector<T> history; ///< Best value per iteration
    };

} // namespace optinum::meta
