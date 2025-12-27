#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace optinum::lie {

    // ===== LIE GROUP CONCEPT =====

    // A Lie group G must provide:
    // - Scalar type (T)
    // - DoF (degrees of freedom in tangent space)
    // - NumParams (number of parameters in storage)
    // - exp: tangent -> group
    // - log: group -> tangent
    // - inverse: group -> group
    // - operator*: group composition
    // - Adj: adjoint representation

    template <typename G>
    concept LieGroup = requires(G g, const G cg, typename G::Tangent tangent) {
        // Type aliases
        typename G::Scalar;
        typename G::Tangent;
        typename G::Params;

        // Static constants
        { G::DoF } -> std::convertible_to<std::size_t>;
        { G::NumParams } -> std::convertible_to<std::size_t>;

        // Identity element
        { G::identity() } -> std::same_as<G>;

        // Exponential map: tangent space -> group
        { G::exp(tangent) } -> std::same_as<G>;

        // Logarithmic map: group -> tangent space
        { cg.log() } -> std::same_as<typename G::Tangent>;

        // Inverse
        { cg.inverse() } -> std::same_as<G>;

        // Group composition
        { cg * cg } -> std::same_as<G>;

        // Adjoint representation
        { cg.Adj() };
    };

    // ===== HELPER CONCEPTS =====

    template <typename T>
    concept Floating = std::is_floating_point_v<T>;

    template <typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

} // namespace optinum::lie
