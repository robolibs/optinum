#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

namespace optinum::lie {

    // ===== NUMERICAL CONSTANTS =====

    template <typename T> struct Constants {
        static_assert(std::is_floating_point_v<T>, "Constants require floating-point type");

        // Pi and related
        static constexpr T pi() noexcept { return T(3.14159265358979323846264338327950288); }
        static constexpr T two_pi() noexcept { return T(2) * pi(); }
        static constexpr T half_pi() noexcept { return pi() / T(2); }

        // Small angle threshold for Taylor series approximations
        // Below this, use Taylor expansion to avoid numerical issues
        // Approximately sqrt(machine epsilon):
        //   float: ~3.4e-4, double: ~1.5e-8
        static constexpr T epsilon() noexcept {
            if constexpr (std::is_same_v<T, float>) {
                return T(3.4e-4);
            } else if constexpr (std::is_same_v<T, double>) {
                return T(1.5e-8);
            } else {
                return T(1e-8); // Default for other floating types
            }
        }

        // Very small threshold for near-zero checks
        static constexpr T epsilon_sq() noexcept { return std::numeric_limits<T>::epsilon(); }

        // Default tolerance for iterative algorithms
        static constexpr T default_tolerance() noexcept { return T(1e-10); }

        // Threshold for "close to pi" singularity detection in SO3
        static constexpr T pi_threshold() noexcept { return pi() - epsilon(); }
    };

    // Convenience aliases
    template <typename T> inline constexpr T pi = Constants<T>::pi();

    template <typename T> inline constexpr T two_pi = Constants<T>::two_pi();

    template <typename T> inline constexpr T half_pi = Constants<T>::half_pi();

    // Epsilon specializations (sqrt of machine epsilon)
    template <typename T> inline constexpr T epsilon = T(1e-8);
    template <> inline constexpr float epsilon<float> = 3.4e-4f;
    template <> inline constexpr double epsilon<double> = 1.5e-8;
    template <> inline constexpr long double epsilon<long double> = 1.5e-10L;

    template <typename T> inline constexpr T epsilon_sq = Constants<T>::epsilon_sq();

} // namespace optinum::lie
