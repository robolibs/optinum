#pragma once

// =============================================================================
// optinum/simd/view/slice.hpp
// Slicing utilities for views: seq(), fseq<>(), all, fix<N>
//
// Usage:
//   auto v = view<4>(vec);
//   v[seq(0, 10)]        // runtime: elements 0..9
//   v[seq(0, 10, 2)]     // runtime: elements 0, 2, 4, 6, 8
//   v[fseq<0, 10>()]     // compile-time: elements 0..9
//   v[all]               // all elements
//
//   auto m = view<4>(mat);
//   m[seq(1, 5), all]    // rows 1..4, all columns
//   m[all, fix<2>]       // all rows, column 2 only
// =============================================================================

#include <cstddef>
#include <optinum/simd/arch/arch.hpp>
#include <type_traits>

namespace optinum::simd {

    // =============================================================================
    // seq - Runtime range slice [start, stop) with optional step
    // =============================================================================

    struct seq {
        std::size_t start;
        std::size_t stop;
        std::size_t step;

        // seq(stop) -> [0, stop)
        constexpr explicit seq(std::size_t stop_) noexcept : start(0), stop(stop_), step(1) {}

        // seq(start, stop) -> [start, stop)
        constexpr seq(std::size_t start_, std::size_t stop_) noexcept : start(start_), stop(stop_), step(1) {}

        // seq(start, stop, step) -> [start, stop) with stride
        constexpr seq(std::size_t start_, std::size_t stop_, std::size_t step_) noexcept
            : start(start_), stop(stop_), step(step_) {}

        // Number of elements in the slice
        constexpr std::size_t size() const noexcept {
            if (stop <= start)
                return 0;
            return (stop - start + step - 1) / step;
        }

        // Get the i-th index in the slice
        constexpr std::size_t operator[](std::size_t i) const noexcept { return start + i * step; }
    };

    // =============================================================================
    // fseq<Start, Stop, Step> - Compile-time range slice
    // =============================================================================

    template <std::size_t Start, std::size_t Stop, std::size_t Step = 1> struct fseq {
        static constexpr std::size_t start = Start;
        static constexpr std::size_t stop = Stop;
        static constexpr std::size_t step = Step;

        static constexpr std::size_t size() noexcept {
            if constexpr (Stop <= Start) {
                return 0;
            } else {
                return (Stop - Start + Step - 1) / Step;
            }
        }

        static constexpr std::size_t operator[](std::size_t i) noexcept { return Start + i * Step; }
    };

    // Deduction guide for fseq
    template <std::size_t Stop> using fseq_end = fseq<0, Stop, 1>;

    // =============================================================================
    // all_t - Placeholder for "select entire dimension"
    // =============================================================================

    struct all_t {
        constexpr all_t() noexcept = default;

        // Convert to seq covering [0, N) when bound to actual dimension size
        constexpr seq bind(std::size_t N) const noexcept { return seq(0, N); }
    };

    // Global instance
    inline constexpr all_t all{};

    // =============================================================================
    // fix<N> - Compile-time single index selection
    // =============================================================================

    template <std::size_t N> struct fix {
        static constexpr std::size_t index = N;

        constexpr std::size_t operator()() const noexcept { return N; }
    };

    // =============================================================================
    // Type traits for slicing
    // =============================================================================

    template <typename T> struct is_slice : std::false_type {};
    template <> struct is_slice<seq> : std::true_type {};
    template <std::size_t S, std::size_t E, std::size_t St> struct is_slice<fseq<S, E, St>> : std::true_type {};
    template <> struct is_slice<all_t> : std::true_type {};

    template <typename T> struct is_fixed_index : std::false_type {};
    template <std::size_t N> struct is_fixed_index<fix<N>> : std::true_type {};

    template <typename T> inline constexpr bool is_slice_v = is_slice<T>::value;
    template <typename T> inline constexpr bool is_fixed_index_v = is_fixed_index<T>::value;

    // Check if slice is compile-time known
    template <typename T> struct is_static_slice : std::false_type {};
    template <std::size_t S, std::size_t E, std::size_t St> struct is_static_slice<fseq<S, E, St>> : std::true_type {};
    template <std::size_t N> struct is_static_slice<fix<N>> : std::true_type {};

    template <typename T> inline constexpr bool is_static_slice_v = is_static_slice<T>::value;

    // =============================================================================
    // Slice resolution helpers
    // =============================================================================

    namespace detail {
        // Resolve slice to concrete start/stop/step given dimension size
        template <typename Slice> struct slice_resolver;

        template <> struct slice_resolver<seq> {
            static constexpr auto resolve(const seq &s, std::size_t /*dim_size*/) noexcept { return s; }
        };

        template <std::size_t S, std::size_t E, std::size_t St> struct slice_resolver<fseq<S, E, St>> {
            static constexpr seq resolve(const fseq<S, E, St> & /*s*/, std::size_t /*dim_size*/) noexcept {
                return seq(S, E, St);
            }
        };

        template <> struct slice_resolver<all_t> {
            static constexpr seq resolve(const all_t & /*a*/, std::size_t dim_size) noexcept {
                return seq(0, dim_size);
            }
        };

        template <std::size_t N> struct slice_resolver<fix<N>> {
            static constexpr seq resolve(const fix<N> & /*f*/, std::size_t /*dim_size*/) noexcept {
                return seq(N, N + 1); // Single element as a 1-element slice
            }
        };
    } // namespace detail

    // Helper function to resolve any slice type to concrete seq
    template <typename Slice> constexpr seq resolve_slice(const Slice &s, std::size_t dim_size) noexcept {
        return detail::slice_resolver<Slice>::resolve(s, dim_size);
    }

} // namespace optinum::simd
