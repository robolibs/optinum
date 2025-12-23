#pragma once

// =============================================================================
// optinum/simd/arch/macros.hpp
// Utility Macros for SIMD and Performance
// =============================================================================

#include <optinum/simd/arch/arch.hpp>

#include <cstddef>

// =============================================================================
// Force Inline
// =============================================================================

#if defined(OPTINUM_COMPILER_MSVC)
#define OPTINUM_INLINE __forceinline
#define OPTINUM_NOINLINE __declspec(noinline)
#elif defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_INLINE inline __attribute__((always_inline))
#define OPTINUM_NOINLINE __attribute__((noinline))
#else
#define OPTINUM_INLINE inline
#define OPTINUM_NOINLINE
#endif

// =============================================================================
// Alignment
// =============================================================================

#if defined(OPTINUM_COMPILER_MSVC)
#define OPTINUM_ALIGN(n) __declspec(align(n))
#elif defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_ALIGN(n) __attribute__((aligned(n)))
#else
#define OPTINUM_ALIGN(n) alignas(n)
#endif

// Default SIMD alignment based on detected width
#if defined(OPTINUM_HAS_AVX512F)
#define OPTINUM_SIMD_ALIGN OPTINUM_ALIGN(64)
inline constexpr std::size_t OPTINUM_SIMD_ALIGNMENT = 64;
#elif defined(OPTINUM_HAS_AVX)
#define OPTINUM_SIMD_ALIGN OPTINUM_ALIGN(32)
inline constexpr std::size_t OPTINUM_SIMD_ALIGNMENT = 32;
#elif defined(OPTINUM_HAS_SSE) || defined(OPTINUM_HAS_NEON)
#define OPTINUM_SIMD_ALIGN OPTINUM_ALIGN(16)
inline constexpr std::size_t OPTINUM_SIMD_ALIGNMENT = 16;
#else
#define OPTINUM_SIMD_ALIGN
inline constexpr std::size_t OPTINUM_SIMD_ALIGNMENT = alignof(double);
#endif

// =============================================================================
// Restrict Pointer (no aliasing)
// =============================================================================

#if defined(OPTINUM_COMPILER_MSVC)
#define OPTINUM_RESTRICT __restrict
#elif defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_RESTRICT __restrict__
#else
#define OPTINUM_RESTRICT
#endif

// =============================================================================
// Branch Prediction Hints
// =============================================================================

#if defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_LIKELY(x) __builtin_expect(!!(x), 1)
#define OPTINUM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define OPTINUM_LIKELY(x) (x)
#define OPTINUM_UNLIKELY(x) (x)
#endif

// =============================================================================
// Prefetch
// =============================================================================

#if defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
// Locality hints: 0 = no temporal locality (use once), 3 = high locality (keep in cache)
#define OPTINUM_PREFETCH_READ(ptr, locality) __builtin_prefetch((ptr), 0, (locality))
#define OPTINUM_PREFETCH_WRITE(ptr, locality) __builtin_prefetch((ptr), 1, (locality))
#elif defined(OPTINUM_COMPILER_MSVC) && defined(OPTINUM_ARCH_X86_64)
#include <intrin.h>
#define OPTINUM_PREFETCH_READ(ptr, locality) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#define OPTINUM_PREFETCH_WRITE(ptr, locality) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define OPTINUM_PREFETCH_READ(ptr, locality) ((void)0)
#define OPTINUM_PREFETCH_WRITE(ptr, locality) ((void)0)
#endif

// =============================================================================
// Unreachable Code
// =============================================================================

#if defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_UNREACHABLE() __builtin_unreachable()
#elif defined(OPTINUM_COMPILER_MSVC)
#define OPTINUM_UNREACHABLE() __assume(0)
#else
#define OPTINUM_UNREACHABLE() ((void)0)
#endif

// =============================================================================
// Assume (optimization hint)
// =============================================================================

#if defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_ASSUME(cond) __builtin_assume(cond)
#elif defined(OPTINUM_COMPILER_MSVC)
#define OPTINUM_ASSUME(cond) __assume(cond)
#elif defined(OPTINUM_COMPILER_GCC) && (__GNUC__ >= 13)
#define OPTINUM_ASSUME(cond) __attribute__((assume(cond)))
#else
#define OPTINUM_ASSUME(cond) ((void)0)
#endif

// =============================================================================
// Pure / Const Function Attributes
// =============================================================================

#if defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
// Function depends only on arguments and global memory, no side effects
#define OPTINUM_PURE __attribute__((pure))
// Function depends only on arguments, no side effects, no global memory access
#define OPTINUM_CONST __attribute__((const))
#else
#define OPTINUM_PURE
#define OPTINUM_CONST
#endif

// =============================================================================
// Hot / Cold Function Attributes
// =============================================================================

#if defined(OPTINUM_COMPILER_GCC) || defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_HOT __attribute__((hot))
#define OPTINUM_COLD __attribute__((cold))
#else
#define OPTINUM_HOT
#define OPTINUM_COLD
#endif

// =============================================================================
// Vectorization Hints
// =============================================================================

#if defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_VECTORIZE _Pragma("clang loop vectorize(enable)")
#define OPTINUM_VECTORIZE_WIDTH(n) _Pragma("clang loop vectorize_width(" #n ")")
#define OPTINUM_UNROLL _Pragma("clang loop unroll(full)")
#define OPTINUM_UNROLL_N(n) _Pragma("clang loop unroll_count(" #n ")")
#elif defined(OPTINUM_COMPILER_GCC)
#define OPTINUM_VECTORIZE _Pragma("GCC ivdep")
#define OPTINUM_VECTORIZE_WIDTH(n)
#define OPTINUM_UNROLL _Pragma("GCC unroll 0")
#define OPTINUM_UNROLL_N(n) _Pragma("GCC unroll " #n)
#elif defined(OPTINUM_COMPILER_MSVC)
#define OPTINUM_VECTORIZE __pragma(loop(ivdep))
#define OPTINUM_VECTORIZE_WIDTH(n)
#define OPTINUM_UNROLL
#define OPTINUM_UNROLL_N(n)
#else
#define OPTINUM_VECTORIZE
#define OPTINUM_VECTORIZE_WIDTH(n)
#define OPTINUM_UNROLL
#define OPTINUM_UNROLL_N(n)
#endif

// =============================================================================
// Diagnostic Control
// =============================================================================

#if defined(OPTINUM_COMPILER_CLANG)
#define OPTINUM_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#define OPTINUM_DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#define OPTINUM_DIAGNOSTIC_IGNORE(w) _Pragma("clang diagnostic ignored \"" w "\"")
#elif defined(OPTINUM_COMPILER_GCC)
#define OPTINUM_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define OPTINUM_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define OPTINUM_DIAGNOSTIC_IGNORE(w) _Pragma("GCC diagnostic ignored \"" w "\"")
#elif defined(OPTINUM_COMPILER_MSVC)
#define OPTINUM_DIAGNOSTIC_PUSH __pragma(warning(push))
#define OPTINUM_DIAGNOSTIC_POP __pragma(warning(pop))
#define OPTINUM_DIAGNOSTIC_IGNORE(w) __pragma(warning(disable : w))
#else
#define OPTINUM_DIAGNOSTIC_PUSH
#define OPTINUM_DIAGNOSTIC_POP
#define OPTINUM_DIAGNOSTIC_IGNORE(w)
#endif
