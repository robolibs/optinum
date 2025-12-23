#pragma once

// =============================================================================
// optinum/simd/arch/arch.hpp
// CPU Architecture and SIMD Capability Detection
// =============================================================================

// =============================================================================
// Compiler Detection (must be outside namespace for macros)
// =============================================================================

#if defined(__clang__)
#define OPTINUM_COMPILER_CLANG 1
#define OPTINUM_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define OPTINUM_COMPILER_GCC 1
#define OPTINUM_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#define OPTINUM_COMPILER_MSVC 1
#define OPTINUM_COMPILER_VERSION _MSC_VER
#elif defined(__INTEL_COMPILER)
#define OPTINUM_COMPILER_INTEL 1
#define OPTINUM_COMPILER_VERSION __INTEL_COMPILER
#else
#define OPTINUM_COMPILER_UNKNOWN 1
#define OPTINUM_COMPILER_VERSION 0
#endif

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
#define OPTINUM_PLATFORM_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
#define OPTINUM_PLATFORM_MACOS 1
#elif defined(__linux__)
#define OPTINUM_PLATFORM_LINUX 1
#elif defined(__unix__)
#define OPTINUM_PLATFORM_UNIX 1
#else
#define OPTINUM_PLATFORM_UNKNOWN 1
#endif

// =============================================================================
// Architecture Detection
// =============================================================================

#if defined(__x86_64__) || defined(_M_X64)
#define OPTINUM_ARCH_X86_64 1
#define OPTINUM_ARCH_64BIT 1
#elif defined(__i386__) || defined(_M_IX86)
#define OPTINUM_ARCH_X86 1
#define OPTINUM_ARCH_32BIT 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define OPTINUM_ARCH_ARM64 1
#define OPTINUM_ARCH_64BIT 1
#elif defined(__arm__) || defined(_M_ARM)
#define OPTINUM_ARCH_ARM32 1
#define OPTINUM_ARCH_32BIT 1
#elif defined(__riscv)
#define OPTINUM_ARCH_RISCV 1
#if __riscv_xlen == 64
#define OPTINUM_ARCH_64BIT 1
#else
#define OPTINUM_ARCH_32BIT 1
#endif
#else
#define OPTINUM_ARCH_UNKNOWN 1
#endif

// =============================================================================
// x86/x86_64 SIMD Detection
// =============================================================================

#if defined(OPTINUM_ARCH_X86_64) || defined(OPTINUM_ARCH_X86)

// SSE
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define OPTINUM_HAS_SSE 1
#endif

// SSE2
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define OPTINUM_HAS_SSE2 1
#endif

// SSE3
#if defined(__SSE3__)
#define OPTINUM_HAS_SSE3 1
#endif

// SSSE3
#if defined(__SSSE3__)
#define OPTINUM_HAS_SSSE3 1
#endif

// SSE4.1
#if defined(__SSE4_1__)
#define OPTINUM_HAS_SSE41 1
#endif

// SSE4.2
#if defined(__SSE4_2__)
#define OPTINUM_HAS_SSE42 1
#endif

// AVX
#if defined(__AVX__)
#define OPTINUM_HAS_AVX 1
#endif

// AVX2
#if defined(__AVX2__)
#define OPTINUM_HAS_AVX2 1
#endif

// AVX-512 Foundation
#if defined(__AVX512F__)
#define OPTINUM_HAS_AVX512F 1
#endif

// AVX-512 Vector Length Extensions
#if defined(__AVX512VL__)
#define OPTINUM_HAS_AVX512VL 1
#endif

// AVX-512 Byte and Word Instructions
#if defined(__AVX512BW__)
#define OPTINUM_HAS_AVX512BW 1
#endif

// AVX-512 Doubleword and Quadword Instructions
#if defined(__AVX512DQ__)
#define OPTINUM_HAS_AVX512DQ 1
#endif

// AVX-512 Conflict Detection
#if defined(__AVX512CD__)
#define OPTINUM_HAS_AVX512CD 1
#endif

// FMA (Fused Multiply-Add)
#if defined(__FMA__)
#define OPTINUM_HAS_FMA 1
#endif

// F16C (Half-precision conversion)
#if defined(__F16C__)
#define OPTINUM_HAS_F16C 1
#endif

// BMI1 (Bit Manipulation Instructions 1)
#if defined(__BMI__)
#define OPTINUM_HAS_BMI1 1
#endif

// BMI2 (Bit Manipulation Instructions 2)
#if defined(__BMI2__)
#define OPTINUM_HAS_BMI2 1
#endif

#endif // x86/x86_64

// =============================================================================
// ARM SIMD Detection
// =============================================================================

#if defined(OPTINUM_ARCH_ARM64) || defined(OPTINUM_ARCH_ARM32)

// NEON (always available on ARM64, optional on ARM32)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define OPTINUM_HAS_NEON 1
#endif

// ARM64 always has NEON
#if defined(OPTINUM_ARCH_ARM64) && !defined(OPTINUM_HAS_NEON)
#define OPTINUM_HAS_NEON 1
#endif

// SVE (Scalable Vector Extension)
#if defined(__ARM_FEATURE_SVE)
#define OPTINUM_HAS_SVE 1
#endif

// SVE2
#if defined(__ARM_FEATURE_SVE2)
#define OPTINUM_HAS_SVE2 1
#endif

#endif // ARM

// =============================================================================
// RISC-V SIMD Detection
// =============================================================================

#if defined(OPTINUM_ARCH_RISCV)

// RISC-V Vector Extension
#if defined(__riscv_v)
#define OPTINUM_HAS_RVV 1
#endif

#endif // RISC-V

// =============================================================================
// SIMD Level Macros
// =============================================================================

#if defined(OPTINUM_HAS_AVX512F)
#define OPTINUM_SIMD_LEVEL 512
#define OPTINUM_SIMD_WIDTH_BYTES 64
#elif defined(OPTINUM_HAS_AVX)
#define OPTINUM_SIMD_LEVEL 256
#define OPTINUM_SIMD_WIDTH_BYTES 32
#elif defined(OPTINUM_HAS_SSE) || defined(OPTINUM_HAS_NEON)
#define OPTINUM_SIMD_LEVEL 128
#define OPTINUM_SIMD_WIDTH_BYTES 16
#else
#define OPTINUM_SIMD_LEVEL 0
#define OPTINUM_SIMD_WIDTH_BYTES 0
#endif

// =============================================================================
// Include Appropriate SIMD Headers (MUST be outside namespace!)
// =============================================================================

#if defined(OPTINUM_HAS_AVX512F)
#include <immintrin.h>
#elif defined(OPTINUM_HAS_AVX2)
#include <immintrin.h>
#elif defined(OPTINUM_HAS_AVX)
#include <immintrin.h>
#elif defined(OPTINUM_HAS_SSE42)
#include <nmmintrin.h>
#elif defined(OPTINUM_HAS_SSE41)
#include <smmintrin.h>
#elif defined(OPTINUM_HAS_SSSE3)
#include <tmmintrin.h>
#elif defined(OPTINUM_HAS_SSE3)
#include <pmmintrin.h>
#elif defined(OPTINUM_HAS_SSE2)
#include <emmintrin.h>
#elif defined(OPTINUM_HAS_SSE)
#include <xmmintrin.h>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <arm_neon.h>
#endif

// =============================================================================
// Namespace with constexpr constants and query functions
// =============================================================================

namespace optinum::simd::arch {

// SIMD width constants
#if defined(OPTINUM_HAS_AVX512F)
    inline constexpr int SIMD_WIDTH_BYTES = 64;
    inline constexpr int SIMD_WIDTH_FLOAT = 16;
    inline constexpr int SIMD_WIDTH_DOUBLE = 8;
#elif defined(OPTINUM_HAS_AVX)
    inline constexpr int SIMD_WIDTH_BYTES = 32;
    inline constexpr int SIMD_WIDTH_FLOAT = 8;
    inline constexpr int SIMD_WIDTH_DOUBLE = 4;
#elif defined(OPTINUM_HAS_SSE) || defined(OPTINUM_HAS_NEON)
    inline constexpr int SIMD_WIDTH_BYTES = 16;
    inline constexpr int SIMD_WIDTH_FLOAT = 4;
    inline constexpr int SIMD_WIDTH_DOUBLE = 2;
#else
    inline constexpr int SIMD_WIDTH_BYTES = 0;
    inline constexpr int SIMD_WIDTH_FLOAT = 1;
    inline constexpr int SIMD_WIDTH_DOUBLE = 1;
#endif

    // Compile-time Feature Query Functions

    [[nodiscard]] consteval bool has_sse() noexcept {
#if defined(OPTINUM_HAS_SSE)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_sse2() noexcept {
#if defined(OPTINUM_HAS_SSE2)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_sse3() noexcept {
#if defined(OPTINUM_HAS_SSE3)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_ssse3() noexcept {
#if defined(OPTINUM_HAS_SSSE3)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_sse41() noexcept {
#if defined(OPTINUM_HAS_SSE41)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_sse42() noexcept {
#if defined(OPTINUM_HAS_SSE42)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_avx() noexcept {
#if defined(OPTINUM_HAS_AVX)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_avx2() noexcept {
#if defined(OPTINUM_HAS_AVX2)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_avx512f() noexcept {
#if defined(OPTINUM_HAS_AVX512F)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_fma() noexcept {
#if defined(OPTINUM_HAS_FMA)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_neon() noexcept {
#if defined(OPTINUM_HAS_NEON)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval bool has_sve() noexcept {
#if defined(OPTINUM_HAS_SVE)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] consteval int simd_level() noexcept { return OPTINUM_SIMD_LEVEL; }

    [[nodiscard]] consteval int simd_width_bytes() noexcept { return SIMD_WIDTH_BYTES; }

    template <typename T> [[nodiscard]] consteval int simd_width() noexcept {
        if constexpr (SIMD_WIDTH_BYTES == 0) {
            return 1;
        } else {
            return SIMD_WIDTH_BYTES / sizeof(T);
        }
    }

} // namespace optinum::simd::arch
