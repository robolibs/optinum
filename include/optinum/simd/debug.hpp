#pragma once

// =============================================================================
// optinum/simd/debug.hpp
// Debug mode infrastructure - runtime bounds and shape checking
// =============================================================================

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <source_location>

namespace optinum::simd::debug {

    // =============================================================================
    // Debug Configuration Macros
    // =============================================================================

    // Master switch for all runtime checks
    // Define OPTINUM_ENABLE_RUNTIME_CHECKS to enable all checks
    // This can be set in your build system or by defining it before including this header
#ifdef OPTINUM_ENABLE_RUNTIME_CHECKS
#ifndef OPTINUM_BOUNDS_CHECK
#define OPTINUM_BOUNDS_CHECK
#endif
#ifndef OPTINUM_SHAPE_CHECK
#define OPTINUM_SHAPE_CHECK
#endif
#endif

    // =============================================================================
    // Internal Helper Functions
    // =============================================================================

    // Print error message and abort
    inline void report_error(const char *message, const char *file, int line, const char *function) {
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "============================================================\n");
        std::fprintf(stderr, "OPTINUM RUNTIME ERROR\n");
        std::fprintf(stderr, "============================================================\n");
        std::fprintf(stderr, "Message:  %s\n", message);
        std::fprintf(stderr, "File:     %s\n", file);
        std::fprintf(stderr, "Line:     %d\n", line);
        std::fprintf(stderr, "Function: %s\n", function);
        std::fprintf(stderr, "============================================================\n");
        std::fprintf(stderr, "\n");
        std::abort();
    }

    // Format and report bounds error
    inline void report_bounds_error(std::size_t index, std::size_t size, const char *file, int line,
                                    const char *function) {
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer), "Index out of bounds: index=%zu, size=%zu", index, size);
        report_error(buffer, file, line, function);
    }

    // Format and report multi-dimensional bounds error
    inline void report_bounds_error_2d(std::size_t row, std::size_t col, std::size_t rows, std::size_t cols,
                                       const char *file, int line, const char *function) {
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer), "Index out of bounds: (%zu,%zu) exceeds (%zu,%zu)", row, col, rows, cols);
        report_error(buffer, file, line, function);
    }

    // Format and report multi-dimensional bounds error (3D)
    inline void report_bounds_error_3d(std::size_t i, std::size_t j, std::size_t k, std::size_t di, std::size_t dj,
                                       std::size_t dk, const char *file, int line, const char *function) {
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer), "Index out of bounds: (%zu,%zu,%zu) exceeds (%zu,%zu,%zu)", i, j, k, di,
                      dj, dk);
        report_error(buffer, file, line, function);
    }

    // Format and report shape mismatch error
    inline void report_shape_error(const char *operation, std::size_t size1, std::size_t size2, const char *file,
                                   int line, const char *function) {
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer), "Shape mismatch in %s: size1=%zu, size2=%zu", operation, size1, size2);
        report_error(buffer, file, line, function);
    }

    // Format and report shape mismatch error (2D)
    inline void report_shape_error_2d(const char *operation, std::size_t rows1, std::size_t cols1, std::size_t rows2,
                                      std::size_t cols2, const char *file, int line, const char *function) {
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer), "Shape mismatch in %s: (%zu,%zu) vs (%zu,%zu)", operation, rows1, cols1,
                      rows2, cols2);
        report_error(buffer, file, line, function);
    }

    // Format and report matrix multiplication shape error
    inline void report_matmul_shape_error(std::size_t m, std::size_t k1, std::size_t k2, std::size_t n,
                                          const char *file, int line, const char *function) {
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer),
                      "Matrix multiplication shape mismatch: (%zu,%zu) * (%zu,%zu) - inner dimensions must match", m,
                      k1, k2, n);
        report_error(buffer, file, line, function);
    }

} // namespace optinum::simd::debug

// =============================================================================
// Public Debug Macros
// =============================================================================

// General assertion with custom message
#ifdef OPTINUM_ENABLE_RUNTIME_CHECKS
#define OPTINUM_ASSERT(condition, message)                                                                             \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            ::optinum::simd::debug::report_error(message, __FILE__, __LINE__, __func__);                               \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT(condition, message) ((void)0)
#endif

// Bounds check for 1D access
#ifdef OPTINUM_BOUNDS_CHECK
#define OPTINUM_ASSERT_BOUNDS(index, size)                                                                             \
    do {                                                                                                               \
        if ((index) >= (size)) {                                                                                       \
            ::optinum::simd::debug::report_bounds_error((index), (size), __FILE__, __LINE__, __func__);                \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT_BOUNDS(index, size) ((void)0)
#endif

// Bounds check for 2D access
#ifdef OPTINUM_BOUNDS_CHECK
#define OPTINUM_ASSERT_BOUNDS_2D(row, col, rows, cols)                                                                 \
    do {                                                                                                               \
        if ((row) >= (rows) || (col) >= (cols)) {                                                                      \
            ::optinum::simd::debug::report_bounds_error_2d((row), (col), (rows), (cols), __FILE__, __LINE__,           \
                                                           __func__);                                                  \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT_BOUNDS_2D(row, col, rows, cols) ((void)0)
#endif

// Bounds check for 3D access
#ifdef OPTINUM_BOUNDS_CHECK
#define OPTINUM_ASSERT_BOUNDS_3D(i, j, k, di, dj, dk)                                                                  \
    do {                                                                                                               \
        if ((i) >= (di) || (j) >= (dj) || (k) >= (dk)) {                                                               \
            ::optinum::simd::debug::report_bounds_error_3d((i), (j), (k), (di), (dj), (dk), __FILE__, __LINE__,        \
                                                           __func__);                                                  \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT_BOUNDS_3D(i, j, k, di, dj, dk) ((void)0)
#endif

// Shape check for 1D operations (e.g., vector addition)
#ifdef OPTINUM_SHAPE_CHECK
#define OPTINUM_ASSERT_SHAPE(size1, size2, operation)                                                                  \
    do {                                                                                                               \
        if ((size1) != (size2)) {                                                                                      \
            ::optinum::simd::debug::report_shape_error((operation), (size1), (size2), __FILE__, __LINE__, __func__);   \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT_SHAPE(size1, size2, operation) ((void)0)
#endif

// Shape check for 2D operations (e.g., matrix addition)
#ifdef OPTINUM_SHAPE_CHECK
#define OPTINUM_ASSERT_SHAPE_2D(rows1, cols1, rows2, cols2, operation)                                                 \
    do {                                                                                                               \
        if ((rows1) != (rows2) || (cols1) != (cols2)) {                                                                \
            ::optinum::simd::debug::report_shape_error_2d((operation), (rows1), (cols1), (rows2), (cols2), __FILE__,   \
                                                          __LINE__, __func__);                                         \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT_SHAPE_2D(rows1, cols1, rows2, cols2, operation) ((void)0)
#endif

// Shape check for matrix multiplication
#ifdef OPTINUM_SHAPE_CHECK
#define OPTINUM_ASSERT_MATMUL_SHAPE(m, k1, k2, n)                                                                      \
    do {                                                                                                               \
        if ((k1) != (k2)) {                                                                                            \
            ::optinum::simd::debug::report_matmul_shape_error((m), (k1), (k2), (n), __FILE__, __LINE__, __func__);     \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT_MATMUL_SHAPE(m, k1, k2, n) ((void)0)
#endif

// Null pointer check
#ifdef OPTINUM_ENABLE_RUNTIME_CHECKS
#define OPTINUM_ASSERT_NOT_NULL(ptr, message)                                                                          \
    do {                                                                                                               \
        if ((ptr) == nullptr) {                                                                                        \
            ::optinum::simd::debug::report_error((message), __FILE__, __LINE__, __func__);                             \
        }                                                                                                              \
    } while (0)
#else
#define OPTINUM_ASSERT_NOT_NULL(ptr, message) ((void)0)
#endif

// =============================================================================
// Debug Mode Information
// =============================================================================

namespace optinum::simd::debug {

    // Check if debug mode is enabled at compile time
    inline constexpr bool is_debug_enabled() {
#ifdef OPTINUM_ENABLE_RUNTIME_CHECKS
        return true;
#else
        return false;
#endif
    }

    // Check if bounds checking is enabled
    inline constexpr bool is_bounds_check_enabled() {
#ifdef OPTINUM_BOUNDS_CHECK
        return true;
#else
        return false;
#endif
    }

    // Check if shape checking is enabled
    inline constexpr bool is_shape_check_enabled() {
#ifdef OPTINUM_SHAPE_CHECK
        return true;
#else
        return false;
#endif
    }

    // Print debug configuration
    inline void print_debug_config() {
        std::printf("OPTINUM Debug Configuration:\n");
        std::printf("  Runtime checks:  %s\n", is_debug_enabled() ? "ENABLED" : "DISABLED");
        std::printf("  Bounds checking: %s\n", is_bounds_check_enabled() ? "ENABLED" : "DISABLED");
        std::printf("  Shape checking:  %s\n", is_shape_check_enabled() ? "ENABLED" : "DISABLED");
    }

} // namespace optinum::simd::debug
