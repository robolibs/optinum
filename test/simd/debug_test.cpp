// =============================================================================
// test/simd/debug_test.cpp
// Tests for debug mode - runtime bounds and shape checking
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/debug.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace on = optinum;

// =============================================================================
// Debug Configuration Tests
// =============================================================================

TEST_CASE("debug - Configuration queries") {
    using namespace on::simd::debug;

    SUBCASE("Check if debug is enabled") {
        // These functions should be constexpr and callable at compile time
        [[maybe_unused]] constexpr bool debug_enabled = is_debug_enabled();
        [[maybe_unused]] constexpr bool bounds_enabled = is_bounds_check_enabled();
        [[maybe_unused]] constexpr bool shape_enabled = is_shape_check_enabled();

        // At runtime, we can check the values
        // When OPTINUM_ENABLE_RUNTIME_CHECKS is defined, all should be true
        // Otherwise, all should be false
#ifdef OPTINUM_ENABLE_RUNTIME_CHECKS
        CHECK(is_debug_enabled() == true);
        CHECK(is_bounds_check_enabled() == true);
        CHECK(is_shape_check_enabled() == true);
#else
        CHECK(is_debug_enabled() == false);
        // Bounds and shape checks might be independently enabled
#endif
    }

    SUBCASE("Print debug configuration") {
        // This should not crash and should print meaningful output
        print_debug_config();
    }
}

// =============================================================================
// Vector Bounds Checking Tests
// =============================================================================

TEST_CASE("debug - Vector bounds checking") {
    using namespace on::simd;

    dp::mat::Vector<float, 4> v_storage;
    Vector<float, 4> v(v_storage);
    v.fill(1.0f);

    SUBCASE("Valid access - should not abort") {
        // These should all work fine
        [[maybe_unused]] float val0 = v[0];
        [[maybe_unused]] float val1 = v[1];
        [[maybe_unused]] float val2 = v[2];
        [[maybe_unused]] float val3 = v[3];
        CHECK(true); // If we get here, no abort occurred
    }

    SUBCASE("Const vector access") {
        const Vector<float, 4> cv = v;
        [[maybe_unused]] float val = cv[0];
        CHECK(true);
    }

// Only test out-of-bounds behavior when bounds checking is enabled
#ifdef OPTINUM_BOUNDS_CHECK
    // Note: We cannot easily test that the program aborts in doctest
    // These tests would normally trigger an abort, so we document the expected behavior
    // In a real scenario, accessing v[4] or v[100] would abort with an error message
#endif
}

// =============================================================================
// Matrix Bounds Checking Tests
// =============================================================================

TEST_CASE("debug - Matrix bounds checking") {
    using namespace on::simd;

    dp::mat::Matrix<float, 3, 3> m_storage;
    Matrix<float, 3, 3> m(m_storage);
    m.fill(2.0f);

    SUBCASE("Valid 2D access - should not abort") {
        [[maybe_unused]] float val00 = m(0, 0);
        [[maybe_unused]] float val11 = m(1, 1);
        [[maybe_unused]] float val22 = m(2, 2);
        [[maybe_unused]] float val02 = m(0, 2);
        [[maybe_unused]] float val20 = m(2, 0);
        CHECK(true);
    }

    SUBCASE("Valid 1D access - should not abort") {
        [[maybe_unused]] float val0 = m[0];
        [[maybe_unused]] float val4 = m[4];
        [[maybe_unused]] float val8 = m[8];
        CHECK(true);
    }

    SUBCASE("Const matrix access") {
        const Matrix<float, 3, 3> cm = m;
        [[maybe_unused]] float val = cm(1, 1);
        [[maybe_unused]] float val2 = cm[5];
        CHECK(true);
    }

// Only test out-of-bounds behavior when bounds checking is enabled
#ifdef OPTINUM_BOUNDS_CHECK
    // In debug mode, m(3, 0), m(0, 3), m(3, 3), m[9], etc. would abort
#endif
}

// =============================================================================
// Macro Behavior Tests
// =============================================================================

TEST_CASE("debug - Macro behavior") {
    SUBCASE("OPTINUM_ASSERT with true condition") {
        OPTINUM_ASSERT(true, "This should not trigger");
        CHECK(true); // Should reach here
    }

    SUBCASE("OPTINUM_ASSERT_BOUNDS with valid index") {
        std::size_t index = 5;
        std::size_t size = 10;
        OPTINUM_ASSERT_BOUNDS(index, size);
        CHECK(true); // Should reach here
    }

    SUBCASE("OPTINUM_ASSERT_BOUNDS_2D with valid indices") {
        std::size_t row = 2, col = 3;
        std::size_t rows = 5, cols = 5;
        OPTINUM_ASSERT_BOUNDS_2D(row, col, rows, cols);
        CHECK(true); // Should reach here
    }

    SUBCASE("OPTINUM_ASSERT_BOUNDS_3D with valid indices") {
        std::size_t i = 1, j = 2, k = 3;
        std::size_t di = 4, dj = 5, dk = 6;
        OPTINUM_ASSERT_BOUNDS_3D(i, j, k, di, dj, dk);
        CHECK(true); // Should reach here
    }

    SUBCASE("OPTINUM_ASSERT_SHAPE with matching sizes") {
        std::size_t size1 = 10, size2 = 10;
        OPTINUM_ASSERT_SHAPE(size1, size2, "test operation");
        CHECK(true); // Should reach here
    }

    SUBCASE("OPTINUM_ASSERT_SHAPE_2D with matching dimensions") {
        std::size_t rows1 = 3, cols1 = 4;
        std::size_t rows2 = 3, cols2 = 4;
        OPTINUM_ASSERT_SHAPE_2D(rows1, cols1, rows2, cols2, "test operation");
        CHECK(true); // Should reach here
    }

    SUBCASE("OPTINUM_ASSERT_MATMUL_SHAPE with matching inner dimensions") {
        std::size_t m = 3, k1 = 4, k2 = 4, n = 5;
        OPTINUM_ASSERT_MATMUL_SHAPE(m, k1, k2, n);
        CHECK(true); // Should reach here
    }

    SUBCASE("OPTINUM_ASSERT_NOT_NULL with valid pointer") {
        int value = 42;
        int *ptr = &value;
        OPTINUM_ASSERT_NOT_NULL(ptr, "Pointer should not be null");
        CHECK(true); // Should reach here
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_CASE("debug - Integration with SIMD operations") {
    using namespace on::simd;

    SUBCASE("Vector operations with matching sizes") {
        dp::mat::Vector<float, 4> v1_storage, v2_storage, v3_storage, v4_storage, v5_storage, v6_storage;
        Vector<float, 4> v1(v1_storage), v2(v2_storage);
        Vector<float, 4> v3(v3_storage), v4(v4_storage), v5(v5_storage), v6(v6_storage);
        v1.fill(1.0f);
        v2.fill(2.0f);

        // Use free functions to write results to output
        add(v3_storage.data(), v1, v2);
        sub(v4_storage.data(), v1, v2);
        mul(v5_storage.data(), v1, v2);
        div(v6_storage.data(), v1, v2);

        CHECK(v3[0] == doctest::Approx(3.0f));
        CHECK(v4[0] == doctest::Approx(-1.0f));
        CHECK(v5[0] == doctest::Approx(2.0f));
        CHECK(v6[0] == doctest::Approx(0.5f));
    }

    SUBCASE("Matrix operations with matching dimensions") {
        dp::mat::Matrix<float, 3, 3> m1_storage, m2_storage, m3_storage, m4_storage;
        Matrix<float, 3, 3> m1(m1_storage), m2(m2_storage);
        Matrix<float, 3, 3> m3(m3_storage), m4(m4_storage);
        m1.fill(1.0f);
        m2.fill(2.0f);

        // Element-wise operations using free functions
        add(m3_storage.data(), m1, m2);
        sub(m4_storage.data(), m1, m2);

        CHECK(m3(0, 0) == doctest::Approx(3.0f));
        CHECK(m4(0, 0) == doctest::Approx(-1.0f));
    }

    SUBCASE("Matrix multiplication with compatible dimensions") {
        dp::mat::Matrix<float, 2, 3> m1_storage;
        dp::mat::Matrix<float, 3, 4> m2_storage;
        dp::mat::Matrix<float, 2, 4> m3_storage;
        Matrix<float, 2, 3> m1(m1_storage);
        Matrix<float, 3, 4> m2(m2_storage);
        Matrix<float, 2, 4> m3(m3_storage);
        m1.fill(1.0f);
        m2.fill(2.0f);

        // 2x3 * 3x4 -> 2x4 should work
        matmul_to(m3_storage.data(), m1, m2);
        CHECK(m3(0, 0) == doctest::Approx(6.0f)); // sum of 3 * (1*2)
    }

    SUBCASE("Matrix-vector multiplication with compatible dimensions") {
        dp::mat::Matrix<float, 3, 4> m_storage;
        dp::mat::Vector<float, 4> v_storage;
        dp::mat::Vector<float, 3> result_storage;
        Matrix<float, 3, 4> m(m_storage);
        Vector<float, 4> v(v_storage);
        Vector<float, 3> result(result_storage);
        m.fill(1.0f);
        v.fill(2.0f);

        // 3x4 * 4 -> 3 should work
        matvec_to(result_storage.data(), m, v);
        CHECK(result[0] == doctest::Approx(8.0f)); // sum of 4 * (1*2)
    }
}

// =============================================================================
// Performance Tests (debug overhead should be zero when disabled)
// =============================================================================

TEST_CASE("debug - Performance characteristics") {
    using namespace on::simd;

    SUBCASE("No overhead when disabled") {
        dp::mat::Vector<float, 100> v_storage;
        Vector<float, 100> v(v_storage);
        v.iota();

        // Access all elements - should have no overhead when debug is disabled
        float sum = 0.0f;
        for (std::size_t i = 0; i < 100; ++i) {
            sum += v[i];
        }

        CHECK(sum == doctest::Approx(4950.0f)); // 0+1+2+...+99 = 99*100/2
    }

    SUBCASE("Matrix access pattern") {
        dp::mat::Matrix<float, 10, 10> m_storage;
        Matrix<float, 10, 10> m(m_storage);
        m.iota();

        float sum = 0.0f;
        for (std::size_t i = 0; i < 10; ++i) {
            for (std::size_t j = 0; j < 10; ++j) {
                sum += m(i, j);
            }
        }

        CHECK(sum == doctest::Approx(4950.0f)); // 0+1+2+...+99
    }
}

// =============================================================================
// Documentation Tests
// =============================================================================

TEST_CASE("debug - Usage examples") {
    using namespace on::simd;
    using namespace on::simd::debug;

    SUBCASE("Example 1: Enable debug checks globally") {
        // To enable all checks, define OPTINUM_ENABLE_RUNTIME_CHECKS
        // before including optinum headers, or set it in your build system:
        // -DOPTINUM_ENABLE_RUNTIME_CHECKS

        dp::mat::Vector<float, 4> v_storage;
        Vector<float, 4> v(v_storage);
        v.fill(1.0f);

        // With debug enabled, this would abort:
        // v[10]; // Out of bounds!

        // Valid access:
        float val = v[2];
        CHECK(val == 1.0f);
    }

    SUBCASE("Example 2: Check debug status at runtime") {
        // You can query the debug status
        if (is_debug_enabled()) {
            // Debug mode is active
            // Extra validation could be added here
        }

        CHECK(true); // Always passes
    }

    SUBCASE("Example 3: Selective checks") {
        // You can enable only specific checks:
        // -DOPTINUM_BOUNDS_CHECK (only bounds checking)
        // -DOPTINUM_SHAPE_CHECK (only shape checking)

        if (is_bounds_check_enabled()) {
            // Bounds checking is active
        }

        if (is_shape_check_enabled()) {
            // Shape checking is active
        }

        CHECK(true);
    }
}

// =============================================================================
// Expected Behavior Documentation
// =============================================================================

/*
 * Debug Mode Expected Behavior:
 *
 * When OPTINUM_ENABLE_RUNTIME_CHECKS is defined:
 *   - All debug checks are enabled
 *   - Out-of-bounds access aborts with detailed error message
 *   - Shape mismatches abort with detailed error message
 *   - Error messages include file, line, and function name
 *
 * When OPTINUM_ENABLE_RUNTIME_CHECKS is NOT defined:
 *   - All debug checks compile to no-ops ((void)0)
 *   - Zero runtime overhead
 *   - No bounds or shape validation
 *   - Maximum performance
 *
 * Selective Checking:
 *   - Define OPTINUM_BOUNDS_CHECK for bounds checking only
 *   - Define OPTINUM_SHAPE_CHECK for shape checking only
 *
 * Error Message Format:
 *   ============================================================
 *   OPTINUM RUNTIME ERROR
 *   ============================================================
 *   Message:  Index out of bounds: index=10, size=4
 *   File:     /path/to/file.cpp
 *   Line:     123
 *   Function: my_function
 *   ============================================================
 */
