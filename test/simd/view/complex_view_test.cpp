// =============================================================================
// test/simd/view/complex_view_test.cpp
// Tests for complex_view - transparent SIMD access to complex arrays
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/bridge.hpp>
#include <optinum/simd/complex.hpp>

#include <cmath>

namespace on = optinum::simd;
namespace dp = datapod;

// =============================================================================
// complex_view Basic Tests
// =============================================================================

TEST_CASE("complex_view - Construction and size queries") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[8];
    for (std::size_t i = 0; i < 8; ++i) {
        data[i] = complex_t{static_cast<double>(i), 0.0};
    }

    SUBCASE("From raw pointer") {
        auto cv = on::view(data, 8);
        CHECK(cv.size() == 8);
        CHECK_FALSE(cv.empty());
    }

    SUBCASE("From C-style array") {
        auto cv = on::view(data);
        CHECK(cv.size() == 8);
    }

    SUBCASE("Const view") {
        const complex_t *const_ptr = data;
        auto cv = on::view(const_ptr, 8);
        CHECK(cv.size() == 8);
    }
}

TEST_CASE("complex_view - Element access") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[4] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};

    auto cv = on::view(data);

    SUBCASE("Operator[]") {
        CHECK(cv[0].real == doctest::Approx(1.0));
        CHECK(cv[0].imag == doctest::Approx(2.0));
        CHECK(cv[3].real == doctest::Approx(7.0));
        CHECK(cv[3].imag == doctest::Approx(8.0));
    }

    SUBCASE("Data pointer") { CHECK(cv.data() == data); }

    SUBCASE("Iteration") {
        int count = 0;
        for (const auto &c : cv) {
            (void)c;
            count++;
        }
        CHECK(count == 4);
    }
}

// =============================================================================
// In-place Operations Tests
// =============================================================================

TEST_CASE("complex_view - Conjugate in-place") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[4] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, -6.0}, {0.0, 1.0}};

    auto cv = on::view(data);
    cv.conjugate_inplace();

    // Conjugate negates imaginary part
    CHECK(data[0].real == doctest::Approx(1.0));
    CHECK(data[0].imag == doctest::Approx(-2.0));

    CHECK(data[1].real == doctest::Approx(3.0));
    CHECK(data[1].imag == doctest::Approx(-4.0));

    CHECK(data[2].imag == doctest::Approx(6.0)); // was -6, now 6

    CHECK(data[3].real == doctest::Approx(0.0));
    CHECK(data[3].imag == doctest::Approx(-1.0));
}

TEST_CASE("complex_view - Normalize in-place") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[3] = {
        {3.0, 4.0}, // magnitude = 5
        {0.0, 2.0}, // magnitude = 2
        {1.0, 0.0}  // magnitude = 1 (already unit)
    };

    auto cv = on::view(data);
    cv.normalize_inplace();

    // Check normalized values
    CHECK(data[0].real == doctest::Approx(0.6)); // 3/5
    CHECK(data[0].imag == doctest::Approx(0.8)); // 4/5

    CHECK(data[1].real == doctest::Approx(0.0));
    CHECK(data[1].imag == doctest::Approx(1.0)); // 2/2

    CHECK(data[2].real == doctest::Approx(1.0));
    CHECK(data[2].imag == doctest::Approx(0.0));

    // Verify unit magnitudes
    for (int i = 0; i < 3; ++i) {
        double mag = std::sqrt(data[i].real * data[i].real + data[i].imag * data[i].imag);
        CHECK(mag == doctest::Approx(1.0));
    }
}

TEST_CASE("complex_view - Scale in-place") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[2] = {{1.0, 2.0}, {3.0, 4.0}};

    auto cv = on::view(data);
    cv.scale_inplace(2.0);

    CHECK(data[0].real == doctest::Approx(2.0));
    CHECK(data[0].imag == doctest::Approx(4.0));
    CHECK(data[1].real == doctest::Approx(6.0));
    CHECK(data[1].imag == doctest::Approx(8.0));
}

TEST_CASE("complex_view - Negate in-place") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[2] = {{1.0, 2.0}, {-3.0, 4.0}};

    auto cv = on::view(data);
    cv.negate_inplace();

    CHECK(data[0].real == doctest::Approx(-1.0));
    CHECK(data[0].imag == doctest::Approx(-2.0));
    CHECK(data[1].real == doctest::Approx(3.0));
    CHECK(data[1].imag == doctest::Approx(-4.0));
}

// =============================================================================
// Operations Returning New Array Tests
// =============================================================================

TEST_CASE("complex_view - Conjugate to output") {
    using complex_t = dp::mat::complex<double>;

    complex_t input[2] = {{1.0, 2.0}, {3.0, -4.0}};
    complex_t output[2];

    auto cv_in = on::view(input);
    (void)cv_in.conjugate_to(output);

    // Input unchanged
    CHECK(input[0].imag == doctest::Approx(2.0));

    // Output is conjugate
    CHECK(output[0].real == doctest::Approx(1.0));
    CHECK(output[0].imag == doctest::Approx(-2.0));
    CHECK(output[1].real == doctest::Approx(3.0));
    CHECK(output[1].imag == doctest::Approx(4.0));
}

// =============================================================================
// Binary Operations Tests
// =============================================================================

TEST_CASE("complex_view - Addition") {
    using complex_t = dp::mat::complex<double>;

    complex_t a[2] = {{1.0, 2.0}, {3.0, 4.0}};
    complex_t b[2] = {{5.0, 6.0}, {7.0, 8.0}};
    complex_t result[2];

    auto cv_a = on::view(a);
    auto cv_b = on::view(b);
    (void)cv_a.add_to(cv_b, result);

    CHECK(result[0].real == doctest::Approx(6.0));
    CHECK(result[0].imag == doctest::Approx(8.0));
    CHECK(result[1].real == doctest::Approx(10.0));
    CHECK(result[1].imag == doctest::Approx(12.0));
}

TEST_CASE("complex_view - Subtraction") {
    using complex_t = dp::mat::complex<double>;

    complex_t a[2] = {{5.0, 6.0}, {7.0, 8.0}};
    complex_t b[2] = {{1.0, 2.0}, {3.0, 4.0}};
    complex_t result[2];

    auto cv_a = on::view(a);
    auto cv_b = on::view(b);
    (void)cv_a.subtract_to(cv_b, result);

    CHECK(result[0].real == doctest::Approx(4.0));
    CHECK(result[0].imag == doctest::Approx(4.0));
    CHECK(result[1].real == doctest::Approx(4.0));
    CHECK(result[1].imag == doctest::Approx(4.0));
}

TEST_CASE("complex_view - Multiplication") {
    using complex_t = dp::mat::complex<double>;

    // (1 + 2i) * (3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
    complex_t a[2] = {{1.0, 2.0}, {1.0, 0.0}};
    complex_t b[2] = {{3.0, 4.0}, {0.0, 1.0}};
    complex_t result[2];

    auto cv_a = on::view(a);
    auto cv_b = on::view(b);
    (void)cv_a.multiply_to(cv_b, result);

    CHECK(result[0].real == doctest::Approx(-5.0));
    CHECK(result[0].imag == doctest::Approx(10.0));

    // 1 * i = i
    CHECK(result[1].real == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(result[1].imag == doctest::Approx(1.0));
}

TEST_CASE("complex_view - Division") {
    using complex_t = dp::mat::complex<double>;

    // (3 + 4i) / (1 + 2i) = (3 + 4i)(1 - 2i) / (1 + 4) = (11 - 2i) / 5 = 2.2 - 0.4i
    complex_t a[2] = {{3.0, 4.0}, {1.0, 0.0}};
    complex_t b[2] = {{1.0, 2.0}, {1.0, 0.0}};
    complex_t result[2];

    auto cv_a = on::view(a);
    auto cv_b = on::view(b);
    (void)cv_a.divide_to(cv_b, result);

    CHECK(result[0].real == doctest::Approx(2.2));
    CHECK(result[0].imag == doctest::Approx(-0.4));

    // 1 / 1 = 1
    CHECK(result[1].real == doctest::Approx(1.0));
    CHECK(result[1].imag == doctest::Approx(0.0).epsilon(1e-10));
}

// =============================================================================
// Reduction Operations Tests
// =============================================================================

TEST_CASE("complex_view - Magnitudes") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[3] = {
        {3.0, 4.0}, // magnitude = 5
        {0.0, 1.0}, // magnitude = 1
        {1.0, 1.0}  // magnitude = sqrt(2)
    };
    double mags[3];

    auto cv = on::view(data);
    cv.magnitudes_to(mags);

    CHECK(mags[0] == doctest::Approx(5.0));
    CHECK(mags[1] == doctest::Approx(1.0));
    CHECK(mags[2] == doctest::Approx(std::sqrt(2.0)));
}

TEST_CASE("complex_view - Phases") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[3] = {
        {1.0, 0.0}, // phase = 0
        {0.0, 1.0}, // phase = pi/2
        {-1.0, 0.0} // phase = pi
    };
    double phases[3];

    auto cv = on::view(data);
    cv.phases_to(phases);

    CHECK(phases[0] == doctest::Approx(0.0));
    CHECK(phases[1] == doctest::Approx(M_PI / 2.0));
    CHECK(phases[2] == doctest::Approx(M_PI));
}

TEST_CASE("complex_view - Real and imaginary parts") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[3] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    double reals[3], imags[3];

    auto cv = on::view(data);
    cv.real_parts_to(reals);
    cv.imag_parts_to(imags);

    CHECK(reals[0] == doctest::Approx(1.0));
    CHECK(reals[1] == doctest::Approx(3.0));
    CHECK(reals[2] == doctest::Approx(5.0));

    CHECK(imags[0] == doctest::Approx(2.0));
    CHECK(imags[1] == doctest::Approx(4.0));
    CHECK(imags[2] == doctest::Approx(6.0));
}

TEST_CASE("complex_view - Sum") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[4] = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};

    auto cv = on::view(data);
    auto s = cv.sum();

    CHECK(s.real == doctest::Approx(10.0));
    CHECK(s.imag == doctest::Approx(10.0));
}

TEST_CASE("complex_view - Dot product (Hermitian)") {
    using complex_t = dp::mat::complex<double>;

    // Hermitian inner product: <a, b> = sum(conj(a) * b)
    complex_t a[2] = {{1.0, 0.0}, {0.0, 1.0}}; // [1, i]
    complex_t b[2] = {{1.0, 0.0}, {0.0, 1.0}}; // [1, i]

    auto cv_a = on::view(a);
    auto cv_b = on::view(b);
    auto d = cv_a.dot(cv_b);

    // conj([1, i]) * [1, i] = [1, -i] * [1, i] = [1*1 + (-i)*i] = [1 + 1] = 2
    CHECK(d.real == doctest::Approx(2.0));
    CHECK(d.imag == doctest::Approx(0.0).epsilon(1e-10));
}

// =============================================================================
// Subview Tests
// =============================================================================

TEST_CASE("complex_view - Subview") {
    using complex_t = dp::mat::complex<double>;

    complex_t data[8];
    for (std::size_t i = 0; i < 8; ++i) {
        data[i] = {static_cast<double>(i), 0.0};
    }

    auto cv = on::view(data);
    auto sub = cv.subview(2, 4); // elements [2, 3, 4, 5]

    CHECK(sub.size() == 4);
    CHECK(sub[0].real == doctest::Approx(2.0));
    CHECK(sub[3].real == doctest::Approx(5.0));
}

// =============================================================================
// Tail Handling Tests (non-multiple of SIMD width)
// =============================================================================

TEST_CASE("complex_view - Tail handling") {
    using complex_t = dp::mat::complex<double>;

    // 5 complex numbers (not a multiple of typical SIMD width 4)
    complex_t data[5] = {{3.0, 4.0}, {0.0, 2.0}, {1.0, 0.0}, {1.0, 1.0}, {5.0, 0.0}};

    auto cv = on::view(data);

    SUBCASE("Normalize handles tail correctly") {
        cv.normalize_inplace();

        for (int i = 0; i < 5; ++i) {
            double mag = std::sqrt(data[i].real * data[i].real + data[i].imag * data[i].imag);
            CHECK(mag == doctest::Approx(1.0));
        }
    }

    SUBCASE("Sum handles tail correctly") {
        auto s = cv.sum();
        CHECK(s.real == doctest::Approx(10.0)); // 3+0+1+1+5
        CHECK(s.imag == doctest::Approx(7.0));  // 4+2+0+1+0
    }
}

// =============================================================================
// Complex Container (owning) Tests
// =============================================================================

TEST_CASE("Complex container - Basic operations") {
    on::Complex<double, 4> nums;

    SUBCASE("Default construction") { CHECK(nums.size() == 4); }

    SUBCASE("Zeros factory") {
        auto z = on::Complex<double, 4>::zeros();
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(z[i].real == doctest::Approx(0.0));
            CHECK(z[i].imag == doctest::Approx(0.0));
        }
    }

    SUBCASE("Ones factory") {
        auto o = on::Complex<double, 4>::ones();
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(o[i].real == doctest::Approx(1.0));
            CHECK(o[i].imag == doctest::Approx(0.0));
        }
    }

    SUBCASE("Unit imaginary factory") {
        auto ui = on::Complex<double, 4>::unit_imaginary();
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(ui[i].real == doctest::Approx(0.0));
            CHECK(ui[i].imag == doctest::Approx(1.0));
        }
    }

    SUBCASE("Conjugate in-place") {
        nums[0] = {1.0, 2.0};
        nums[1] = {3.0, 4.0};
        nums[2] = {5.0, 6.0};
        nums[3] = {7.0, 8.0};

        nums.conjugate_inplace();

        CHECK(nums[0].imag == doctest::Approx(-2.0));
        CHECK(nums[1].imag == doctest::Approx(-4.0));
    }

    SUBCASE("Binary operations") {
        on::Complex<double, 2> a, b;
        a[0] = {1.0, 2.0};
        a[1] = {3.0, 4.0};
        b[0] = {5.0, 6.0};
        b[1] = {7.0, 8.0};

        auto sum = a + b;
        CHECK(sum[0].real == doctest::Approx(6.0));
        CHECK(sum[0].imag == doctest::Approx(8.0));

        auto prod = a * b;
        // (1+2i)*(5+6i) = 5+6i+10i-12 = -7+16i
        CHECK(prod[0].real == doctest::Approx(-7.0));
        CHECK(prod[0].imag == doctest::Approx(16.0));
    }

    SUBCASE("Iteration") {
        auto o = on::Complex<double, 4>::ones();
        int count = 0;
        for (const auto &c : o) {
            CHECK(c.real == doctest::Approx(1.0));
            count++;
        }
        CHECK(count == 4);
    }

    SUBCASE("Sum reduction") {
        nums[0] = {1.0, 1.0};
        nums[1] = {2.0, 2.0};
        nums[2] = {3.0, 3.0};
        nums[3] = {4.0, 4.0};

        auto s = nums.sum();
        CHECK(s.real == doctest::Approx(10.0));
        CHECK(s.imag == doctest::Approx(10.0));
    }

    SUBCASE("Dot product") {
        on::Complex<double, 2> a, b;
        a[0] = {1.0, 0.0};
        a[1] = {0.0, 1.0};
        b[0] = {1.0, 0.0};
        b[1] = {0.0, 1.0};

        auto d = a.dot(b);
        CHECK(d.real == doctest::Approx(2.0));
        CHECK(d.imag == doctest::Approx(0.0).epsilon(1e-10));
    }
}
