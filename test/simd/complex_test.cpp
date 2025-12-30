#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace simd = optinum::simd;

TEST_CASE("Complex construction and element access") {
    dp::mat::vector<dp::mat::complex<double>, 4> c;
    c[0] = dp::mat::complex<double>{1.0, 2.0};
    c[1] = dp::mat::complex<double>{3.0, 4.0};
    c[2] = dp::mat::complex<double>{5.0, 6.0};
    c[3] = dp::mat::complex<double>{7.0, 8.0};

    CHECK(c[0].real == doctest::Approx(1.0));
    CHECK(c[0].imag == doctest::Approx(2.0));
    CHECK(c[1].real == doctest::Approx(3.0));
    CHECK(c[1].imag == doctest::Approx(4.0));
}

TEST_CASE("Complex factory functions") {
    dp::mat::vector<dp::mat::complex<float>, 3> zeros;
    for (std::size_t i = 0; i < 3; ++i) {
        zeros[i] = dp::mat::complex<float>{0.0f, 0.0f};
    }
    CHECK(zeros[0].real == doctest::Approx(0.0f));
    CHECK(zeros[0].imag == doctest::Approx(0.0f));
    CHECK(zeros[2].real == doctest::Approx(0.0f));

    dp::mat::vector<dp::mat::complex<float>, 3> ones;
    for (std::size_t i = 0; i < 3; ++i) {
        ones[i] = dp::mat::complex<float>{1.0f, 0.0f};
    }
    CHECK(ones[0].real == doctest::Approx(1.0f));
    CHECK(ones[0].imag == doctest::Approx(0.0f));

    dp::mat::vector<dp::mat::complex<float>, 3> unit_imag;
    for (std::size_t i = 0; i < 3; ++i) {
        unit_imag[i] = dp::mat::complex<float>{0.0f, 1.0f};
    }
    CHECK(unit_imag[0].real == doctest::Approx(0.0f));
    CHECK(unit_imag[0].imag == doctest::Approx(1.0f));
}

TEST_CASE("Complex arithmetic") {
    dp::mat::vector<dp::mat::complex<double>, 2> a, b;
    a[0] = dp::mat::complex<double>{1.0, 2.0};
    a[1] = dp::mat::complex<double>{3.0, 4.0};
    b[0] = dp::mat::complex<double>{5.0, 6.0};
    b[1] = dp::mat::complex<double>{7.0, 8.0};

    // Addition
    dp::mat::vector<dp::mat::complex<double>, 2> sum;
    for (std::size_t i = 0; i < 2; ++i) {
        sum[i] = dp::mat::complex<double>{a[i].real + b[i].real, a[i].imag + b[i].imag};
    }
    CHECK(sum[0].real == doctest::Approx(6.0));
    CHECK(sum[0].imag == doctest::Approx(8.0));

    // Subtraction
    dp::mat::vector<dp::mat::complex<double>, 2> diff;
    for (std::size_t i = 0; i < 2; ++i) {
        diff[i] = dp::mat::complex<double>{a[i].real - b[i].real, a[i].imag - b[i].imag};
    }
    CHECK(diff[0].real == doctest::Approx(-4.0));
    CHECK(diff[0].imag == doctest::Approx(-4.0));

    // Multiplication: (1+2i)*(5+6i) = 5 + 6i + 10i - 12 = -7 + 16i
    dp::mat::vector<dp::mat::complex<double>, 2> prod;
    for (std::size_t i = 0; i < 2; ++i) {
        prod[i] = dp::mat::complex<double>{a[i].real * b[i].real - a[i].imag * b[i].imag,
                                           a[i].real * b[i].imag + a[i].imag * b[i].real};
    }
    CHECK(prod[0].real == doctest::Approx(-7.0));
    CHECK(prod[0].imag == doctest::Approx(16.0));
}

TEST_CASE("Complex conjugate") {
    dp::mat::vector<dp::mat::complex<double>, 2> c;
    c[0] = dp::mat::complex<double>{3.0, 4.0};
    c[1] = dp::mat::complex<double>{-1.0, 2.0};

    dp::mat::vector<dp::mat::complex<double>, 2> conj;
    for (std::size_t i = 0; i < 2; ++i) {
        conj[i] = dp::mat::complex<double>{c[i].real, -c[i].imag};
    }
    CHECK(conj[0].real == doctest::Approx(3.0));
    CHECK(conj[0].imag == doctest::Approx(-4.0));
    CHECK(conj[1].real == doctest::Approx(-1.0));
    CHECK(conj[1].imag == doctest::Approx(-2.0));
}

TEST_CASE("Complex real/imag extraction") {
    dp::mat::vector<dp::mat::complex<double>, 3> c;
    c[0] = dp::mat::complex<double>{1.0, 2.0};
    c[1] = dp::mat::complex<double>{3.0, 4.0};
    c[2] = dp::mat::complex<double>{5.0, 6.0};

    double reals[3], imags[3];
    for (std::size_t i = 0; i < 3; ++i) {
        reals[i] = c[i].real;
        imags[i] = c[i].imag;
    }

    CHECK(reals[0] == doctest::Approx(1.0));
    CHECK(reals[1] == doctest::Approx(3.0));
    CHECK(reals[2] == doctest::Approx(5.0));
    CHECK(imags[0] == doctest::Approx(2.0));
    CHECK(imags[1] == doctest::Approx(4.0));
    CHECK(imags[2] == doctest::Approx(6.0));
}

TEST_CASE("Complex scalar multiplication and division") {
    dp::mat::vector<dp::mat::complex<double>, 2> data;
    data[0] = dp::mat::complex<double>{2.0, 4.0};
    data[1] = dp::mat::complex<double>{6.0, 8.0};

    simd::Complex<double, 2> c(data);

    SUBCASE("operator*=(T scalar)") {
        c *= 2.0;
        CHECK(c[0].real == doctest::Approx(4.0));
        CHECK(c[0].imag == doctest::Approx(8.0));
        CHECK(c[1].real == doctest::Approx(12.0));
        CHECK(c[1].imag == doctest::Approx(16.0));
    }

    SUBCASE("operator/=(T scalar)") {
        c /= 2.0;
        CHECK(c[0].real == doctest::Approx(1.0));
        CHECK(c[0].imag == doctest::Approx(2.0));
        CHECK(c[1].real == doctest::Approx(3.0));
        CHECK(c[1].imag == doctest::Approx(4.0));
    }
}
