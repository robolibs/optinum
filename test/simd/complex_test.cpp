#include <doctest/doctest.h>
#include <optinum/simd/complex.hpp>

using optinum::simd::Complex;
namespace dp = datapod;

TEST_CASE("Complex construction and element access") {
    Complex<double, 4> c;
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
    auto zeros = Complex<float, 3>::zeros();
    CHECK(zeros[0].real == doctest::Approx(0.0f));
    CHECK(zeros[0].imag == doctest::Approx(0.0f));
    CHECK(zeros[2].real == doctest::Approx(0.0f));

    auto ones = Complex<float, 3>::ones();
    CHECK(ones[0].real == doctest::Approx(1.0f));
    CHECK(ones[0].imag == doctest::Approx(0.0f));

    auto imag = Complex<float, 3>::unit_imaginary();
    CHECK(imag[0].real == doctest::Approx(0.0f));
    CHECK(imag[0].imag == doctest::Approx(1.0f));
}

TEST_CASE("Complex arithmetic") {
    Complex<double, 2> a, b;
    a[0] = dp::mat::complex<double>{1.0, 2.0};
    a[1] = dp::mat::complex<double>{3.0, 4.0};
    b[0] = dp::mat::complex<double>{5.0, 6.0};
    b[1] = dp::mat::complex<double>{7.0, 8.0};

    auto sum = a + b;
    CHECK(sum[0].real == doctest::Approx(6.0));
    CHECK(sum[0].imag == doctest::Approx(8.0));

    auto diff = a - b;
    CHECK(diff[0].real == doctest::Approx(-4.0));
    CHECK(diff[0].imag == doctest::Approx(-4.0));

    auto prod = a * b;
    // (1+2i)*(5+6i) = 5 + 6i + 10i - 12 = -7 + 16i
    CHECK(prod[0].real == doctest::Approx(-7.0));
    CHECK(prod[0].imag == doctest::Approx(16.0));
}

TEST_CASE("Complex conjugate") {
    Complex<double, 2> c;
    c[0] = dp::mat::complex<double>{3.0, 4.0};
    c[1] = dp::mat::complex<double>{-1.0, 2.0};

    auto conj = c.conjugate();
    CHECK(conj[0].real == doctest::Approx(3.0));
    CHECK(conj[0].imag == doctest::Approx(-4.0));
    CHECK(conj[1].real == doctest::Approx(-1.0));
    CHECK(conj[1].imag == doctest::Approx(-2.0));
}

TEST_CASE("Complex real/imag extraction") {
    Complex<double, 3> c;
    c[0] = dp::mat::complex<double>{1.0, 2.0};
    c[1] = dp::mat::complex<double>{3.0, 4.0};
    c[2] = dp::mat::complex<double>{5.0, 6.0};

    double reals[3], imags[3];
    c.real_parts(reals);
    c.imag_parts(imags);

    CHECK(reals[0] == doctest::Approx(1.0));
    CHECK(reals[1] == doctest::Approx(3.0));
    CHECK(reals[2] == doctest::Approx(5.0));
    CHECK(imags[0] == doctest::Approx(2.0));
    CHECK(imags[1] == doctest::Approx(4.0));
    CHECK(imags[2] == doctest::Approx(6.0));
}
