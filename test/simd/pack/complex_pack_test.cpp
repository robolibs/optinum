#include <doctest/doctest.h>
#include <optinum/simd/pack/complex.hpp>

using optinum::simd::pack;
namespace dp = datapod;

TEST_CASE("Complex pack construction") {
    using complex_t = dp::mat::Complex<float>;
    using cpack = pack<complex_t, 4>;

    // Zero initialization
    auto z = cpack::zero();
    CHECK(z.real()[0] == doctest::Approx(0.0f));
    CHECK(z.imag()[0] == doctest::Approx(0.0f));

    // Broadcast single value
    cpack p(complex_t{3.0f, 4.0f});
    CHECK(p.real()[0] == doctest::Approx(3.0f));
    CHECK(p.imag()[0] == doctest::Approx(4.0f));

    // From real and imaginary packs
    pack<float, 4> r(1.0f);
    pack<float, 4> i(2.0f);
    cpack p2(r, i);
    CHECK(p2.real()[0] == doctest::Approx(1.0f));
    CHECK(p2.imag()[0] == doctest::Approx(2.0f));
}

TEST_CASE("Complex pack arithmetic") {
    using complex_t = dp::mat::Complex<double>;
    using cpack = pack<complex_t, 2>;

    // Create two complex packs
    pack<double, 2> r1, i1, r2, i2;
    r1.data_[0] = 1.0;
    r1.data_[1] = 3.0;
    i1.data_[0] = 2.0;
    i1.data_[1] = 4.0;
    r2.data_[0] = 5.0;
    r2.data_[1] = 7.0;
    i2.data_[0] = 6.0;
    i2.data_[1] = 8.0;

    cpack a(r1, i1); // [1+2i, 3+4i]
    cpack b(r2, i2); // [5+6i, 7+8i]

    // Addition
    auto sum = a + b;
    CHECK(sum.real()[0] == doctest::Approx(6.0));  // 1+5
    CHECK(sum.imag()[0] == doctest::Approx(8.0));  // 2+6
    CHECK(sum.real()[1] == doctest::Approx(10.0)); // 3+7
    CHECK(sum.imag()[1] == doctest::Approx(12.0)); // 4+8

    // Subtraction
    auto diff = a - b;
    CHECK(diff.real()[0] == doctest::Approx(-4.0)); // 1-5
    CHECK(diff.imag()[0] == doctest::Approx(-4.0)); // 2-6

    // Multiplication: (1+2i)*(5+6i) = 5 + 6i + 10i - 12 = -7 + 16i
    auto prod = a * b;
    CHECK(prod.real()[0] == doctest::Approx(-7.0));
    CHECK(prod.imag()[0] == doctest::Approx(16.0));
}

TEST_CASE("Complex pack conjugate and magnitude") {
    using complex_t = dp::mat::Complex<float>;
    using cpack = pack<complex_t, 4>;

    pack<float, 4> r, i;
    r.data_[0] = 3.0f;
    i.data_[0] = 4.0f; // 3+4i, |z| = 5
    r.data_[1] = 5.0f;
    i.data_[1] = 12.0f; // 5+12i, |z| = 13
    r.data_[2] = 0.0f;
    i.data_[2] = 1.0f; // i, |z| = 1
    r.data_[3] = 1.0f;
    i.data_[3] = 0.0f; // 1, |z| = 1

    cpack z(r, i);

    // Conjugate
    auto conj = z.conjugate();
    CHECK(conj.real()[0] == doctest::Approx(3.0f));
    CHECK(conj.imag()[0] == doctest::Approx(-4.0f));
    CHECK(conj.real()[1] == doctest::Approx(5.0f));
    CHECK(conj.imag()[1] == doctest::Approx(-12.0f));

    // Magnitude
    auto mag = z.magnitude();
    CHECK(mag[0] == doctest::Approx(5.0f));
    CHECK(mag[1] == doctest::Approx(13.0f));
    CHECK(mag[2] == doctest::Approx(1.0f));
    CHECK(mag[3] == doctest::Approx(1.0f));

    // Magnitude squared
    auto mag2 = z.magnitude_squared();
    CHECK(mag2[0] == doctest::Approx(25.0f));
    CHECK(mag2[1] == doctest::Approx(169.0f));
}

TEST_CASE("Complex pack interleaved load/store") {
    using complex_t = dp::mat::Complex<double>;
    using cpack = pack<complex_t, 4>;

    complex_t data[4] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};

    // Load interleaved
    auto p = cpack::loadu_interleaved(data);
    CHECK(p.real()[0] == doctest::Approx(1.0));
    CHECK(p.imag()[0] == doctest::Approx(2.0));
    CHECK(p.real()[3] == doctest::Approx(7.0));
    CHECK(p.imag()[3] == doctest::Approx(8.0));

    // Store interleaved
    complex_t output[4];
    p.storeu_interleaved(output);
    CHECK(output[0].real == doctest::Approx(1.0));
    CHECK(output[0].imag == doctest::Approx(2.0));
    CHECK(output[3].real == doctest::Approx(7.0));
    CHECK(output[3].imag == doctest::Approx(8.0));
}

TEST_CASE("Complex pack split load/store") {
    using complex_t = dp::mat::Complex<float>;
    using cpack = pack<complex_t, 4>;

    float reals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float imags[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    // Load split
    auto p = cpack::loadu_split(reals, imags);
    CHECK(p.real()[0] == doctest::Approx(1.0f));
    CHECK(p.imag()[0] == doctest::Approx(5.0f));
    CHECK(p.real()[3] == doctest::Approx(4.0f));
    CHECK(p.imag()[3] == doctest::Approx(8.0f));

    // Store split
    float out_real[4], out_imag[4];
    p.storeu_split(out_real, out_imag);
    CHECK(out_real[0] == doctest::Approx(1.0f));
    CHECK(out_imag[0] == doctest::Approx(5.0f));
    CHECK(out_real[3] == doctest::Approx(4.0f));
    CHECK(out_imag[3] == doctest::Approx(8.0f));
}
