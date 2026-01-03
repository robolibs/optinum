#include <cmath>
#include <doctest/doctest.h>
#include <optinum/simd/math/complex_math.hpp>

using optinum::simd::pack;
namespace dp = datapod;

TEST_CASE("Complex exp") {
    using complex_t = dp::mat::Complex<double>;
    using cpack = pack<complex_t, 2>;

    // exp(i*pi) = -1
    pack<double, 2> r, i;
    r.data_[0] = 0.0;
    i.data_[0] = M_PI;

    cpack z(r, i);
    auto result = exp(z);

    CHECK(result.real()[0] == doctest::Approx(-1.0).epsilon(1e-10));
    CHECK(result.imag()[0] == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("Complex log and exp inverse") {
    using complex_t = dp::mat::Complex<float>;
    using cpack = pack<complex_t, 4>;

    pack<float, 4> r, i;
    r.data_[0] = 2.0f;
    i.data_[0] = 3.0f;
    r.data_[1] = 1.0f;
    i.data_[1] = 1.0f;

    cpack z(r, i);
    auto log_z = log(z);
    auto exp_log_z = exp(log_z);

    // exp(log(z)) = z (complex math has ~3-5 ULP accuracy)
    CHECK(exp_log_z.real()[0] == doctest::Approx(2.0f).epsilon(1e-3));
    CHECK(exp_log_z.imag()[0] == doctest::Approx(3.0f).epsilon(1e-3));
    CHECK(exp_log_z.real()[1] == doctest::Approx(1.0f).epsilon(1e-3));
    CHECK(exp_log_z.imag()[1] == doctest::Approx(1.0f).epsilon(1e-3));
}

TEST_CASE("Complex sqrt") {
    using complex_t = dp::mat::Complex<double>;
    using cpack = pack<complex_t, 2>;

    // sqrt(-1) = i
    pack<double, 2> r, i;
    r.data_[0] = -1.0;
    i.data_[0] = 0.0;

    cpack z(r, i);
    auto result = sqrt(z);

    CHECK(result.real()[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(result.imag()[0] == doctest::Approx(1.0).epsilon(1e-10));
}

TEST_CASE("Complex sin and cos") {
    using complex_t = dp::mat::Complex<float>;
    using cpack = pack<complex_t, 4>;

    pack<float, 4> r, i;
    r.data_[0] = 1.0f;
    i.data_[0] = 0.0f; // Real value

    cpack z(r, i);
    auto sin_z = sin(z);
    auto cos_z = cos(z);

    // For real values, should match std::sin and std::cos (within SIMD tolerance)
    CHECK(sin_z.real()[0] == doctest::Approx(std::sin(1.0f)).epsilon(1e-4));
    CHECK(sin_z.imag()[0] == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(cos_z.real()[0] == doctest::Approx(std::cos(1.0f)).epsilon(1e-4));
    CHECK(cos_z.imag()[0] == doctest::Approx(0.0f).epsilon(1e-5));

    // sin²(z) + cos²(z) = 1
    auto sin_sq = sin_z * sin_z;
    auto cos_sq = cos_z * cos_z;
    auto sum = sin_sq + cos_sq;
    CHECK(sum.real()[0] == doctest::Approx(1.0f).epsilon(1e-3));
    CHECK(sum.imag()[0] == doctest::Approx(0.0f).epsilon(1e-3));
}

TEST_CASE("Complex sinh and cosh") {
    using complex_t = dp::mat::Complex<double>;
    using cpack = pack<complex_t, 2>;

    pack<double, 2> r, i;
    r.data_[0] = 0.5;
    i.data_[0] = 0.0;

    cpack z(r, i);
    auto sinh_z = sinh(z);
    auto cosh_z = cosh(z);

    // For real values
    CHECK(sinh_z.real()[0] == doctest::Approx(std::sinh(0.5)).epsilon(1e-10));
    CHECK(sinh_z.imag()[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(cosh_z.real()[0] == doctest::Approx(std::cosh(0.5)).epsilon(1e-10));
    CHECK(cosh_z.imag()[0] == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("Complex asin") {
    using complex_t = dp::mat::Complex<float>;
    using cpack = pack<complex_t, 4>;

    pack<float, 4> r, i;
    r.data_[0] = 0.5f;
    i.data_[0] = 0.0f;

    cpack z(r, i);
    auto asin_z = asin(z);

    // asin(0.5) ≈ 0.5236 (pi/6)
    CHECK(asin_z.real()[0] == doctest::Approx(std::asin(0.5f)).epsilon(1e-5));
    CHECK(asin_z.imag()[0] == doctest::Approx(0.0f).epsilon(1e-5));
}

TEST_CASE("Complex power") {
    using complex_t = dp::mat::Complex<double>;
    using cpack = pack<complex_t, 2>;

    // (1+i)^2 = 1 + 2i - 1 = 2i
    pack<double, 2> r1, i1, r2, i2;
    r1.data_[0] = 1.0;
    i1.data_[0] = 1.0;
    r2.data_[0] = 2.0;
    i2.data_[0] = 0.0;

    cpack z(r1, i1);
    cpack w(r2, i2);
    auto result = pow(z, w);

    CHECK(result.real()[0] == doctest::Approx(0.0).epsilon(1e-3));
    CHECK(result.imag()[0] == doctest::Approx(2.0).epsilon(1e-3));
}
