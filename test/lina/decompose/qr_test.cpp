#include <doctest/doctest.h>
#include <optinum/lina/basic/matmul.hpp>
#include <optinum/lina/basic/transpose.hpp>
#include <optinum/lina/decompose/qr.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;


TEST_CASE("lina::qr reconstructs A ~= Q*R") {
    dp::mat::Matrix<double, 3, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 2.0;
    a(2, 0) = 3.0;
    a(0, 1) = 4.0;
    a(1, 1) = 5.0;
    a(2, 1) = 6.0;

    const auto f = lina::qr(Matrix<double, 3, 2>(a));
    const auto qr_a = lina::matmul(f.q, f.r);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            CHECK(qr_a(i, j) == doctest::Approx(a(i, j)).epsilon(1e-6));
        }
    }
}

TEST_CASE("lina::qr Q is approximately orthonormal") {
    dp::mat::Matrix<double, 3, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 2.0;
    a(2, 0) = 3.0;
    a(0, 1) = 4.0;
    a(1, 1) = 5.0;
    a(2, 1) = 6.0;

    const auto f = lina::qr(Matrix<double, 3, 2>(a));
    const auto qt = lina::transpose(f.q);
    const auto i3 = lina::matmul(qt, f.q);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            const double expected = (i == j) ? 1.0 : 0.0;
            CHECK(i3(i, j) == doctest::Approx(expected).epsilon(1e-6));
        }
    }
}
