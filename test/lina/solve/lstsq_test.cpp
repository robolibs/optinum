#include <doctest/doctest.h>
#include <optinum/lina/basic/matmul.hpp>
#include <optinum/lina/solve/lstsq.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;


TEST_CASE("lina::lstsq recovers exact solution for consistent overdetermined system") {
    // A (3x2), x (2), b = A x
    dp::mat::Matrix<double, 3, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 2.0;
    a(2, 0) = 3.0;
    a(0, 1) = 4.0;
    a(1, 1) = 5.0;
    a(2, 1) = 6.0;

    dp::mat::Vector<double, 2> x_true;
    x_true[0] = 1.0;
    x_true[1] = -1.0;

    auto b = lina::matmul(Matrix<double, 3, 2>(a), Vector<double, 2>(x_true));
    const auto x = lina::lstsq(Matrix<double, 3, 2>(a), Vector<double, 3>(b));

    CHECK(x[0] == doctest::Approx(x_true[0]).epsilon(1e-6));
    CHECK(x[1] == doctest::Approx(x_true[1]).epsilon(1e-6));
}
