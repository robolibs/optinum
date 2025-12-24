#include <doctest/doctest.h>
#include <optinum/lina/decompose/eigen.hpp>

using optinum::lina::eigen_sym;
using optinum::simd::Matrix;
using optinum::simd::Vector;

TEST_CASE("lina::eigen_sym A*v ~= lambda*v") {
    Matrix<double, 3, 3> a;
    // Symmetric matrix
    a(0, 0) = 4.0;
    a(1, 0) = 1.0;
    a(2, 0) = 1.0;
    a(0, 1) = 1.0;
    a(1, 1) = 3.0;
    a(2, 1) = 0.0;
    a(0, 2) = 1.0;
    a(1, 2) = 0.0;
    a(2, 2) = 2.0;

    const auto e = eigen_sym<double, 3>(a, 128);

    for (std::size_t k = 0; k < 3; ++k) {
        Vector<double, 3> v;
        for (std::size_t i = 0; i < 3; ++i) {
            v[i] = e.vectors(i, k);
        }
        const auto av = a * v;
        for (std::size_t i = 0; i < 3; ++i) {
            CHECK(av[i] == doctest::Approx(e.values[k] * v[i]).epsilon(1e-6));
        }
    }
}

