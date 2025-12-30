#include <doctest/doctest.h>
#include <optinum/lina/decompose/eigen.hpp>
#include <optinum/simd/backend/matmul.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

TEST_CASE("lina::eigen_sym A*v ~= lambda*v") {
    dp::mat::matrix<double, 3, 3> a;
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

    const auto e = lina::eigen_sym<double, 3>(Matrix<double, 3, 3>(a), 128);

    for (std::size_t k = 0; k < 3; ++k) {
        dp::mat::vector<double, 3> v;
        for (std::size_t i = 0; i < 3; ++i) {
            v[i] = e.vectors(i, k);
        }
        // Compute av = A * v using matvec_to
        dp::mat::vector<double, 3> av;
        optinum::simd::matvec_to(av.data(), Matrix<double, 3, 3>(a), Vector<double, 3>(v));
        for (std::size_t i = 0; i < 3; ++i) {
            CHECK(av[i] == doctest::Approx(e.values[k] * v[i]).epsilon(1e-6));
        }
    }
}
