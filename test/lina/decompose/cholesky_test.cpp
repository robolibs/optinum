#include <doctest/doctest.h>
#include <optinum/lina/decompose/cholesky.hpp>

using optinum::lina::cholesky;
using optinum::simd::Matrix;

TEST_CASE("lina::cholesky reconstructs A = L*L^T") {
    Matrix<double, 3, 3> a;
    // SPD matrix
    a(0, 0) = 4.0;
    a(1, 0) = 12.0;
    a(2, 0) = -16.0;
    a(0, 1) = 12.0;
    a(1, 1) = 37.0;
    a(2, 1) = -43.0;
    a(0, 2) = -16.0;
    a(1, 2) = -43.0;
    a(2, 2) = 98.0;

    const auto f = cholesky(a);
    CHECK(f.success);

    const auto lt = optinum::simd::transpose(f.l);
    const auto recon = f.l * lt;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(recon(i, j) == doctest::Approx(a(i, j)).epsilon(1e-8));
        }
    }
}

