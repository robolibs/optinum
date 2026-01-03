#include <doctest/doctest.h>
#include <optinum/lina/decompose/cholesky.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;

namespace dp = datapod;

TEST_CASE("lina::cholesky reconstructs A = L*L^T") {
    dp::mat::Matrix<double, 3, 3> a;
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

    const auto f = lina::cholesky(Matrix<double, 3, 3>(a));
    CHECK(f.success);

    // Manually compute L * L^T
    dp::mat::Matrix<double, 3, 3> recon;
    for (std::size_t i = 0; i < 9; ++i)
        recon[i] = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                // recon(i,j) += L(i,k) * L^T(k,j) = L(i,k) * L(j,k)
                recon(i, j) += f.l(i, k) * f.l(j, k);
            }
        }
    }
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(recon(i, j) == doctest::Approx(a(i, j)).epsilon(1e-8));
        }
    }
}
