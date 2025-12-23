#include <doctest/doctest.h>
#include <optinum/lina/decompose/svd.hpp>

using optinum::lina::svd;
using optinum::simd::Matrix;

template <typename T, std::size_t M, std::size_t N>
static Matrix<T, M, N> reconstruct(const optinum::lina::SVD<T, M, N> &f) {
    // A ~= U(:,0:K-1) * diag(S) * Vt
    constexpr std::size_t K = (M < N) ? M : N;

    Matrix<T, M, N> usv;
    usv.fill(T{});

    // Compute U(:,0:K-1) * diag(S) -> MxK
    Matrix<T, M, K> us;
    for (std::size_t j = 0; j < K; ++j) {
        for (std::size_t i = 0; i < M; ++i) {
            us(i, j) = f.u(i, j) * f.s[j];
        }
    }

    // Multiply (MxK) * (KxN) where Vt is NxN but we only need first K rows of Vt
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t i = 0; i < M; ++i) {
            T acc{};
            for (std::size_t k = 0; k < K; ++k) {
                acc += us(i, k) * f.vt(k, j);
            }
            usv(i, j) = acc;
        }
    }

    return usv;
}

TEST_CASE("lina::svd reconstructs A (3x2)") {
    Matrix<double, 3, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 2.0;
    a(2, 0) = 3.0;
    a(0, 1) = 4.0;
    a(1, 1) = 5.0;
    a(2, 1) = 6.0;

    const auto f = svd<double, 3, 2>(a, 64);
    const auto r = reconstruct(f);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            CHECK(r(i, j) == doctest::Approx(a(i, j)).epsilon(1e-6));
        }
    }
}

