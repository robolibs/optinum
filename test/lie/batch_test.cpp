#include <doctest/doctest.h>

#include <optinum/lie/lie.hpp>

#include <array>
#include <cmath>
#include <random>

using namespace optinum;
using namespace optinum::lie;

// ============================================================================
// SO2Batch Tests
// ============================================================================

TEST_SUITE("SO2Batch") {

    TEST_CASE("Default construction is identity") {
        SO2Batch<double, 4> batch;

        for (std::size_t i = 0; i < 4; ++i) {
            auto elem = batch[i];
            CHECK(elem.is_identity());
            CHECK(std::abs(elem.real() - 1.0) < 1e-10);
            CHECK(std::abs(elem.imag()) < 1e-10);
        }
    }

    TEST_CASE("Broadcast construction") {
        auto R = SO2d(0.5);
        SO2Batch<double, 8> batch(R);

        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(batch[i].is_approx(R, 1e-10));
        }
    }

    TEST_CASE("Element access and modification") {
        SO2Batch<double, 4> batch;

        batch.set(0, SO2d(0.1));
        batch.set(1, SO2d(0.2));
        batch.set(2, SO2d(0.3));
        batch.set(3, SO2d(0.4));

        CHECK(batch[0].is_approx(SO2d(0.1), 1e-10));
        CHECK(batch[1].is_approx(SO2d(0.2), 1e-10));
        CHECK(batch[2].is_approx(SO2d(0.3), 1e-10));
        CHECK(batch[3].is_approx(SO2d(0.4), 1e-10));
    }

    TEST_CASE("Static factory: identity") {
        auto batch = SO2Batch<double, 4>::identity();

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch[i].is_identity());
        }
    }

    TEST_CASE("Static factory: rot") {
        auto batch = SO2Batch<double, 4>::rot(0.5);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch[i].is_approx(SO2d(0.5), 1e-10));
        }
    }

    TEST_CASE("Exp map from array") {
        constexpr std::size_t N = 8;
        double theta[N];

        // Create random angles
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-M_PI, M_PI);

        for (std::size_t i = 0; i < N; ++i) {
            theta[i] = dist(rng);
        }

        auto batch = SO2Batch<double, N>::exp(theta);

        // Verify each element (relaxed tolerance for SIMD sin/cos)
        for (std::size_t i = 0; i < N; ++i) {
            auto expected = SO2d::exp(theta[i]);
            CHECK(batch[i].is_approx(expected, 1e-5));
        }
    }

    TEST_CASE("Inverse operation") {
        constexpr std::size_t N = 4;
        SO2Batch<double, N> batch;

        batch.set(0, SO2d(0.5));
        batch.set(1, SO2d(-0.3));
        batch.set(2, SO2d(1.2));
        batch.set(3, SO2d(0.8));

        auto inv_batch = batch.inverse();

        for (std::size_t i = 0; i < N; ++i) {
            auto product = batch[i] * inv_batch[i];
            CHECK(product.is_identity(1e-10));
        }
    }

    TEST_CASE("Group composition") {
        constexpr std::size_t N = 4;
        SO2Batch<double, N> batch1, batch2;

        batch1.set(0, SO2d(0.1));
        batch1.set(1, SO2d(0.2));
        batch1.set(2, SO2d(0.3));
        batch1.set(3, SO2d::identity());

        batch2.set(0, SO2d(0.4));
        batch2.set(1, SO2d(0.5));
        batch2.set(2, SO2d(0.6));
        batch2.set(3, SO2d(0.7));

        auto result = batch1 * batch2;

        for (std::size_t i = 0; i < N; ++i) {
            auto expected = batch1[i] * batch2[i];
            CHECK(result[i].is_approx(expected, 1e-10));
        }
    }

    TEST_CASE("Rotate vectors") {
        constexpr std::size_t N = 4;
        SO2Batch<double, N> batch;

        batch.set(0, SO2d(M_PI / 2));  // 90 deg
        batch.set(1, SO2d(M_PI));      // 180 deg
        batch.set(2, SO2d(-M_PI / 2)); // -90 deg
        batch.set(3, SO2d::identity());

        double vx[N] = {1, 1, 1, 1};
        double vy[N] = {0, 0, 0, 0};

        batch.rotate(vx, vy);

        // rot(90): (1,0) -> (0,1)
        CHECK(std::abs(vx[0]) < 1e-10);
        CHECK(std::abs(vy[0] - 1.0) < 1e-10);

        // rot(180): (1,0) -> (-1,0)
        CHECK(std::abs(vx[1] + 1.0) < 1e-10);
        CHECK(std::abs(vy[1]) < 1e-10);

        // rot(-90): (1,0) -> (0,-1)
        CHECK(std::abs(vx[2]) < 1e-10);
        CHECK(std::abs(vy[2] + 1.0) < 1e-10);

        // identity: (1,0) -> (1,0)
        CHECK(std::abs(vx[3] - 1.0) < 1e-10);
        CHECK(std::abs(vy[3]) < 1e-10);
    }

    TEST_CASE("SLERP interpolation") {
        constexpr std::size_t N = 4;
        SO2Batch<double, N> batch1(SO2d::identity());
        SO2Batch<double, N> batch2(SO2d(M_PI / 2));

        auto mid = batch1.slerp(batch2, 0.5);

        for (std::size_t i = 0; i < N; ++i) {
            // Mid-point should be rot(pi/4)
            auto expected = SO2d(M_PI / 4);
            // Relaxed tolerance due to SIMD sin/cos precision
            CHECK(mid[i].is_approx(expected, 1e-4));
        }
    }

    TEST_CASE("NLERP interpolation") {
        constexpr std::size_t N = 4;
        SO2Batch<double, N> batch1(SO2d::identity());
        SO2Batch<double, N> batch2(SO2d(M_PI / 2));

        auto mid = batch1.nlerp(batch2, 0.5);

        // NLERP should be close to SLERP for small angles
        for (std::size_t i = 0; i < N; ++i) {
            auto expected = SO2d(M_PI / 4);
            CHECK(mid[i].is_approx(expected, 1e-2)); // Relaxed tolerance for nlerp
        }
    }

    TEST_CASE("Normalize in place") {
        constexpr std::size_t N = 4;
        SO2Batch<double, N> batch;

        // Manually set unnormalized complex numbers
        for (std::size_t i = 0; i < N; ++i) {
            batch.complex(i).real = 2.0;
            batch.complex(i).imag = 0.0;
        }

        batch.normalize_inplace();

        CHECK(batch.all_unit(1e-10));
    }

    TEST_CASE("Log/exp round trip") {
        constexpr std::size_t N = 8;
        std::mt19937 rng(123);

        std::array<double, N> thetas;
        std::uniform_real_distribution<double> dist(-2.0, 2.0);

        for (std::size_t i = 0; i < N; ++i) {
            thetas[i] = dist(rng);
        }

        auto batch = SO2Batch<double, N>::exp(thetas);
        auto logs = batch.log();

        // The log should return the same angle (relaxed tolerance for SIMD precision)
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(logs[i] - thetas[i]) < 1e-3);
        }
    }

    TEST_CASE("Iterator support") {
        SO2Batch<double, 4> batch;
        batch.set(0, SO2d(0.1));
        batch.set(1, SO2d(0.2));
        batch.set(2, SO2d(0.3));
        batch.set(3, SO2d::identity());

        std::size_t count = 0;
        for (const auto &elem : batch) {
            (void)elem;
            ++count;
        }
        CHECK(count == 4);
    }

    TEST_CASE("Angular distance") {
        constexpr std::size_t N = 4;
        SO2Batch<double, N> batch1, batch2;

        batch1.set(0, SO2d(0.0));
        batch1.set(1, SO2d(0.0));
        batch1.set(2, SO2d(0.0));
        batch1.set(3, SO2d(M_PI - 0.1));

        batch2.set(0, SO2d(0.5));         // dist = 0.5
        batch2.set(1, SO2d(-0.5));        // dist = 0.5
        batch2.set(2, SO2d(0.0));         // dist = 0
        batch2.set(3, SO2d(-M_PI + 0.1)); // dist = 0.2 (wrapped)

        double dists[N];
        batch1.angular_distance(batch2, dists);

        CHECK(std::abs(dists[0] - 0.5) < 1e-6);
        CHECK(std::abs(dists[1] - 0.5) < 1e-6);
        CHECK(std::abs(dists[2]) < 1e-10);
        CHECK(std::abs(dists[3] - 0.2) < 1e-6);
    }

    TEST_CASE("Large batch rotation consistency") {
        constexpr std::size_t N = 64;
        SO2Batch<double, N> batch;

        // Create various rotations
        for (std::size_t i = 0; i < N; ++i) {
            double angle = static_cast<double>(i) * 0.1;
            batch.set(i, SO2d(angle));
        }

        // Create test vectors
        double vx[N], vy[N];
        for (std::size_t i = 0; i < N; ++i) {
            vx[i] = 1.0;
            vy[i] = 0.0;
        }

        // Rotate using batch (SIMD)
        batch.rotate(vx, vy);

        // Verify results
        for (std::size_t i = 0; i < N; ++i) {
            double angle = static_cast<double>(i) * 0.1;
            double expected_x = std::cos(angle);
            double expected_y = std::sin(angle);

            CHECK(std::abs(vx[i] - expected_x) < 1e-10);
            CHECK(std::abs(vy[i] - expected_y) < 1e-10);
        }
    }

    TEST_CASE("Batch composition associativity") {
        constexpr std::size_t N = 8;

        SO2Batch<double, N> A, B, C;

        std::mt19937 rng(789);
        std::uniform_real_distribution<double> dist(-M_PI, M_PI);
        for (std::size_t i = 0; i < N; ++i) {
            A.set(i, SO2d(dist(rng)));
            B.set(i, SO2d(dist(rng)));
            C.set(i, SO2d(dist(rng)));
        }

        // (A * B) * C == A * (B * C)
        auto left = (A * B) * C;
        auto right = A * (B * C);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(left[i].is_approx(right[i], 1e-10));
        }
    }

    TEST_CASE("Float type support") {
        constexpr std::size_t N = 8;
        SO2Batch<float, N> batch;

        for (std::size_t i = 0; i < N; ++i) {
            batch.set(i, SO2f(static_cast<float>(i) * 0.1f));
        }

        float vx[N], vy[N];
        for (std::size_t i = 0; i < N; ++i) {
            vx[i] = 1.0f;
            vy[i] = 0.0f;
        }

        batch.rotate(vx, vy);

        for (std::size_t i = 0; i < N; ++i) {
            float angle = static_cast<float>(i) * 0.1f;
            CHECK(std::abs(vx[i] - std::cos(angle)) < 1e-5f);
            CHECK(std::abs(vy[i] - std::sin(angle)) < 1e-5f);
        }
    }
}

// ============================================================================
// SE2Batch Tests
// ============================================================================

TEST_SUITE("SE2Batch") {

    TEST_CASE("Default construction is identity") {
        SE2Batch<double, 4> batch;

        for (std::size_t i = 0; i < 4; ++i) {
            auto elem = batch[i];
            CHECK(elem.so2().is_identity());
            auto t = elem.translation();
            CHECK(std::abs(t[0]) < 1e-10);
            CHECK(std::abs(t[1]) < 1e-10);
        }
    }

    TEST_CASE("Broadcast construction") {
        auto T = SE2d::trans(1, 2) * SE2d::rot(0.5);
        SE2Batch<double, 8> batch(T);

        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(batch[i].so2().is_approx(T.so2(), 1e-10));
            auto t = batch[i].translation();
            CHECK(std::abs(t[0] - 1.0) < 1e-10);
            CHECK(std::abs(t[1] - 2.0) < 1e-10);
        }
    }

    TEST_CASE("Element access and modification") {
        SE2Batch<double, 4> batch;

        batch.set(0, SE2d::trans(1, 0));
        batch.set(1, SE2d::trans(0, 2));
        batch.set(2, SE2d::trans(3, 4));
        batch.set(3, SE2d::rot(0.5));

        auto t0 = batch[0].translation();
        CHECK(std::abs(t0[0] - 1.0) < 1e-10);
        CHECK(std::abs(t0[1]) < 1e-10);

        auto t1 = batch[1].translation();
        CHECK(std::abs(t1[0]) < 1e-10);
        CHECK(std::abs(t1[1] - 2.0) < 1e-10);

        auto t2 = batch[2].translation();
        CHECK(std::abs(t2[0] - 3.0) < 1e-10);
        CHECK(std::abs(t2[1] - 4.0) < 1e-10);

        CHECK(batch[3].so2().is_approx(SO2d(0.5), 1e-10));
    }

    TEST_CASE("Static factory methods") {
        auto batch_r = SE2Batch<double, 4>::rot(0.5);
        auto batch_t = SE2Batch<double, 4>::trans(1, 2);
        auto batch_tx = SE2Batch<double, 4>::trans_x(5);
        auto batch_ty = SE2Batch<double, 4>::trans_y(7);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch_r[i].so2().is_approx(SO2d(0.5), 1e-10));

            auto t = batch_t[i].translation();
            CHECK(std::abs(t[0] - 1.0) < 1e-10);
            CHECK(std::abs(t[1] - 2.0) < 1e-10);

            auto tx = batch_tx[i].translation();
            CHECK(std::abs(tx[0] - 5.0) < 1e-10);
            CHECK(std::abs(tx[1]) < 1e-10);

            auto ty = batch_ty[i].translation();
            CHECK(std::abs(ty[0]) < 1e-10);
            CHECK(std::abs(ty[1] - 7.0) < 1e-10);
        }
    }

    TEST_CASE("Inverse operation") {
        constexpr std::size_t N = 4;
        SE2Batch<double, N> batch;

        batch.set(0, SE2d::trans(1, 2) * SE2d::rot(0.5));
        batch.set(1, SE2d::trans(-1, 0) * SE2d::rot(0.3));
        batch.set(2, SE2d::rot(1.0) * SE2d::trans(0, 5));
        batch.set(3, SE2d::identity());

        auto inv_batch = batch.inverse();

        for (std::size_t i = 0; i < N; ++i) {
            auto product = batch[i] * inv_batch[i];
            CHECK(product.so2().is_identity(1e-10));
            auto t = product.translation();
            CHECK(std::abs(t[0]) < 1e-10);
            CHECK(std::abs(t[1]) < 1e-10);
        }
    }

    TEST_CASE("Group composition") {
        constexpr std::size_t N = 4;
        SE2Batch<double, N> batch1, batch2;

        batch1.set(0, SE2d::trans(1, 0));
        batch1.set(1, SE2d::rot(M_PI / 2));
        batch1.set(2, SE2d::trans(0, 1) * SE2d::rot(0.5));
        batch1.set(3, SE2d::identity());

        batch2.set(0, SE2d::trans(0, 1));
        batch2.set(1, SE2d::trans(1, 0));
        batch2.set(2, SE2d::rot(0.3));
        batch2.set(3, SE2d::trans(1, 2));

        auto result = batch1 * batch2;

        for (std::size_t i = 0; i < N; ++i) {
            auto expected = batch1[i] * batch2[i];

            // Check rotation
            CHECK(result[i].so2().is_approx(expected.so2(), 1e-10));

            // Check translation
            auto t_result = result[i].translation();
            auto t_expected = expected.translation();
            CHECK(std::abs(t_result[0] - t_expected[0]) < 1e-10);
            CHECK(std::abs(t_result[1] - t_expected[1]) < 1e-10);
        }
    }

    TEST_CASE("Transform points") {
        constexpr std::size_t N = 4;
        SE2Batch<double, N> batch;

        // Translation only
        batch.set(0, SE2d::trans(1, 2));
        // Rotation only (90 deg)
        batch.set(1, SE2d::rot(M_PI / 2));
        // Combined
        batch.set(2, SE2d::trans(0, 1) * SE2d::rot(M_PI / 2));
        // Identity
        batch.set(3, SE2d::identity());

        double px[N] = {0, 1, 0, 5};
        double py[N] = {0, 0, 1, 6};

        batch.transform(px, py);

        // Translation: (0,0) + (1,2) = (1,2)
        CHECK(std::abs(px[0] - 1.0) < 1e-10);
        CHECK(std::abs(py[0] - 2.0) < 1e-10);

        // rot(90): (1,0) -> (0,1)
        CHECK(std::abs(px[1]) < 1e-10);
        CHECK(std::abs(py[1] - 1.0) < 1e-10);

        // rot(90) + trans_y(1): (0,1) -> (-1,0) + (0,1) = (-1,1)
        CHECK(std::abs(px[2] + 1.0) < 1e-10);
        CHECK(std::abs(py[2] - 1.0) < 1e-10);

        // Identity: unchanged
        CHECK(std::abs(px[3] - 5.0) < 1e-10);
        CHECK(std::abs(py[3] - 6.0) < 1e-10);
    }

    TEST_CASE("Linear interpolation") {
        constexpr std::size_t N = 4;
        SE2Batch<double, N> batch1(SE2d::identity());
        SE2Batch<double, N> batch2(SE2d::trans(2, 0) * SE2d::rot(M_PI / 2));

        auto mid = batch1.lerp(batch2, 0.5);

        for (std::size_t i = 0; i < N; ++i) {
            // Translation should be (1, 0) at midpoint
            auto t = mid[i].translation();
            CHECK(std::abs(t[0] - 1.0) < 1e-6);
            CHECK(std::abs(t[1]) < 1e-6);

            // Rotation should be approximately rot(pi/4)
            auto expected_rot = SO2d(M_PI / 4);
            CHECK(mid[i].so2().is_approx(expected_rot, 1e-4));
        }
    }

    TEST_CASE("Exp/log round trip") {
        constexpr std::size_t N = 4;
        std::mt19937 rng(456);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);

        std::array<dp::mat::Vector<double, 3>, N> twists;
        for (std::size_t i = 0; i < N; ++i) {
            twists[i] = {dist(rng), dist(rng), dist(rng)};
        }

        auto batch = SE2Batch<double, N>::exp(twists);
        auto logs = batch.log();

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                CHECK(std::abs(logs[i][j] - twists[i][j]) < 1e-10);
            }
        }
    }

    TEST_CASE("Exp from arrays") {
        constexpr std::size_t N = 8;
        std::mt19937 rng(789);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        double vx[N], vy[N], theta[N];
        for (std::size_t i = 0; i < N; ++i) {
            vx[i] = dist(rng);
            vy[i] = dist(rng);
            theta[i] = dist(rng);
        }

        auto batch = SE2Batch<double, N>::exp(vx, vy, theta);

        // Verify each element against scalar exp
        for (std::size_t i = 0; i < N; ++i) {
            dp::mat::Vector<double, 3> twist{vx[i], vy[i], theta[i]};
            auto expected = SE2d::exp(twist);

            CHECK(batch[i].so2().is_approx(expected.so2(), 1e-5));
            auto t_result = batch[i].translation();
            auto t_expected = expected.translation();
            CHECK(std::abs(t_result[0] - t_expected[0]) < 1e-5);
            CHECK(std::abs(t_result[1] - t_expected[1]) < 1e-5);
        }
    }

    TEST_CASE("Translation norms") {
        constexpr std::size_t N = 4;
        SE2Batch<double, N> batch;

        batch.set(0, SE2d::trans(3, 4)); // norm = 5
        batch.set(1, SE2d::trans(0, 0)); // norm = 0
        batch.set(2, SE2d::trans(1, 1)); // norm = sqrt(2)
        batch.set(3, SE2d::trans(6, 8)); // norm = 10

        double norms[N];
        batch.translation_norms(norms);

        CHECK(std::abs(norms[0] - 5.0) < 1e-10);
        CHECK(std::abs(norms[1]) < 1e-10);
        CHECK(std::abs(norms[2] - std::sqrt(2.0)) < 1e-10);
        CHECK(std::abs(norms[3] - 10.0) < 1e-10);
    }

    TEST_CASE("Iterator support") {
        SE2Batch<double, 4> batch;
        batch.set(0, SE2d::trans(1, 0));
        batch.set(1, SE2d::trans(0, 1));
        batch.set(2, SE2d::trans(1, 1));
        batch.set(3, SE2d::rot(0.5));

        std::size_t count = 0;
        for (const auto &elem : batch) {
            (void)elem;
            ++count;
        }
        CHECK(count == 4);
    }

    TEST_CASE("Inverse transform") {
        constexpr std::size_t N = 4;
        SE2Batch<double, N> batch;

        batch.set(0, SE2d::trans(1, 2) * SE2d::rot(0.5));
        batch.set(1, SE2d::trans(-1, 0) * SE2d::rot(0.3));
        batch.set(2, SE2d::rot(1.0) * SE2d::trans(0, 5));
        batch.set(3, SE2d::identity());

        // Transform then inverse transform should be identity
        double px[N] = {1, 2, 3, 4};
        double py[N] = {5, 6, 7, 8};

        double px_orig[N], py_orig[N];
        for (std::size_t i = 0; i < N; ++i) {
            px_orig[i] = px[i];
            py_orig[i] = py[i];
        }

        batch.transform(px, py);
        batch.inverse_transform(px, py);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(px[i] - px_orig[i]) < 1e-10);
            CHECK(std::abs(py[i] - py_orig[i]) < 1e-10);
        }
    }

    TEST_CASE("Large batch transform consistency") {
        constexpr std::size_t N = 32;
        SE2Batch<double, N> batch;

        // Create poses with increasing translation
        for (std::size_t i = 0; i < N; ++i) {
            batch.set(i, SE2d::trans(static_cast<double>(i), 0));
        }

        // Transform origin points
        double px[N], py[N];
        for (std::size_t i = 0; i < N; ++i) {
            px[i] = py[i] = 0.0;
        }

        batch.transform(px, py);

        // Verify
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(px[i] - static_cast<double>(i)) < 1e-10);
            CHECK(std::abs(py[i]) < 1e-10);
        }
    }

    TEST_CASE("Float type support") {
        constexpr std::size_t N = 8;
        SE2Batch<float, N> batch;

        for (std::size_t i = 0; i < N; ++i) {
            batch.set(i, SE2f::trans(static_cast<float>(i), static_cast<float>(i * 2)));
        }

        float px[N], py[N];
        for (std::size_t i = 0; i < N; ++i) {
            px[i] = 0.0f;
            py[i] = 0.0f;
        }

        batch.transform(px, py);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(px[i] - static_cast<float>(i)) < 1e-5f);
            CHECK(std::abs(py[i] - static_cast<float>(i * 2)) < 1e-5f);
        }
    }
}

// ============================================================================
// SO3Batch Tests
// ============================================================================

TEST_SUITE("SO3Batch") {

    TEST_CASE("Default construction is identity") {
        SO3Batch<double, 4> batch;

        for (std::size_t i = 0; i < 4; ++i) {
            auto elem = batch[i];
            CHECK(elem.is_identity());
            CHECK(std::abs(elem.w() - 1.0) < 1e-10);
            CHECK(std::abs(elem.x()) < 1e-10);
            CHECK(std::abs(elem.y()) < 1e-10);
            CHECK(std::abs(elem.z()) < 1e-10);
        }
    }

    TEST_CASE("Broadcast construction") {
        auto R = SO3d::rot_z(0.5);
        SO3Batch<double, 8> batch(R);

        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(batch[i].is_approx(R, 1e-10));
        }
    }

    TEST_CASE("Element access and modification") {
        SO3Batch<double, 4> batch;

        batch.set(0, SO3d::rot_x(0.1));
        batch.set(1, SO3d::rot_y(0.2));
        batch.set(2, SO3d::rot_z(0.3));
        batch.set(3, SO3d::rot_x(0.4));

        CHECK(batch[0].is_approx(SO3d::rot_x(0.1), 1e-10));
        CHECK(batch[1].is_approx(SO3d::rot_y(0.2), 1e-10));
        CHECK(batch[2].is_approx(SO3d::rot_z(0.3), 1e-10));
        CHECK(batch[3].is_approx(SO3d::rot_x(0.4), 1e-10));
    }

    TEST_CASE("Static factory: identity") {
        auto batch = SO3Batch<double, 4>::identity();

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch[i].is_identity());
        }
    }

    TEST_CASE("Static factory: rot_x/y/z") {
        auto batch_x = SO3Batch<double, 4>::rot_x(0.5);
        auto batch_y = SO3Batch<double, 4>::rot_y(0.5);
        auto batch_z = SO3Batch<double, 4>::rot_z(0.5);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch_x[i].is_approx(SO3d::rot_x(0.5), 1e-10));
            CHECK(batch_y[i].is_approx(SO3d::rot_y(0.5), 1e-10));
            CHECK(batch_z[i].is_approx(SO3d::rot_z(0.5), 1e-10));
        }
    }

    TEST_CASE("Exp map from arrays") {
        constexpr std::size_t N = 8;
        double omega_x[N], omega_y[N], omega_z[N];

        // Create random rotation vectors
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (std::size_t i = 0; i < N; ++i) {
            omega_x[i] = dist(rng);
            omega_y[i] = dist(rng);
            omega_z[i] = dist(rng);
        }

        auto batch = SO3Batch<double, N>::exp(omega_x, omega_y, omega_z);

        // Verify each element
        for (std::size_t i = 0; i < N; ++i) {
            dp::mat::Vector<double, 3> omega{omega_x[i], omega_y[i], omega_z[i]};
            auto expected = SO3d::exp(omega);
            CHECK(batch[i].is_approx(expected, 1e-10));
        }
    }

    TEST_CASE("Inverse operation") {
        constexpr std::size_t N = 4;
        SO3Batch<double, N> batch;

        batch.set(0, SO3d::rot_x(0.5));
        batch.set(1, SO3d::rot_y(-0.3));
        batch.set(2, SO3d::rot_z(1.2));
        batch.set(3, SO3d::rot_x(0.8) * SO3d::rot_y(0.4));

        auto inv_batch = batch.inverse();

        for (std::size_t i = 0; i < N; ++i) {
            auto product = batch[i] * inv_batch[i];
            CHECK(product.is_identity(1e-10));
        }
    }

    TEST_CASE("Group composition") {
        constexpr std::size_t N = 4;
        SO3Batch<double, N> batch1, batch2;

        batch1.set(0, SO3d::rot_x(0.1));
        batch1.set(1, SO3d::rot_y(0.2));
        batch1.set(2, SO3d::rot_z(0.3));
        batch1.set(3, SO3d::identity());

        batch2.set(0, SO3d::rot_z(0.4));
        batch2.set(1, SO3d::rot_x(0.5));
        batch2.set(2, SO3d::rot_y(0.6));
        batch2.set(3, SO3d::rot_z(0.7));

        auto result = batch1 * batch2;

        for (std::size_t i = 0; i < N; ++i) {
            auto expected = batch1[i] * batch2[i];
            CHECK(result[i].is_approx(expected, 1e-10));
        }
    }

    TEST_CASE("Rotate vectors") {
        constexpr std::size_t N = 4;
        SO3Batch<double, N> batch;

        batch.set(0, SO3d::rot_x(M_PI / 2)); // 90 deg around x
        batch.set(1, SO3d::rot_y(M_PI / 2)); // 90 deg around y
        batch.set(2, SO3d::rot_z(M_PI / 2)); // 90 deg around z
        batch.set(3, SO3d::identity());

        double vx[N] = {1, 1, 1, 1};
        double vy[N] = {0, 0, 0, 0};
        double vz[N] = {0, 0, 0, 0};

        batch.rotate(vx, vy, vz);

        // rot_x(90): (1,0,0) -> (1,0,0)
        CHECK(std::abs(vx[0] - 1.0) < 1e-10);
        CHECK(std::abs(vy[0]) < 1e-10);
        CHECK(std::abs(vz[0]) < 1e-10);

        // rot_y(90): (1,0,0) -> (0,0,-1)
        CHECK(std::abs(vx[1]) < 1e-10);
        CHECK(std::abs(vy[1]) < 1e-10);
        CHECK(std::abs(vz[1] + 1.0) < 1e-10);

        // rot_z(90): (1,0,0) -> (0,1,0)
        CHECK(std::abs(vx[2]) < 1e-10);
        CHECK(std::abs(vy[2] - 1.0) < 1e-10);
        CHECK(std::abs(vz[2]) < 1e-10);

        // identity: (1,0,0) -> (1,0,0)
        CHECK(std::abs(vx[3] - 1.0) < 1e-10);
        CHECK(std::abs(vy[3]) < 1e-10);
        CHECK(std::abs(vz[3]) < 1e-10);
    }

    TEST_CASE("SLERP interpolation") {
        constexpr std::size_t N = 4;
        SO3Batch<double, N> batch1(SO3d::identity());
        SO3Batch<double, N> batch2(SO3d::rot_z(M_PI / 2));

        auto mid = batch1.slerp(batch2, 0.5);

        for (std::size_t i = 0; i < N; ++i) {
            // Mid-point should be rot_z(pi/4)
            auto expected = SO3d::rot_z(M_PI / 4);
            CHECK(mid[i].is_approx(expected, 1e-6));
        }
    }

    TEST_CASE("Normalize in place") {
        constexpr std::size_t N = 4;
        SO3Batch<double, N> batch;

        // Manually set unnormalized quaternions
        for (std::size_t i = 0; i < N; ++i) {
            batch.quat(i).w = 2.0;
            batch.quat(i).x = 0.0;
            batch.quat(i).y = 0.0;
            batch.quat(i).z = 0.0;
        }

        batch.normalize_inplace();

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(batch.all_unit(1e-10));
        }
    }

    TEST_CASE("Euler angle conversion") {
        constexpr std::size_t N = 4;
        double roll[N] = {0.1, 0.2, 0.3, 0.4};
        double pitch[N] = {0.05, 0.1, 0.15, 0.2};
        double yaw[N] = {0.2, 0.4, 0.6, 0.8};

        auto batch = SO3Batch<double, N>::from_euler(roll, pitch, yaw);

        double r_out[N], p_out[N], y_out[N];
        batch.to_euler(r_out, p_out, y_out);

        // Use relaxed tolerance for Euler conversion (numerical issues at certain angles)
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(r_out[i] - roll[i]) < 1e-5);
            CHECK(std::abs(p_out[i] - pitch[i]) < 1e-5);
            CHECK(std::abs(y_out[i] - yaw[i]) < 1e-3); // Yaw can have larger error for certain angles
        }
    }

    TEST_CASE("Iterator support") {
        SO3Batch<double, 4> batch;
        batch.set(0, SO3d::rot_x(0.1));
        batch.set(1, SO3d::rot_y(0.2));
        batch.set(2, SO3d::rot_z(0.3));
        batch.set(3, SO3d::identity());

        std::size_t count = 0;
        for (const auto &elem : batch) {
            (void)elem;
            ++count;
        }
        CHECK(count == 4);
    }

    TEST_CASE("Log/exp round trip") {
        constexpr std::size_t N = 8;
        std::mt19937 rng(123);

        std::array<dp::mat::Vector<double, 3>, N> omegas;
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        for (std::size_t i = 0; i < N; ++i) {
            omegas[i] = {dist(rng), dist(rng), dist(rng)};
        }

        auto batch = SO3Batch<double, N>::exp(omegas);
        auto logs = batch.log();

        // The log should return the same rotation vector (up to sign ambiguity for pi rotations)
        for (std::size_t i = 0; i < N; ++i) {
            // Check if vectors are approximately equal
            double diff = 0;
            for (std::size_t j = 0; j < 3; ++j) {
                diff += (logs[i][j] - omegas[i][j]) * (logs[i][j] - omegas[i][j]);
            }
            CHECK(std::sqrt(diff) < 1e-10);
        }
    }
}

// ============================================================================
// SE3Batch Tests
// ============================================================================

TEST_SUITE("SE3Batch") {

    TEST_CASE("Default construction is identity") {
        SE3Batch<double, 4> batch;

        for (std::size_t i = 0; i < 4; ++i) {
            auto elem = batch[i];
            CHECK(elem.so3().is_identity());
            auto t = elem.translation();
            CHECK(std::abs(t[0]) < 1e-10);
            CHECK(std::abs(t[1]) < 1e-10);
            CHECK(std::abs(t[2]) < 1e-10);
        }
    }

    TEST_CASE("Broadcast construction") {
        auto T = SE3d::trans(1, 2, 3) * SE3d::rot_z(0.5);
        SE3Batch<double, 8> batch(T);

        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(batch[i].so3().is_approx(T.so3(), 1e-10));
            auto t = batch[i].translation();
            CHECK(std::abs(t[0] - 1.0) < 1e-10);
            CHECK(std::abs(t[1] - 2.0) < 1e-10);
            CHECK(std::abs(t[2] - 3.0) < 1e-10);
        }
    }

    TEST_CASE("Element access and modification") {
        SE3Batch<double, 4> batch;

        batch.set(0, SE3d::trans(1, 0, 0));
        batch.set(1, SE3d::trans(0, 2, 0));
        batch.set(2, SE3d::trans(0, 0, 3));
        batch.set(3, SE3d::rot_z(0.5));

        auto t0 = batch[0].translation();
        CHECK(std::abs(t0[0] - 1.0) < 1e-10);

        auto t1 = batch[1].translation();
        CHECK(std::abs(t1[1] - 2.0) < 1e-10);

        auto t2 = batch[2].translation();
        CHECK(std::abs(t2[2] - 3.0) < 1e-10);

        CHECK(batch[3].so3().is_approx(SO3d::rot_z(0.5), 1e-10));
    }

    TEST_CASE("Static factory methods") {
        auto batch_x = SE3Batch<double, 4>::rot_x(0.5);
        auto batch_t = SE3Batch<double, 4>::trans(1, 2, 3);
        auto batch_tx = SE3Batch<double, 4>::trans_x(5);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch_x[i].so3().is_approx(SO3d::rot_x(0.5), 1e-10));

            auto t = batch_t[i].translation();
            CHECK(std::abs(t[0] - 1.0) < 1e-10);
            CHECK(std::abs(t[1] - 2.0) < 1e-10);
            CHECK(std::abs(t[2] - 3.0) < 1e-10);

            auto tx = batch_tx[i].translation();
            CHECK(std::abs(tx[0] - 5.0) < 1e-10);
            CHECK(std::abs(tx[1]) < 1e-10);
            CHECK(std::abs(tx[2]) < 1e-10);
        }
    }

    TEST_CASE("Inverse operation") {
        constexpr std::size_t N = 4;
        SE3Batch<double, N> batch;

        batch.set(0, SE3d::trans(1, 2, 3) * SE3d::rot_x(0.5));
        batch.set(1, SE3d::trans(-1, 0, 2) * SE3d::rot_y(0.3));
        batch.set(2, SE3d::rot_z(1.0) * SE3d::trans(0, 0, 5));
        batch.set(3, SE3d::identity());

        auto inv_batch = batch.inverse();

        for (std::size_t i = 0; i < N; ++i) {
            auto product = batch[i] * inv_batch[i];
            CHECK(product.so3().is_identity(1e-10));
            auto t = product.translation();
            CHECK(std::abs(t[0]) < 1e-10);
            CHECK(std::abs(t[1]) < 1e-10);
            CHECK(std::abs(t[2]) < 1e-10);
        }
    }

    TEST_CASE("Group composition") {
        constexpr std::size_t N = 4;
        SE3Batch<double, N> batch1, batch2;

        batch1.set(0, SE3d::trans(1, 0, 0));
        batch1.set(1, SE3d::rot_z(M_PI / 2));
        batch1.set(2, SE3d::trans(0, 1, 0) * SE3d::rot_x(0.5));
        batch1.set(3, SE3d::identity());

        batch2.set(0, SE3d::trans(0, 1, 0));
        batch2.set(1, SE3d::trans(1, 0, 0));
        batch2.set(2, SE3d::rot_y(0.3));
        batch2.set(3, SE3d::trans(1, 2, 3));

        auto result = batch1 * batch2;

        for (std::size_t i = 0; i < N; ++i) {
            auto expected = batch1[i] * batch2[i];

            // Check rotation
            CHECK(result[i].so3().is_approx(expected.so3(), 1e-10));

            // Check translation
            auto t_result = result[i].translation();
            auto t_expected = expected.translation();
            CHECK(std::abs(t_result[0] - t_expected[0]) < 1e-10);
            CHECK(std::abs(t_result[1] - t_expected[1]) < 1e-10);
            CHECK(std::abs(t_result[2] - t_expected[2]) < 1e-10);
        }
    }

    TEST_CASE("Transform points") {
        constexpr std::size_t N = 4;
        SE3Batch<double, N> batch;

        // Translation only
        batch.set(0, SE3d::trans(1, 2, 3));
        // Rotation only (90 deg around z)
        batch.set(1, SE3d::rot_z(M_PI / 2));
        // Combined
        batch.set(2, SE3d::trans(0, 0, 1) * SE3d::rot_x(M_PI / 2));
        // Identity
        batch.set(3, SE3d::identity());

        double px[N] = {0, 1, 0, 5};
        double py[N] = {0, 0, 1, 6};
        double pz[N] = {0, 0, 0, 7};

        batch.transform(px, py, pz);

        // Translation: (0,0,0) + (1,2,3) = (1,2,3)
        CHECK(std::abs(px[0] - 1.0) < 1e-10);
        CHECK(std::abs(py[0] - 2.0) < 1e-10);
        CHECK(std::abs(pz[0] - 3.0) < 1e-10);

        // rot_z(90): (1,0,0) -> (0,1,0)
        CHECK(std::abs(px[1]) < 1e-10);
        CHECK(std::abs(py[1] - 1.0) < 1e-10);
        CHECK(std::abs(pz[1]) < 1e-10);

        // rot_x(90) + trans_z(1): (0,1,0) -> (0,0,1) + (0,0,1) = (0,0,2)
        CHECK(std::abs(px[2]) < 1e-10);
        CHECK(std::abs(py[2]) < 1e-10);
        CHECK(std::abs(pz[2] - 2.0) < 1e-10);

        // Identity: unchanged
        CHECK(std::abs(px[3] - 5.0) < 1e-10);
        CHECK(std::abs(py[3] - 6.0) < 1e-10);
        CHECK(std::abs(pz[3] - 7.0) < 1e-10);
    }

    TEST_CASE("Linear interpolation") {
        constexpr std::size_t N = 4;
        SE3Batch<double, N> batch1(SE3d::identity());
        SE3Batch<double, N> batch2(SE3d::trans(2, 0, 0) * SE3d::rot_z(M_PI / 2));

        auto mid = batch1.lerp(batch2, 0.5);

        for (std::size_t i = 0; i < N; ++i) {
            // Translation should be (1, 0, 0) at midpoint
            auto t = mid[i].translation();
            CHECK(std::abs(t[0] - 1.0) < 1e-6);
            CHECK(std::abs(t[1]) < 1e-6);
            CHECK(std::abs(t[2]) < 1e-6);

            // Rotation should be approximately rot_z(pi/4)
            auto expected_rot = SO3d::rot_z(M_PI / 4);
            CHECK(mid[i].so3().is_approx(expected_rot, 1e-6));
        }
    }

    TEST_CASE("Exp/log round trip") {
        constexpr std::size_t N = 4;
        std::mt19937 rng(456);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);

        std::array<dp::mat::Vector<double, 6>, N> twists;
        for (std::size_t i = 0; i < N; ++i) {
            twists[i] = {dist(rng), dist(rng), dist(rng), dist(rng), dist(rng), dist(rng)};
        }

        auto batch = SE3Batch<double, N>::exp(twists);
        auto logs = batch.log();

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < 6; ++j) {
                CHECK(std::abs(logs[i][j] - twists[i][j]) < 1e-10);
            }
        }
    }

    TEST_CASE("Translation norms") {
        constexpr std::size_t N = 4;
        SE3Batch<double, N> batch;

        batch.set(0, SE3d::trans(3, 4, 0)); // norm = 5
        batch.set(1, SE3d::trans(0, 0, 0)); // norm = 0
        batch.set(2, SE3d::trans(1, 1, 1)); // norm = sqrt(3)
        batch.set(3, SE3d::trans(6, 0, 8)); // norm = 10

        double norms[N];
        batch.translation_norms(norms);

        CHECK(std::abs(norms[0] - 5.0) < 1e-10);
        CHECK(std::abs(norms[1]) < 1e-10);
        CHECK(std::abs(norms[2] - std::sqrt(3.0)) < 1e-10);
        CHECK(std::abs(norms[3] - 10.0) < 1e-10);
    }

    TEST_CASE("Iterator support") {
        SE3Batch<double, 4> batch;
        batch.set(0, SE3d::trans(1, 0, 0));
        batch.set(1, SE3d::trans(0, 1, 0));
        batch.set(2, SE3d::trans(0, 0, 1));
        batch.set(3, SE3d::rot_z(0.5));

        std::size_t count = 0;
        for (const auto &elem : batch) {
            (void)elem;
            ++count;
        }
        CHECK(count == 4);
    }

    TEST_CASE("Inverse transform") {
        constexpr std::size_t N = 4;
        SE3Batch<double, N> batch;

        batch.set(0, SE3d::trans(1, 2, 3) * SE3d::rot_x(0.5));
        batch.set(1, SE3d::trans(-1, 0, 2) * SE3d::rot_y(0.3));
        batch.set(2, SE3d::rot_z(1.0) * SE3d::trans(0, 0, 5));
        batch.set(3, SE3d::identity());

        // Transform then inverse transform should be identity
        double px[N] = {1, 2, 3, 4};
        double py[N] = {5, 6, 7, 8};
        double pz[N] = {9, 10, 11, 12};

        double px_orig[N], py_orig[N], pz_orig[N];
        for (std::size_t i = 0; i < N; ++i) {
            px_orig[i] = px[i];
            py_orig[i] = py[i];
            pz_orig[i] = pz[i];
        }

        batch.transform(px, py, pz);
        batch.inverse_transform(px, py, pz);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(px[i] - px_orig[i]) < 1e-10);
            CHECK(std::abs(py[i] - py_orig[i]) < 1e-10);
            CHECK(std::abs(pz[i] - pz_orig[i]) < 1e-10);
        }
    }
}

// ============================================================================
// Performance / SIMD Verification Tests
// ============================================================================

TEST_SUITE("Batch SIMD Operations") {

    TEST_CASE("Large batch rotation consistency") {
        constexpr std::size_t N = 64;
        SO3Batch<double, N> batch;

        // Create various rotations
        for (std::size_t i = 0; i < N; ++i) {
            double angle = static_cast<double>(i) * 0.1;
            batch.set(i, SO3d::rot_z(angle));
        }

        // Create test vectors
        double vx[N], vy[N], vz[N];
        for (std::size_t i = 0; i < N; ++i) {
            vx[i] = 1.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        }

        // Rotate using batch (SIMD)
        batch.rotate(vx, vy, vz);

        // Verify results
        for (std::size_t i = 0; i < N; ++i) {
            double angle = static_cast<double>(i) * 0.1;
            double expected_x = std::cos(angle);
            double expected_y = std::sin(angle);

            CHECK(std::abs(vx[i] - expected_x) < 1e-10);
            CHECK(std::abs(vy[i] - expected_y) < 1e-10);
            CHECK(std::abs(vz[i]) < 1e-10);
        }
    }

    TEST_CASE("Large batch transform consistency") {
        constexpr std::size_t N = 32;
        SE3Batch<double, N> batch;

        // Create poses with increasing translation
        for (std::size_t i = 0; i < N; ++i) {
            batch.set(i, SE3d::trans(static_cast<double>(i), 0, 0));
        }

        // Transform origin points
        double px[N], py[N], pz[N];
        for (std::size_t i = 0; i < N; ++i) {
            px[i] = py[i] = pz[i] = 0.0;
        }

        batch.transform(px, py, pz);

        // Verify
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(px[i] - static_cast<double>(i)) < 1e-10);
            CHECK(std::abs(py[i]) < 1e-10);
            CHECK(std::abs(pz[i]) < 1e-10);
        }
    }

    TEST_CASE("Batch composition associativity") {
        constexpr std::size_t N = 8;

        SO3Batch<double, N> A, B, C;

        std::mt19937 rng(789);
        for (std::size_t i = 0; i < N; ++i) {
            A.set(i, SO3d::sample_uniform(rng));
            B.set(i, SO3d::sample_uniform(rng));
            C.set(i, SO3d::sample_uniform(rng));
        }

        // (A * B) * C == A * (B * C)
        auto left = (A * B) * C;
        auto right = A * (B * C);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(left[i].is_approx(right[i], 1e-10));
        }
    }
}

// ============================================================================
// RxSO3Batch Tests
// ============================================================================

TEST_SUITE("RxSO3Batch") {

    TEST_CASE("Default construction is identity") {
        RxSO3Batch<double, 4> batch;

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch[i].is_identity(1e-10));
            CHECK(std::abs(batch[i].scale() - 1.0) < 1e-10);
        }
    }

    TEST_CASE("Broadcast construction") {
        auto elem = RxSO3d(2.0, SO3d::rot_x(0.5));
        RxSO3Batch<double, 4> batch(elem);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch[i].is_approx(elem, 1e-10));
        }
    }

    TEST_CASE("Element access and modification") {
        RxSO3Batch<double, 4> batch;

        auto elem0 = RxSO3d(1.5, SO3d::rot_x(0.1));
        auto elem1 = RxSO3d(2.0, SO3d::rot_y(0.2));
        auto elem2 = RxSO3d(0.5, SO3d::rot_z(0.3));
        auto elem3 = RxSO3d(3.0, SO3d::identity());

        batch.set(0, elem0);
        batch.set(1, elem1);
        batch.set(2, elem2);
        batch.set(3, elem3);

        CHECK(batch[0].is_approx(elem0, 1e-10));
        CHECK(batch[1].is_approx(elem1, 1e-10));
        CHECK(batch[2].is_approx(elem2, 1e-10));
        CHECK(batch[3].is_approx(elem3, 1e-10));
    }

    TEST_CASE("Static factory: identity") {
        auto batch = RxSO3Batch<double, 4>::identity();

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(batch[i].is_identity(1e-10));
        }
    }

    TEST_CASE("Static factory: scale_only") {
        auto batch = RxSO3Batch<double, 4>::scale_only(2.5);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(std::abs(batch[i].scale() - 2.5) < 1e-10);
            CHECK(batch[i].so3().is_identity(1e-10));
        }
    }

    TEST_CASE("Exp map from arrays") {
        constexpr std::size_t N = 8;
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        std::uniform_real_distribution<double> scale_dist(-0.5, 0.5);

        double sigma[N], wx[N], wy[N], wz[N];
        for (std::size_t i = 0; i < N; ++i) {
            sigma[i] = scale_dist(rng); // log(scale)
            wx[i] = dist(rng);
            wy[i] = dist(rng);
            wz[i] = dist(rng);
        }

        auto batch = RxSO3Batch<double, N>::exp(sigma, wx, wy, wz);

        // Verify against scalar exp
        for (std::size_t i = 0; i < N; ++i) {
            dp::mat::Vector<double, 4> tangent{{sigma[i], wx[i], wy[i], wz[i]}};
            auto expected = RxSO3d::exp(tangent);
            CHECK(batch[i].is_approx(expected, 1e-5));
        }
    }

    TEST_CASE("Inverse operation") {
        constexpr std::size_t N = 4;
        RxSO3Batch<double, N> batch;

        batch.set(0, RxSO3d(2.0, SO3d::rot_x(0.5)));
        batch.set(1, RxSO3d(0.5, SO3d::rot_y(0.3)));
        batch.set(2, RxSO3d(1.5, SO3d::rot_z(1.0)));
        batch.set(3, RxSO3d::identity());

        auto inv_batch = batch.inverse();

        for (std::size_t i = 0; i < N; ++i) {
            auto product = batch[i] * inv_batch[i];
            CHECK(product.is_identity(1e-10));
        }
    }

    TEST_CASE("Group composition") {
        constexpr std::size_t N = 4;
        RxSO3Batch<double, N> batch1, batch2;

        batch1.set(0, RxSO3d(2.0, SO3d::rot_x(0.1)));
        batch1.set(1, RxSO3d(1.5, SO3d::rot_y(0.2)));
        batch1.set(2, RxSO3d(0.8, SO3d::rot_z(0.3)));
        batch1.set(3, RxSO3d::identity());

        batch2.set(0, RxSO3d(0.5, SO3d::rot_z(0.1)));
        batch2.set(1, RxSO3d(2.0, SO3d::rot_x(0.2)));
        batch2.set(2, RxSO3d(1.0, SO3d::rot_y(0.3)));
        batch2.set(3, RxSO3d(3.0, SO3d::rot_x(0.5)));

        auto result = batch1 * batch2;

        for (std::size_t i = 0; i < N; ++i) {
            auto expected = batch1[i] * batch2[i];
            CHECK(result[i].is_approx(expected, 1e-10));
        }
    }

    TEST_CASE("Transform vectors") {
        constexpr std::size_t N = 4;
        RxSO3Batch<double, N> batch;

        // Scale=2, no rotation
        batch.set(0, RxSO3d(2.0, SO3d::identity()));
        // Scale=1, 90 deg rotation around Z
        batch.set(1, RxSO3d(1.0, SO3d::rot_z(M_PI / 2)));
        // Scale=0.5, 90 deg rotation around X
        batch.set(2, RxSO3d(0.5, SO3d::rot_x(M_PI / 2)));
        // Identity
        batch.set(3, RxSO3d::identity());

        double vx[N] = {1, 1, 0, 5};
        double vy[N] = {0, 0, 1, 6};
        double vz[N] = {0, 0, 0, 7};

        batch.transform(vx, vy, vz);

        // Scale=2, identity: (1,0,0) -> (2,0,0)
        CHECK(std::abs(vx[0] - 2.0) < 1e-10);
        CHECK(std::abs(vy[0]) < 1e-10);
        CHECK(std::abs(vz[0]) < 1e-10);

        // Scale=1, rot_z(90): (1,0,0) -> (0,1,0)
        CHECK(std::abs(vx[1]) < 1e-10);
        CHECK(std::abs(vy[1] - 1.0) < 1e-10);
        CHECK(std::abs(vz[1]) < 1e-10);

        // Scale=0.5, rot_x(90): (0,1,0) -> (0,0,0.5)
        CHECK(std::abs(vx[2]) < 1e-10);
        CHECK(std::abs(vy[2]) < 1e-10);
        CHECK(std::abs(vz[2] - 0.5) < 1e-10);

        // Identity: unchanged
        CHECK(std::abs(vx[3] - 5.0) < 1e-10);
        CHECK(std::abs(vy[3] - 6.0) < 1e-10);
        CHECK(std::abs(vz[3] - 7.0) < 1e-10);
    }

    TEST_CASE("Scale extraction") {
        constexpr std::size_t N = 4;
        RxSO3Batch<double, N> batch;

        batch.set(0, RxSO3d(2.0, SO3d::rot_x(0.1)));
        batch.set(1, RxSO3d(0.5, SO3d::rot_y(0.2)));
        batch.set(2, RxSO3d(1.0, SO3d::rot_z(0.3)));
        batch.set(3, RxSO3d(3.0, SO3d::identity()));

        double scales[N];
        batch.scales(scales);

        CHECK(std::abs(scales[0] - 2.0) < 1e-10);
        CHECK(std::abs(scales[1] - 0.5) < 1e-10);
        CHECK(std::abs(scales[2] - 1.0) < 1e-10);
        CHECK(std::abs(scales[3] - 3.0) < 1e-10);
    }

    TEST_CASE("SO3 extraction") {
        constexpr std::size_t N = 4;
        RxSO3Batch<double, N> batch;

        auto rot0 = SO3d::rot_x(0.1);
        auto rot1 = SO3d::rot_y(0.2);
        auto rot2 = SO3d::rot_z(0.3);
        auto rot3 = SO3d::identity();

        batch.set(0, RxSO3d(2.0, rot0));
        batch.set(1, RxSO3d(0.5, rot1));
        batch.set(2, RxSO3d(1.5, rot2));
        batch.set(3, RxSO3d(3.0, rot3));

        auto so3_batch = batch.so3();

        CHECK(so3_batch[0].is_approx(rot0, 1e-10));
        CHECK(so3_batch[1].is_approx(rot1, 1e-10));
        CHECK(so3_batch[2].is_approx(rot2, 1e-10));
        CHECK(so3_batch[3].is_approx(rot3, 1e-10));
    }

    TEST_CASE("Set scale (uniform)") {
        constexpr std::size_t N = 4;
        RxSO3Batch<double, N> batch;

        auto rot0 = SO3d::rot_x(0.1);
        auto rot1 = SO3d::rot_y(0.2);
        auto rot2 = SO3d::rot_z(0.3);
        auto rot3 = SO3d::identity();

        batch.set(0, RxSO3d(2.0, rot0));
        batch.set(1, RxSO3d(0.5, rot1));
        batch.set(2, RxSO3d(1.5, rot2));
        batch.set(3, RxSO3d(3.0, rot3));

        batch.set_scale(1.0);

        // Rotations should be preserved
        CHECK(batch[0].so3().is_approx(rot0, 1e-10));
        CHECK(batch[1].so3().is_approx(rot1, 1e-10));
        CHECK(batch[2].so3().is_approx(rot2, 1e-10));
        CHECK(batch[3].so3().is_approx(rot3, 1e-10));

        // All scales should be 1.0
        double scales[N];
        batch.scales(scales);
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(scales[i] - 1.0) < 1e-10);
        }
    }

    TEST_CASE("SLERP interpolation") {
        constexpr std::size_t N = 4;
        RxSO3Batch<double, N> batch1(RxSO3d(1.0, SO3d::identity()));
        RxSO3Batch<double, N> batch2(RxSO3d(3.0, SO3d::rot_z(M_PI / 2)));

        auto mid = batch1.slerp(batch2, 0.5);

        for (std::size_t i = 0; i < N; ++i) {
            // Scale should be interpolated: (1 + 3) / 2 = 2
            CHECK(std::abs(mid[i].scale() - 2.0) < 1e-6);

            // Rotation should be approximately rot_z(pi/4)
            auto expected_rot = SO3d::rot_z(M_PI / 4);
            CHECK(mid[i].so3().is_approx(expected_rot, 1e-4));
        }
    }

    TEST_CASE("Log/exp round trip") {
        constexpr std::size_t N = 8;
        std::mt19937 rng(456);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        std::uniform_real_distribution<double> scale_dist(-0.5, 0.5);

        double sigma[N], wx[N], wy[N], wz[N];
        for (std::size_t i = 0; i < N; ++i) {
            sigma[i] = scale_dist(rng);
            wx[i] = dist(rng);
            wy[i] = dist(rng);
            wz[i] = dist(rng);
        }

        auto batch = RxSO3Batch<double, N>::exp(sigma, wx, wy, wz);

        double sigma_out[N], wx_out[N], wy_out[N], wz_out[N];
        batch.log(sigma_out, wx_out, wy_out, wz_out);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::abs(sigma_out[i] - sigma[i]) < 1e-6);
            CHECK(std::abs(wx_out[i] - wx[i]) < 1e-6);
            CHECK(std::abs(wy_out[i] - wy[i]) < 1e-6);
            CHECK(std::abs(wz_out[i] - wz[i]) < 1e-6);
        }
    }

    TEST_CASE("Iterator support") {
        RxSO3Batch<double, 4> batch;
        batch.set(0, RxSO3d(1.0, SO3d::rot_x(0.1)));
        batch.set(1, RxSO3d(2.0, SO3d::rot_y(0.2)));
        batch.set(2, RxSO3d(0.5, SO3d::rot_z(0.3)));
        batch.set(3, RxSO3d::identity());

        std::size_t count = 0;
        for (const auto &elem : batch) {
            (void)elem;
            ++count;
        }
        CHECK(count == 4);
    }

    TEST_CASE("Validation") {
        RxSO3Batch<double, 4> batch;

        // All identity should be valid
        CHECK(batch.all_valid());
        CHECK(batch.all_identity());

        // After modification, not all identity
        batch.set(0, RxSO3d(2.0, SO3d::rot_x(0.1)));
        CHECK(batch.all_valid());
        CHECK_FALSE(batch.all_identity());
    }

    TEST_CASE("Large batch consistency") {
        constexpr std::size_t N = 32;
        RxSO3Batch<double, N> batch;

        std::mt19937 rng(789);
        std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
        std::uniform_real_distribution<double> scale_dist(0.5, 2.0);

        for (std::size_t i = 0; i < N; ++i) {
            auto rot = SO3d::rot_x(angle_dist(rng)) * SO3d::rot_y(angle_dist(rng)) * SO3d::rot_z(angle_dist(rng));
            batch.set(i, RxSO3d(scale_dist(rng), rot));
        }

        // Test inverse consistency
        auto inv_batch = batch.inverse();
        for (std::size_t i = 0; i < N; ++i) {
            auto product = batch[i] * inv_batch[i];
            CHECK(product.is_identity(1e-10));
        }
    }

    TEST_CASE("Composition associativity") {
        constexpr std::size_t N = 8;

        RxSO3Batch<double, N> A, B, C;

        std::mt19937 rng(321);
        std::uniform_real_distribution<double> angle_dist(-0.5, 0.5);
        std::uniform_real_distribution<double> scale_dist(0.5, 2.0);

        for (std::size_t i = 0; i < N; ++i) {
            A.set(i, RxSO3d(scale_dist(rng), SO3d::rot_x(angle_dist(rng))));
            B.set(i, RxSO3d(scale_dist(rng), SO3d::rot_y(angle_dist(rng))));
            C.set(i, RxSO3d(scale_dist(rng), SO3d::rot_z(angle_dist(rng))));
        }

        // (A * B) * C == A * (B * C)
        auto left = (A * B) * C;
        auto right = A * (B * C);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(left[i].is_approx(right[i], 1e-10));
        }
    }

    TEST_CASE("Float type support") {
        RxSO3Batch<float, 8> batch;

        batch.set(0, RxSO3f(2.0f, SO3f::rot_x(0.1f)));
        batch.set(1, RxSO3f(0.5f, SO3f::rot_y(0.2f)));

        auto inv = batch.inverse();
        auto product = batch * inv;

        CHECK(product[0].is_identity(1e-5f));
        CHECK(product[1].is_identity(1e-5f));
    }
}
