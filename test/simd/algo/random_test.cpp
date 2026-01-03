// =============================================================================
// test/simd/algo/random_test.cpp
// Tests for SIMD random number generation utilities
// =============================================================================

#include <doctest/doctest.h>

#include <optinum/simd/algo/random.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace on = optinum;

// =============================================================================
// random_uniform tests
// =============================================================================

TEST_CASE("random_uniform - pack<float, 8>") {
    std::mt19937 rng(42);

    auto p = on::simd::random_uniform<float, 8>(rng, 0.0f, 1.0f);

    // All values should be in [0, 1)
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(p[i] >= 0.0f);
        CHECK(p[i] < 1.0f);
    }
}

TEST_CASE("random_uniform - pack<double, 4>") {
    std::mt19937 rng(42);

    auto p = on::simd::random_uniform<double, 4>(rng, -5.0, 5.0);

    // All values should be in [-5, 5)
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(p[i] >= -5.0);
        CHECK(p[i] < 5.0);
    }
}

TEST_CASE("random_uniform - different seeds produce different values") {
    std::mt19937 rng1(42);
    std::mt19937 rng2(123);

    auto p1 = on::simd::random_uniform<float, 8>(rng1, 0.0f, 1.0f);
    auto p2 = on::simd::random_uniform<float, 8>(rng2, 0.0f, 1.0f);

    // At least one value should be different
    bool all_same = true;
    for (std::size_t i = 0; i < 8; ++i) {
        if (std::abs(p1[i] - p2[i]) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    CHECK_FALSE(all_same);
}

TEST_CASE("random_uniform_fill - float array") {
    std::mt19937 rng(42);
    std::vector<float> arr(100);

    on::simd::random_uniform_fill(arr.data(), arr.size(), rng, 0.0f, 1.0f);

    // All values should be in [0, 1)
    for (const auto &v : arr) {
        CHECK(v >= 0.0f);
        CHECK(v < 1.0f);
    }

    // Check that values are not all the same
    float first = arr[0];
    bool all_same = std::all_of(arr.begin(), arr.end(), [first](float v) { return std::abs(v - first) < 1e-6f; });
    CHECK_FALSE(all_same);
}

TEST_CASE("random_uniform_fill - double array with custom range") {
    std::mt19937 rng(42);
    std::vector<double> arr(64);

    on::simd::random_uniform_fill(arr.data(), arr.size(), rng, -10.0, 10.0);

    // All values should be in [-10, 10)
    for (const auto &v : arr) {
        CHECK(v >= -10.0);
        CHECK(v < 10.0);
    }
}

TEST_CASE("random_uniform_fill - non-power-of-2 size") {
    std::mt19937 rng(42);
    std::vector<float> arr(37); // Not divisible by 8

    on::simd::random_uniform_fill(arr.data(), arr.size(), rng, 0.0f, 1.0f);

    // All values should be in [0, 1)
    for (const auto &v : arr) {
        CHECK(v >= 0.0f);
        CHECK(v < 1.0f);
    }
}

// =============================================================================
// random_normal tests
// =============================================================================

TEST_CASE("random_normal_pair - pack<float, 8>") {
    std::mt19937 rng(42);

    auto [z0, z1] = on::simd::random_normal_pair<float, 8>(rng, 0.0f, 1.0f);

    // Values should be finite
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(std::isfinite(z0[i]));
        CHECK(std::isfinite(z1[i]));
    }

    // Most values should be within 3 standard deviations
    int within_3sigma_z0 = 0;
    int within_3sigma_z1 = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        if (std::abs(z0[i]) < 3.0f)
            within_3sigma_z0++;
        if (std::abs(z1[i]) < 3.0f)
            within_3sigma_z1++;
    }
    CHECK(within_3sigma_z0 >= 6); // At least 75% within 3σ
    CHECK(within_3sigma_z1 >= 6);
}

TEST_CASE("random_normal_pair - pack<double, 4>") {
    std::mt19937 rng(42);

    auto [z0, z1] = on::simd::random_normal_pair<double, 4>(rng, 0.0, 1.0);

    // Values should be finite
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(std::isfinite(z0[i]));
        CHECK(std::isfinite(z1[i]));
    }
}

TEST_CASE("random_normal_pair - custom mean and stddev") {
    std::mt19937 rng(42);
    const double mean = 100.0;
    const double stddev = 15.0;

    auto [z0, z1] = on::simd::random_normal_pair<double, 4>(rng, mean, stddev);

    // Values should be centered around mean
    // Most should be within 3σ of mean
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(std::abs(z0[i] - mean) < 5 * stddev); // Within 5σ
        CHECK(std::abs(z1[i] - mean) < 5 * stddev);
    }
}

TEST_CASE("random_normal - single pack") {
    std::mt19937 rng(42);

    auto z = on::simd::random_normal<float, 8>(rng, 0.0f, 1.0f);

    // Values should be finite
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(std::isfinite(z[i]));
    }
}

TEST_CASE("random_normal_fill - statistical properties") {
    std::mt19937 rng(42);
    const std::size_t n = 10000;
    std::vector<float> arr(n);

    on::simd::random_normal_fill(arr.data(), n, rng, 0.0f, 1.0f);

    // Compute sample mean
    double sum = std::accumulate(arr.begin(), arr.end(), 0.0);
    double sample_mean = sum / n;

    // Compute sample variance
    double sq_sum = 0.0;
    for (const auto &v : arr) {
        sq_sum += (v - sample_mean) * (v - sample_mean);
    }
    double sample_var = sq_sum / (n - 1);
    double sample_stddev = std::sqrt(sample_var);

    // Mean should be close to 0 (within 0.1 for n=10000)
    CHECK(std::abs(sample_mean) < 0.1);

    // Stddev should be close to 1 (within 0.1 for n=10000)
    CHECK(std::abs(sample_stddev - 1.0) < 0.1);
}

TEST_CASE("random_normal_fill - custom mean and stddev") {
    std::mt19937 rng(42);
    const std::size_t n = 10000;
    const double mean = 50.0;
    const double stddev = 10.0;
    std::vector<double> arr(n);

    on::simd::random_normal_fill(arr.data(), n, rng, mean, stddev);

    // Compute sample mean
    double sum = std::accumulate(arr.begin(), arr.end(), 0.0);
    double sample_mean = sum / n;

    // Compute sample variance
    double sq_sum = 0.0;
    for (const auto &v : arr) {
        sq_sum += (v - sample_mean) * (v - sample_mean);
    }
    double sample_var = sq_sum / (n - 1);
    double sample_stddev = std::sqrt(sample_var);

    // Mean should be close to 50 (within 1 for n=10000)
    CHECK(std::abs(sample_mean - mean) < 1.0);

    // Stddev should be close to 10 (within 1 for n=10000)
    CHECK(std::abs(sample_stddev - stddev) < 1.0);
}

TEST_CASE("random_normal_fill - non-power-of-2 size") {
    std::mt19937 rng(42);
    std::vector<float> arr(37); // Not divisible by 16 (2*W for float)

    on::simd::random_normal_fill(arr.data(), arr.size(), rng, 0.0f, 1.0f);

    // All values should be finite
    for (const auto &v : arr) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("random_normal_fill - small array") {
    std::mt19937 rng(42);
    std::vector<double> arr(3); // Smaller than 2*W

    on::simd::random_normal_fill(arr.data(), arr.size(), rng, 0.0, 1.0);

    // All values should be finite
    for (const auto &v : arr) {
        CHECK(std::isfinite(v));
    }
}

// =============================================================================
// random_uniform_int tests
// =============================================================================

TEST_CASE("random_uniform_int - basic") {
    std::mt19937 rng(42);
    int arr[8];

    on::simd::random_uniform_int<int, 8>(arr, rng, 0, 100);

    // All values should be in [0, 100]
    for (int i = 0; i < 8; ++i) {
        CHECK(arr[i] >= 0);
        CHECK(arr[i] <= 100);
    }
}

// =============================================================================
// Reproducibility tests
// =============================================================================

TEST_CASE("random_uniform - reproducible with same seed") {
    std::mt19937 rng1(42);
    std::mt19937 rng2(42);

    auto p1 = on::simd::random_uniform<float, 8>(rng1, 0.0f, 1.0f);
    auto p2 = on::simd::random_uniform<float, 8>(rng2, 0.0f, 1.0f);

    // Should produce identical results
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(p1[i] == p2[i]);
    }
}

TEST_CASE("random_normal_pair - reproducible with same seed") {
    std::mt19937 rng1(42);
    std::mt19937 rng2(42);

    auto [z0_1, z1_1] = on::simd::random_normal_pair<double, 4>(rng1, 0.0, 1.0);
    auto [z0_2, z1_2] = on::simd::random_normal_pair<double, 4>(rng2, 0.0, 1.0);

    // Should produce identical results
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(z0_1[i] == z0_2[i]);
        CHECK(z1_1[i] == z1_2[i]);
    }
}
