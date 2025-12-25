// =============================================================================
// examples/boolean_real_world_benchmark.cpp
// Real-world benchmark showing boolean functions in practical applications:
// - Data sanitization (removing NaN/Inf before computation)
// - Safe division with NaN/Inf checking
// - Physics simulation with error detection
// - Statistical outlier detection
// =============================================================================

#include <chrono>
#include <cmath>
#include <datapod/matrix.hpp>
#include <iostream>
#include <limits>
#include <random>

#include <optinum/simd/algo/elementwise.hpp>
#include <optinum/simd/bridge.hpp>
#include <optinum/simd/math/isfinite.hpp>
#include <optinum/simd/math/isinf.hpp>
#include <optinum/simd/math/isnan.hpp>

namespace on = optinum;

volatile double g_sink = 0.0;

void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }
void clobber() { asm volatile("" : : : "memory"); }

// =============================================================================
// SCENARIO 1: Data Sanitization for Machine Learning
// Replace NaN/Inf with zeros before feeding to neural network
// =============================================================================

void benchmark_data_sanitization() {
    constexpr size_t N = 10000;
    constexpr size_t ITERATIONS = 10000;

    alignas(32) datapod::mat::vector<float, N> input;
    alignas(32) datapod::mat::vector<float, N> output;

    // Simulate real ML data: mostly finite, some corrupt values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (size_t i = 0; i < N; ++i) {
        float r = dist(rng);
        if (i % 100 == 0)
            input[i] = NAN; // 1% NaN
        else if (i % 150 == 0)
            input[i] = INFINITY; // 0.67% Inf
        else if (i % 200 == 0)
            input[i] = r / 0.0f; // Division by zero -> Inf
        else
            input[i] = r;
    }

    std::cout << "========================================\n";
    std::cout << "SCENARIO 1: Data Sanitization\n";
    std::cout << "========================================\n";
    std::cout << "Task: Replace NaN/Inf with 0.0 before ML processing\n";
    std::cout << "Array size: " << N << " floats\n";
    std::cout << "Iterations: " << ITERATIONS << "\n\n";

    // SIMD version
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t i = 0; i < N; i += 8) {
            auto pack = on::simd::pack<float, 8>::loadu(&input[i]);
            auto finite_mask = on::simd::isfinite(pack);

            // Replace non-finite values with 0.0
            auto zero = on::simd::pack<float, 8>(0.0f);
            auto result = on::simd::blend(zero, pack, finite_mask);

            result.storeu(&output[i]);
        }
        clobber();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Scalar version
    alignas(32) datapod::mat::vector<float, N> scalar_output;
    start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            scalar_output[i] = std::isfinite(input[i]) ? input[i] : 0.0f;
        }
        clobber();
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "SIMD:   " << simd_time << " ms\n";
    std::cout << "Scalar: " << scalar_time << " ms\n";
    std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n";
    std::cout << "Throughput: " << (N * ITERATIONS / (simd_time * 1e3)) << " M elements/sec (SIMD)\n";
    std::cout << "Throughput: " << (N * ITERATIONS / (scalar_time * 1e3)) << " M elements/sec (Scalar)\n\n";

    g_sink += output[0] + scalar_output[0];
}

// =============================================================================
// SCENARIO 2: Safe Division with Error Detection
// Compute y = a / b, but count and skip divisions that produce NaN/Inf
// =============================================================================

void benchmark_safe_division() {
    constexpr size_t N = 10000;
    constexpr size_t ITERATIONS = 10000;

    alignas(32) datapod::mat::vector<double, N> numerator;
    alignas(32) datapod::mat::vector<double, N> denominator;
    alignas(32) datapod::mat::vector<double, N> result;

    // Simulate real data: some denominators near zero, some zero
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::uniform_real_distribution<double> small_dist(-1e-10, 1e-10);

    for (size_t i = 0; i < N; ++i) {
        numerator[i] = dist(rng);
        if (i % 50 == 0)
            denominator[i] = 0.0; // 2% exact zeros
        else if (i % 100 == 0)
            denominator[i] = small_dist(rng); // 1% near-zero (may cause overflow)
        else
            denominator[i] = dist(rng);
    }

    std::cout << "========================================\n";
    std::cout << "SCENARIO 2: Safe Division\n";
    std::cout << "========================================\n";
    std::cout << "Task: Compute a/b, detect and count NaN/Inf results\n";
    std::cout << "Array size: " << N << " doubles\n";
    std::cout << "Iterations: " << ITERATIONS << "\n\n";

    // SIMD version
    auto start = std::chrono::high_resolution_clock::now();
    int simd_error_count = 0;
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t i = 0; i < N; i += 4) {
            auto a = on::simd::pack<double, 4>::loadu(&numerator[i]);
            auto b = on::simd::pack<double, 4>::loadu(&denominator[i]);
            auto div_result = a / b;

            // Check for NaN or Inf
            auto nan_mask = on::simd::isnan(div_result);
            auto inf_mask = on::simd::isinf(div_result);
            auto error_mask = nan_mask | inf_mask;

            simd_error_count += error_mask.popcount();

            // Replace errors with 0.0
            auto zero = on::simd::pack<double, 4>(0.0);
            auto safe_result = on::simd::blend(div_result, zero, error_mask);

            safe_result.storeu(&result[i]);
        }
        clobber();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Scalar version
    alignas(32) datapod::mat::vector<double, N> scalar_result;
    start = std::chrono::high_resolution_clock::now();
    int scalar_error_count = 0;
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            double div_result = numerator[i] / denominator[i];
            if (std::isnan(div_result) || std::isinf(div_result)) {
                scalar_error_count++;
                scalar_result[i] = 0.0;
            } else {
                scalar_result[i] = div_result;
            }
        }
        clobber();
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "SIMD:   " << simd_time << " ms (errors detected: " << simd_error_count / ITERATIONS << ")\n";
    std::cout << "Scalar: " << scalar_time << " ms (errors detected: " << scalar_error_count / ITERATIONS << ")\n";
    std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n";
    std::cout << "Error detection rate: " << (100.0 * simd_error_count / (N * ITERATIONS)) << "%\n\n";

    g_sink += result[0] + scalar_result[0];
}

// =============================================================================
// SCENARIO 3: Physics Simulation Error Detection
// Simulate particle positions/velocities, detect when particles escape bounds
// (become infinite) or have computational errors (NaN)
// =============================================================================

void benchmark_physics_error_detection() {
    constexpr size_t N_PARTICLES = 10000;
    constexpr size_t ITERATIONS = 1000;

    alignas(32) datapod::mat::vector<float, N_PARTICLES> pos_x;
    alignas(32) datapod::mat::vector<float, N_PARTICLES> pos_y;
    alignas(32) datapod::mat::vector<float, N_PARTICLES> pos_z;
    alignas(32) datapod::mat::vector<float, N_PARTICLES> vel_x;
    alignas(32) datapod::mat::vector<float, N_PARTICLES> vel_y;
    alignas(32) datapod::mat::vector<float, N_PARTICLES> vel_z;

    // Initialize particles
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> vel_dist(-10.0f, 10.0f);

    for (size_t i = 0; i < N_PARTICLES; ++i) {
        pos_x[i] = pos_dist(rng);
        pos_y[i] = pos_dist(rng);
        pos_z[i] = pos_dist(rng);
        vel_x[i] = vel_dist(rng);
        vel_y[i] = vel_dist(rng);
        vel_z[i] = vel_dist(rng);

        // Inject some errors to simulate numerical instability
        if (i % 1000 == 0) {
            pos_x[i] = INFINITY; // Particle escaped
        }
        if (i % 1500 == 0) {
            vel_y[i] = NAN; // Computational error
        }
    }

    std::cout << "========================================\n";
    std::cout << "SCENARIO 3: Physics Simulation Error Detection\n";
    std::cout << "========================================\n";
    std::cout << "Task: Detect escaped particles (Inf) and errors (NaN) in 3D simulation\n";
    std::cout << "Particles: " << N_PARTICLES << "\n";
    std::cout << "Iterations: " << ITERATIONS << "\n\n";

    // SIMD version
    auto start = std::chrono::high_resolution_clock::now();
    int simd_escaped = 0;
    int simd_errors = 0;
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t i = 0; i < N_PARTICLES; i += 8) {
            auto px = on::simd::pack<float, 8>::loadu(&pos_x[i]);
            auto py = on::simd::pack<float, 8>::loadu(&pos_y[i]);
            auto pz = on::simd::pack<float, 8>::loadu(&pos_z[i]);
            auto vx = on::simd::pack<float, 8>::loadu(&vel_x[i]);
            auto vy = on::simd::pack<float, 8>::loadu(&vel_y[i]);
            auto vz = on::simd::pack<float, 8>::loadu(&vel_z[i]);

            // Check for infinities (escaped particles)
            auto inf_mask = on::simd::isinf(px) | on::simd::isinf(py) | on::simd::isinf(pz) | on::simd::isinf(vx) |
                            on::simd::isinf(vy) | on::simd::isinf(vz);

            // Check for NaN (computational errors)
            auto nan_mask = on::simd::isnan(px) | on::simd::isnan(py) | on::simd::isnan(pz) | on::simd::isnan(vx) |
                            on::simd::isnan(vy) | on::simd::isnan(vz);

            simd_escaped += inf_mask.popcount();
            simd_errors += nan_mask.popcount();
        }
        clobber();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Scalar version
    start = std::chrono::high_resolution_clock::now();
    int scalar_escaped = 0;
    int scalar_errors = 0;
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t i = 0; i < N_PARTICLES; ++i) {
            bool has_inf = std::isinf(pos_x[i]) || std::isinf(pos_y[i]) || std::isinf(pos_z[i]) ||
                           std::isinf(vel_x[i]) || std::isinf(vel_y[i]) || std::isinf(vel_z[i]);

            bool has_nan = std::isnan(pos_x[i]) || std::isnan(pos_y[i]) || std::isnan(pos_z[i]) ||
                           std::isnan(vel_x[i]) || std::isnan(vel_y[i]) || std::isnan(vel_z[i]);

            if (has_inf)
                scalar_escaped++;
            if (has_nan)
                scalar_errors++;
        }
        clobber();
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "SIMD:   " << simd_time << " ms\n";
    std::cout << "  Escaped particles: " << simd_escaped / ITERATIONS << " per iteration\n";
    std::cout << "  Computational errors: " << simd_errors / ITERATIONS << " per iteration\n";
    std::cout << "Scalar: " << scalar_time << " ms\n";
    std::cout << "  Escaped particles: " << scalar_escaped / ITERATIONS << " per iteration\n";
    std::cout << "  Computational errors: " << scalar_errors / ITERATIONS << " per iteration\n";
    std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n";
    std::cout << "Checks per second: " << (N_PARTICLES * 6 * ITERATIONS / (simd_time * 1e3)) << " M (SIMD)\n";
    std::cout << "Checks per second: " << (N_PARTICLES * 6 * ITERATIONS / (scalar_time * 1e3)) << " M (Scalar)\n\n";
}

// =============================================================================
// SCENARIO 4: Statistical Outlier Detection and Filtering
// Compute mean/stddev, identify outliers as those > 3 sigma, but also
// detect and exclude NaN/Inf from statistics
// =============================================================================

void benchmark_outlier_detection() {
    constexpr size_t N = 100000;
    constexpr size_t ITERATIONS = 1000;

    alignas(32) datapod::mat::vector<double, N> data;
    alignas(32) datapod::mat::vector<double, N> filtered;

    // Generate data: mostly normal, some outliers, some NaN/Inf
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        if (i % 10000 == 0)
            data[i] = NAN; // 0.01% NaN
        else if (i % 5000 == 0)
            data[i] = INFINITY; // 0.02% Inf
        else if (i % 100 == 0)
            data[i] = normal(rng) * 10.0; // 1% extreme outliers (10 sigma)
        else
            data[i] = normal(rng);
    }

    std::cout << "========================================\n";
    std::cout << "SCENARIO 4: Statistical Outlier Detection\n";
    std::cout << "========================================\n";
    std::cout << "Task: Filter data, excluding NaN/Inf and outliers > 3 sigma\n";
    std::cout << "Array size: " << N << " doubles\n";
    std::cout << "Iterations: " << ITERATIONS << "\n\n";

    // SIMD version
    auto start = std::chrono::high_resolution_clock::now();
    int simd_excluded = 0;
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        // First pass: compute sum and count of finite values only
        double sum = 0.0;
        int count = 0;
        for (size_t i = 0; i < N; i += 4) {
            auto pack = on::simd::pack<double, 4>::loadu(&data[i]);
            auto finite_mask = on::simd::isfinite(pack);

            count += finite_mask.popcount();

            // Sum only finite values (zero out non-finite)
            auto zero = on::simd::pack<double, 4>(0.0);
            auto finite_values = on::simd::blend(zero, pack, finite_mask);

            // Manual horizontal sum
            for (int j = 0; j < 4; ++j) {
                if (finite_mask[j])
                    sum += finite_values[j];
            }
        }

        double mean = count > 0 ? sum / count : 0.0;

        // Second pass: mark outliers (|x - mean| > 3.0)
        for (size_t i = 0; i < N; i += 4) {
            auto pack = on::simd::pack<double, 4>::loadu(&data[i]);
            auto finite_mask = on::simd::isfinite(pack);

            auto mean_pack = on::simd::pack<double, 4>(mean);
            auto diff = on::simd::abs(pack - mean_pack);
            auto threshold = on::simd::pack<double, 4>(3.0);

            // Outlier if |x - mean| > 3.0
            auto outlier_mask = on::simd::cmp_gt(diff, threshold);

            // Exclude if non-finite OR outlier
            auto exclude_mask = !finite_mask | outlier_mask;

            simd_excluded += exclude_mask.popcount();

            // Write filtered data (0.0 for excluded)
            auto zero = on::simd::pack<double, 4>(0.0);
            auto result = on::simd::blend(pack, zero, exclude_mask);
            result.storeu(&filtered[i]);
        }
        clobber();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Scalar version
    alignas(32) datapod::mat::vector<double, N> scalar_filtered;
    start = std::chrono::high_resolution_clock::now();
    int scalar_excluded = 0;
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        // First pass: compute mean of finite values
        double sum = 0.0;
        int count = 0;
        for (size_t i = 0; i < N; ++i) {
            if (std::isfinite(data[i])) {
                sum += data[i];
                count++;
            }
        }
        double mean = count > 0 ? sum / count : 0.0;

        // Second pass: filter
        for (size_t i = 0; i < N; ++i) {
            bool is_finite = std::isfinite(data[i]);
            bool is_outlier = is_finite && std::abs(data[i] - mean) > 3.0;
            bool exclude = !is_finite || is_outlier;

            if (exclude) {
                scalar_excluded++;
                scalar_filtered[i] = 0.0;
            } else {
                scalar_filtered[i] = data[i];
            }
        }
        clobber();
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "SIMD:   " << simd_time << " ms\n";
    std::cout << "  Excluded: " << simd_excluded / ITERATIONS << " per iteration\n";
    std::cout << "Scalar: " << scalar_time << " ms\n";
    std::cout << "  Excluded: " << scalar_excluded / ITERATIONS << " per iteration\n";
    std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n";
    std::cout << "Processing rate: " << (N * ITERATIONS / (simd_time * 1e3)) << " M elements/sec (SIMD)\n";
    std::cout << "Processing rate: " << (N * ITERATIONS / (scalar_time * 1e3)) << " M elements/sec (Scalar)\n\n";

    g_sink += filtered[0] + scalar_filtered[0];
}

int main() {
    std::cout << "=============================================================================\n";
    std::cout << "           SIMD BOOLEAN FUNCTIONS - REAL-WORLD BENCHMARKS\n";
    std::cout << "=============================================================================\n";
    std::cout << "\n";
    std::cout << "These benchmarks demonstrate practical uses of isinf/isnan/isfinite:\n";
    std::cout << "1. Data sanitization (ML preprocessing)\n";
    std::cout << "2. Safe numerical operations with error detection\n";
    std::cout << "3. Physics simulation error monitoring\n";
    std::cout << "4. Statistical outlier detection\n";
    std::cout << "\n";

    benchmark_data_sanitization();
    benchmark_safe_division();
    benchmark_physics_error_detection();
    benchmark_outlier_detection();

    std::cout << "=============================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "=============================================================================\n";
    std::cout << "Boolean functions are most effective when:\n";
    std::cout << "  - Combined with other operations (blend, filtering, conditional compute)\n";
    std::cout << "  - Processing large datasets with mixed clean/corrupt data\n";
    std::cout << "  - Multiple checks per element (6 checks in physics simulation)\n";
    std::cout << "  - Part of a data validation pipeline\n";
    std::cout << "\n";
    std::cout << "Expected speedups: 1.5-4x depending on complexity of surrounding operations\n";

    return 0;
}
