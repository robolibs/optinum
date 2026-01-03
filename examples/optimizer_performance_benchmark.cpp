// =============================================================================
// Optimizer Performance Benchmark
// Measures raw update performance (iterations/second) for each optimizer
// =============================================================================

#include <chrono>
#include <iomanip>
#include <iostream>
#include <optinum/optinum.hpp>

using namespace std::chrono;
using namespace optinum;
namespace dp = datapod;
using namespace optinum::opti;

// Type aliases for convenience
using Momentum = GradientDescent<MomentumUpdate>;
using RMSprop = GradientDescent<RMSPropUpdate>;
using Adam = GradientDescent<AdamUpdate>;

// Benchmark configuration
constexpr std::size_t WARMUP_ITERS = 1000;
constexpr std::size_t BENCH_ITERS = 100000;

// Helper to measure iterations per second
template <typename OptimizerType, std::size_t N> double bench_optimizer(OptimizerType &optimizer, std::size_t iters) {
    Sphere<double, N> sphere;
    dp::mat::Vector<double, N> x;

    // Initialize to non-zero values
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = static_cast<double>(i) - static_cast<double>(N) / 2.0;
    }

    // Configure optimizer
    optimizer.step_size = 0.01;
    optimizer.max_iterations = iters;
    optimizer.tolerance = 0.0; // Disable early stopping

    auto start = high_resolution_clock::now();
    auto result = optimizer.optimize(sphere, x);
    auto end = high_resolution_clock::now();

    double elapsed_sec = duration_cast<duration<double>>(end - start).count();
    return static_cast<double>(result.iterations) / elapsed_sec;
}

// Run benchmark for a specific optimizer and dimension
template <typename OptimizerType, std::size_t... Dims> void run_benchmark(const char *name) {
    std::cout << std::left << std::setw(20) << name;

    (
        [&] {
            OptimizerType optimizer;

            // Warmup
            bench_optimizer<OptimizerType, Dims>(optimizer, WARMUP_ITERS);

            // Actual benchmark
            double iters_per_sec = bench_optimizer<OptimizerType, Dims>(optimizer, BENCH_ITERS);
            double millions_per_sec = iters_per_sec / 1e6;

            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << millions_per_sec << " ";
        }(),
        ...);

    std::cout << "\n";
}

int main() {
    std::cout << "=== Optimizer Performance Benchmark ===\n\n";
    std::cout << "Measuring raw update performance (million iterations/sec)\n";
    std::cout << "Higher is better\n\n";

    std::cout << "Benchmark config:\n";
    std::cout << "  - Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "  - Benchmark iterations: " << BENCH_ITERS << "\n";
    std::cout << "  - Function: Sphere f(x) = Σ xᵢ²\n";
    std::cout << "  - Step size: 0.01 (all optimizers)\n\n";

    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Optimizer";
    std::cout << std::setw(12) << "N=2" << std::setw(12) << "N=4" << std::setw(12) << "N=8" << std::setw(12) << "N=16"
              << std::setw(12) << "N=32" << "\n";
    std::cout << "-------------------------------------------------------------------------\n";

    // Benchmark each optimizer across different dimensions
    run_benchmark<GradientDescent<>, 2, 4, 8, 16, 32>("Vanilla GD");
    run_benchmark<Momentum, 2, 4, 8, 16, 32>("Momentum");
    run_benchmark<RMSprop, 2, 4, 8, 16, 32>("RMSprop");
    run_benchmark<Adam, 2, 4, 8, 16, 32>("Adam");

    std::cout << "-------------------------------------------------------------------------\n\n";

    // Per-update timing for N=16 (typical use case)
    std::cout << "\n=== Per-Update Timing (N=16) ===\n\n";
    std::cout << std::left << std::setw(20) << "Optimizer" << std::setw(20) << "Iters/sec (M)" << std::setw(20)
              << "Time/iter (ns)" << "\n";
    std::cout << "---------------------------------------------------------------\n";

    auto measure_timing = [](const char *name, auto &optimizer) {
        double iters_per_sec = bench_optimizer<decltype(optimizer), 16>(optimizer, BENCH_ITERS);
        double millions_per_sec = iters_per_sec / 1e6;
        double ns_per_iter = 1e9 / iters_per_sec;

        std::cout << std::left << std::setw(20) << name << std::setw(20) << std::fixed << std::setprecision(2)
                  << millions_per_sec << std::setw(20) << std::fixed << std::setprecision(1) << ns_per_iter << "\n";
    };

    GradientDescent<> vanilla;
    bench_optimizer<GradientDescent<>, 16>(vanilla, WARMUP_ITERS); // Warmup
    measure_timing("Vanilla GD", vanilla);

    Momentum momentum;
    bench_optimizer<Momentum, 16>(momentum, WARMUP_ITERS); // Warmup
    measure_timing("Momentum", momentum);

    RMSprop rmsprop;
    bench_optimizer<RMSprop, 16>(rmsprop, WARMUP_ITERS); // Warmup
    measure_timing("RMSprop", rmsprop);

    Adam adam;
    bench_optimizer<Adam, 16>(adam, WARMUP_ITERS); // Warmup
    measure_timing("Adam", adam);

    std::cout << "---------------------------------------------------------------\n\n";

    // SIMD efficiency analysis
    std::cout << "\n=== SIMD Efficiency Analysis ===\n\n";
    std::cout << "Speedup going from N=2 to N=32 (ideal: 16x for AVX, 8x for AVX512):\n\n";

    auto calc_speedup = [](const char *name, auto optimizer_factory) {
        auto optimizer2 = optimizer_factory();
        auto optimizer32 = optimizer_factory();

        bench_optimizer<decltype(optimizer2), 2>(optimizer2, WARMUP_ITERS);
        bench_optimizer<decltype(optimizer32), 32>(optimizer32, WARMUP_ITERS);

        double speed_n2 = bench_optimizer<decltype(optimizer2), 2>(optimizer2, BENCH_ITERS);
        double speed_n32 = bench_optimizer<decltype(optimizer32), 32>(optimizer32, BENCH_ITERS);

        // Throughput speedup (elements/sec)
        double throughput_n2 = speed_n2 * 2;
        double throughput_n32 = speed_n32 * 32;
        double speedup = throughput_n32 / throughput_n2;

        std::cout << std::left << std::setw(20) << name;
        std::cout << "N=2: " << std::setw(8) << std::fixed << std::setprecision(2) << speed_n2 / 1e6 << " M/s   ";
        std::cout << "N=32: " << std::setw(8) << throughput_n32 / throughput_n2 << "x throughput speedup\n";
    };

    calc_speedup("Vanilla GD", []() { return GradientDescent<>(); });
    calc_speedup("Momentum", []() { return Momentum(); });
    calc_speedup("RMSprop", []() { return RMSprop(); });
    calc_speedup("Adam", []() { return Adam(); });

    std::cout << "\nNote: Ideal SIMD speedup depends on vector width:\n";
    std::cout << "  - AVX (256-bit): 4x for double, 8x for float\n";
    std::cout << "  - AVX-512 (512-bit): 8x for double, 16x for float\n";
    std::cout << "  - Actual speedup may be lower due to memory bandwidth limits\n\n";

    return 0;
}
