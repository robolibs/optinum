#pragma once

// =============================================================================
// optinum/simd/timing.hpp
// High-resolution timing utilities for benchmarking
// =============================================================================

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

namespace optinum::simd {

    /**
     * @brief High-resolution timer using std::chrono
     */
    class Timer {
      public:
        using clock = std::chrono::high_resolution_clock;
        using time_point = std::chrono::time_point<clock>;
        using duration = std::chrono::duration<double>;

      private:
        time_point start_;
        bool running_;

      public:
        Timer() noexcept : start_(clock::now()), running_(true) {}

        // Start/restart timer
        void start() noexcept {
            start_ = clock::now();
            running_ = true;
        }

        // Stop timer and return elapsed time in seconds
        [[nodiscard]] double stop() noexcept {
            if (!running_)
                return 0.0;
            running_ = false;
            auto end = clock::now();
            duration elapsed = end - start_;
            return elapsed.count();
        }

        // Get elapsed time without stopping
        [[nodiscard]] double elapsed() const noexcept {
            auto end = clock::now();
            duration elapsed = end - start_;
            return elapsed.count();
        }

        // Get elapsed time in milliseconds
        [[nodiscard]] double elapsed_ms() const noexcept { return elapsed() * 1000.0; }

        // Get elapsed time in microseconds
        [[nodiscard]] double elapsed_us() const noexcept { return elapsed() * 1000000.0; }

        // Get elapsed time in nanoseconds
        [[nodiscard]] double elapsed_ns() const noexcept { return elapsed() * 1000000000.0; }

        // Check if running
        [[nodiscard]] bool is_running() const noexcept { return running_; }
    };

    /**
     * @brief Scoped timer - automatically prints elapsed time on destruction
     */
    class ScopedTimer {
      private:
        Timer timer_;
        std::string label_;
        bool print_on_exit_;

      public:
        explicit ScopedTimer(const std::string &label = "", bool print_on_exit = true) noexcept
            : timer_(), label_(label), print_on_exit_(print_on_exit) {
            if (!label_.empty()) {
                std::cout << "[" << label_ << "] Starting...\n";
            }
        }

        ~ScopedTimer() {
            if (print_on_exit_) {
                double elapsed = timer_.stop();
                if (!label_.empty()) {
                    std::cout << "[" << label_ << "] ";
                }
                std::cout << "Elapsed: " << std::fixed << std::setprecision(6) << elapsed << " seconds\n";
            }
        }

        // Get elapsed time without ending scope
        [[nodiscard]] double elapsed() const noexcept { return timer_.elapsed(); }
        [[nodiscard]] double elapsed_ms() const noexcept { return timer_.elapsed_ms(); }
    };

    /**
     * @brief Benchmark a function by running it multiple times
     *
     * @param func Function to benchmark
     * @param iterations Number of iterations
     * @param warmup Number of warmup iterations (not timed)
     * @return Average time per iteration in seconds
     */
    template <typename Func>
    [[nodiscard]] inline double benchmark(Func &&func, std::size_t iterations = 100, std::size_t warmup = 10) {
        // Warmup
        for (std::size_t i = 0; i < warmup; ++i) {
            func();
        }

        // Timed iterations
        Timer timer;
        for (std::size_t i = 0; i < iterations; ++i) {
            func();
        }
        double total_time = timer.stop();

        return total_time / static_cast<double>(iterations);
    }

    /**
     * @brief Compare two functions and print speedup
     *
     * @param label1 Label for first function
     * @param func1 First function
     * @param label2 Label for second function
     * @param func2 Second function
     * @param iterations Number of iterations per function
     */
    template <typename Func1, typename Func2>
    inline void benchmark_compare(const std::string &label1, Func1 &&func1, const std::string &label2, Func2 &&func2,
                                  std::size_t iterations = 100) {
        std::cout << "\n=== Benchmark Comparison ===\n";

        double time1 = benchmark(std::forward<Func1>(func1), iterations);
        std::cout << label1 << ": " << std::fixed << std::setprecision(6) << time1 * 1000.0 << " ms/iter\n";

        double time2 = benchmark(std::forward<Func2>(func2), iterations);
        std::cout << label2 << ": " << std::fixed << std::setprecision(6) << time2 * 1000.0 << " ms/iter\n";

        double speedup = time1 / time2;
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2) << speedup << "x";
        if (speedup > 1.0) {
            std::cout << " (" << label2 << " is faster)";
        } else if (speedup < 1.0) {
            std::cout << " (" << label1 << " is faster)";
        }
        std::cout << "\n============================\n\n";
    }

    /**
     * @brief Measure throughput (operations per second)
     *
     * @param func Function to measure
     * @param num_operations Number of operations per function call
     * @param iterations Number of iterations
     * @return Operations per second
     */
    template <typename Func>
    [[nodiscard]] inline double throughput(Func &&func, std::size_t num_operations, std::size_t iterations = 100) {
        double avg_time = benchmark(std::forward<Func>(func), iterations);
        return static_cast<double>(num_operations) / avg_time;
    }

    /**
     * @brief Format throughput as human-readable string
     */
    inline std::string format_throughput(double ops_per_sec) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);

        if (ops_per_sec >= 1e9) {
            ss << (ops_per_sec / 1e9) << " GOP/s";
        } else if (ops_per_sec >= 1e6) {
            ss << (ops_per_sec / 1e6) << " MOP/s";
        } else if (ops_per_sec >= 1e3) {
            ss << (ops_per_sec / 1e3) << " KOP/s";
        } else {
            ss << ops_per_sec << " OP/s";
        }

        return ss.str();
    }

    /**
     * @brief Print benchmark statistics
     */
    inline void print_benchmark_header() {
        std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
        std::cout << "║           Benchmark Results                       ║\n";
        std::cout << "╚═══════════════════════════════════════════════════╝\n\n";
    }

    /**
     * @brief Print benchmark result line
     */
    inline void print_benchmark_result(const std::string &name, double time_ms, double speedup = 0.0) {
        std::cout << std::left << std::setw(30) << name << " │ ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_ms << " ms";

        if (speedup > 0.0) {
            std::cout << " │ " << std::setw(8) << std::fixed << std::setprecision(2) << speedup << "x";
        }

        std::cout << "\n";
    }

} // namespace optinum::simd
