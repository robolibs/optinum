#include <chrono>
#include <doctest/doctest.h>
#include <optinum/simd/timing.hpp>
#include <thread>

using optinum::simd::benchmark;
using optinum::simd::ScopedTimer;
using optinum::simd::throughput;
using optinum::simd::Timer;

TEST_CASE("Timer basic functionality") {
    Timer timer;

    // Sleep for ~10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double elapsed = timer.stop();

    // Should be at least 10ms (0.01 seconds)
    CHECK(elapsed >= 0.009); // Allow some tolerance
    CHECK(elapsed < 0.05);   // But not too much
}

TEST_CASE("Timer elapsed without stopping") {
    Timer timer;

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    double elapsed1 = timer.elapsed();
    CHECK(elapsed1 >= 0.004);

    // Timer should still be running
    CHECK(timer.is_running());

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    double elapsed2 = timer.elapsed();
    CHECK(elapsed2 > elapsed1); // More time has passed
}

TEST_CASE("Timer milliseconds and microseconds") {
    Timer timer;

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double ms = timer.elapsed_ms();
    double us = timer.elapsed_us();

    CHECK(ms >= 9.0);
    CHECK(ms < 50.0);
    CHECK(us >= 9000.0);
    CHECK(us / 1000.0 == doctest::Approx(ms).epsilon(0.01));
}

TEST_CASE("Scoped timer") {
    // Just test that it doesn't crash
    {
        ScopedTimer t("Test scope", false); // Don't print on exit
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        double elapsed = t.elapsed();
        CHECK(elapsed >= 0.004);
    }
    // Destructor should have been called without issues
}

TEST_CASE("Benchmark function") {
    int counter = 0;
    auto func = [&counter]() {
        counter++;
        // Some trivial work
        volatile int x = 0;
        for (int i = 0; i < 100; ++i) {
            x += i;
        }
    };

    // Benchmark with 10 iterations, 2 warmup
    double avg_time = benchmark(func, 10, 2);

    // Should have run 12 times total (2 warmup + 10 timed)
    CHECK(counter == 12);

    // Average time should be small but positive
    CHECK(avg_time > 0.0);
    CHECK(avg_time < 0.01); // Less than 10ms per iteration
}

TEST_CASE("Throughput measurement") {
    const std::size_t num_ops = 1000;
    auto func = []() {
        volatile double x = 0.0;
        for (std::size_t i = 0; i < 1000; ++i) {
            x += static_cast<double>(i);
        }
    };

    double ops_per_sec = throughput(func, num_ops, 10);

    CHECK(ops_per_sec > 0.0);
    // Format should not crash
    std::string formatted = optinum::simd::format_throughput(ops_per_sec);
    CHECK(!formatted.empty());
}

TEST_CASE("Timer restart") {
    Timer timer;

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double elapsed1 = timer.stop();

    // Restart
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double elapsed2 = timer.stop();

    // Second measurement should be less than first
    CHECK(elapsed2 < elapsed1);
    CHECK(elapsed2 >= 0.004);
}

TEST_CASE("Format throughput") {
    CHECK(optinum::simd::format_throughput(1.5e9).find("GOP/s") != std::string::npos);
    CHECK(optinum::simd::format_throughput(2.3e6).find("MOP/s") != std::string::npos);
    CHECK(optinum::simd::format_throughput(4.5e3).find("KOP/s") != std::string::npos);
    CHECK(optinum::simd::format_throughput(123.0).find("OP/s") != std::string::npos);
}
