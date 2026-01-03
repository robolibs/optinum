#include <doctest/doctest.h>
#include <fstream>
#include <optinum/simd/io.hpp>
#include <optinum/simd/simd.hpp>
#include <sstream>

namespace dp = datapod;
namespace simd = optinum::simd;

TEST_CASE("Print scalar") {
    dp::mat::Scalar<double> storage(3.14159);
    simd::Scalar<double> s(storage);

    // Just make sure it doesn't crash
    std::stringstream ss;
    auto old_buf = std::cout.rdbuf(ss.rdbuf());
    simd::print(s, 2, "pi");
    std::cout.rdbuf(old_buf);

    std::string output = ss.str();
    CHECK(output.find("pi") != std::string::npos);
    CHECK(output.find("3.14") != std::string::npos);
}

TEST_CASE("Print vector") {
    dp::mat::Vector<float, 3> storage;
    simd::Vector<float, 3> v(storage);
    v[0] = 1.0f;
    v[1] = 2.5f;
    v[2] = 3.75f;

    std::stringstream ss;
    auto old_buf = std::cout.rdbuf(ss.rdbuf());
    simd::print(v, 2, "vec");
    std::cout.rdbuf(old_buf);

    std::string output = ss.str();
    CHECK(output.find("vec") != std::string::npos);
    CHECK(output.find("1.00") != std::string::npos);
    CHECK(output.find("2.50") != std::string::npos);
}

TEST_CASE("Print matrix") {
    dp::mat::Matrix<double, 2, 2> storage;
    simd::Matrix<double, 2, 2> m(storage);
    m(0, 0) = 1.0;
    m(1, 0) = 2.0;
    m(0, 1) = 3.0;
    m(1, 1) = 4.0;

    std::stringstream ss;
    auto old_buf = std::cout.rdbuf(ss.rdbuf());
    simd::print(m, 1, "mat");
    std::cout.rdbuf(old_buf);

    std::string output = ss.str();
    CHECK(output.find("mat") != std::string::npos);
    CHECK(output.find("1.0") != std::string::npos);
    CHECK(output.find("4.0") != std::string::npos);
}

TEST_CASE("Print complex") {
    dp::mat::Vector<dp::mat::Complex<double>, 2> storage;
    simd::Complex<double, 2> c(storage);
    c[0] = {3.0, 4.0};
    c[1] = {-1.0, 2.5};

    std::stringstream ss;
    auto old_buf = std::cout.rdbuf(ss.rdbuf());
    simd::print(c, 1, "complex");
    std::cout.rdbuf(old_buf);

    std::string output = ss.str();
    CHECK(output.find("complex") != std::string::npos);
    CHECK(output.find("3.0") != std::string::npos);
    CHECK(output.find("4.0") != std::string::npos);
}

TEST_CASE("Write and read vector") {
    dp::mat::Vector<double, 4> storage;
    simd::Vector<double, 4> v(storage);
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;
    v[3] = 4.0;

    const std::string filename = "/tmp/test_vector.txt";

    // Write
    bool write_ok = simd::write(v, filename, 10);
    CHECK(write_ok);

    // Read
    dp::mat::Vector<double, 4> storage2;
    simd::Vector<double, 4> v2(storage2);
    bool read_ok = simd::read(v2, filename);
    CHECK(read_ok);

    // Verify
    CHECK(v2[0] == doctest::Approx(1.0));
    CHECK(v2[1] == doctest::Approx(2.0));
    CHECK(v2[2] == doctest::Approx(3.0));
    CHECK(v2[3] == doctest::Approx(4.0));
}

TEST_CASE("Write and read matrix") {
    dp::mat::Matrix<float, 2, 3> storage;
    simd::Matrix<float, 2, 3> m(storage);
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(0, 2) = 3.0f;
    m(1, 0) = 4.0f;
    m(1, 1) = 5.0f;
    m(1, 2) = 6.0f;

    const std::string filename = "/tmp/test_matrix.csv";

    // Write
    bool write_ok = simd::write(m, filename, 10);
    CHECK(write_ok);

    // Read
    dp::mat::Matrix<float, 2, 3> storage2;
    simd::Matrix<float, 2, 3> m2(storage2);
    bool read_ok = simd::read(m2, filename);
    CHECK(read_ok);

    // Verify
    CHECK(m2(0, 0) == doctest::Approx(1.0f));
    CHECK(m2(0, 1) == doctest::Approx(2.0f));
    CHECK(m2(1, 2) == doctest::Approx(6.0f));
}

TEST_CASE("Write complex") {
    dp::mat::Vector<dp::mat::Complex<double>, 3> storage;
    simd::Complex<double, 3> c(storage);
    c[0] = {1.0, 2.0};
    c[1] = {3.0, 4.0};
    c[2] = {5.0, 6.0};

    const std::string filename = "/tmp/test_complex.csv";

    // Write
    bool write_ok = simd::write(c, filename, 10);
    CHECK(write_ok);

    // Verify file exists and has content
    std::ifstream file(filename);
    CHECK(file.is_open());

    std::string line;
    std::getline(file, line);
    CHECK(line.find("1.") != std::string::npos);
    CHECK(line.find("2.") != std::string::npos);
}

TEST_CASE("Write scalar") {
    dp::mat::Scalar<double> storage(42.123456789);
    simd::Scalar<double> s(storage);

    const std::string filename = "/tmp/test_scalar.txt";

    // Write
    bool write_ok = simd::write(s, filename, 8);
    CHECK(write_ok);

    // Verify
    std::ifstream file(filename);
    CHECK(file.is_open());

    double value;
    file >> value;
    CHECK(value == doctest::Approx(42.123456789).epsilon(1e-8));
}
