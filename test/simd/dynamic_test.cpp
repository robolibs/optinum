#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace simd = optinum::simd;

TEST_CASE("Dynamic Vector - Basic operations") {
    SUBCASE("Runtime construction") {
        std::size_t n = 5;
        simd::Vector<double, simd::Dynamic> v(n);

        CHECK(v.size() == 5);
        CHECK_FALSE(v.empty());
    }

    SUBCASE("Construction with value") {
        simd::Vector<double, simd::Dynamic> v(10, 3.14);

        CHECK(v.size() == 10);
        for (std::size_t i = 0; i < 10; ++i) {
            CHECK(v[i] == doctest::Approx(3.14));
        }
    }

    SUBCASE("Resize") {
        simd::Vector<double, simd::Dynamic> v(5);
        v.resize(10);

        CHECK(v.size() == 10);
    }

    SUBCASE("Element access") {
        simd::Vector<double, simd::Dynamic> v(3);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;

        CHECK(v[0] == 1.0);
        CHECK(v[1] == 2.0);
        CHECK(v[2] == 3.0);
    }
}

TEST_CASE("Dynamic Matrix - Basic operations") {
    SUBCASE("Runtime construction") {
        std::size_t r = 3, c = 4;
        simd::Matrix<double, simd::Dynamic, simd::Dynamic> m(r, c);

        CHECK(m.rows() == 3);
        CHECK(m.cols() == 4);
    }

    SUBCASE("Resize") {
        simd::Matrix<double, simd::Dynamic, simd::Dynamic> m(2, 2);
        m.resize(3, 4);

        CHECK(m.rows() == 3);
        CHECK(m.cols() == 4);
    }

    SUBCASE("Element access") {
        simd::Matrix<double, simd::Dynamic, simd::Dynamic> m(2, 2);
        m(0, 0) = 1.0;
        m(0, 1) = 2.0;
        m(1, 0) = 3.0;
        m(1, 1) = 4.0;

        CHECK(m(0, 0) == 1.0);
        CHECK(m(0, 1) == 2.0);
        CHECK(m(1, 0) == 3.0);
        CHECK(m(1, 1) == 4.0);
    }
}

TEST_CASE("Dynamic vs Fixed-Size API compatibility") {
    SUBCASE("Vector API is identical") {
        // Fixed-size
        simd::Vector<double, 5> v_fixed;
        v_fixed[0] = 1.0;
        auto size_fixed = v_fixed.size();

        // Dynamic
        simd::Vector<double, simd::Dynamic> v_dynamic(5);
        v_dynamic[0] = 1.0;
        auto size_dynamic = v_dynamic.size();

        CHECK(size_fixed == size_dynamic);
        CHECK(v_fixed[0] == v_dynamic[0]);
    }

    SUBCASE("Matrix API is identical") {
        // Fixed-size
        simd::Matrix<double, 3, 3> m_fixed;
        m_fixed(0, 0) = 1.0;
        auto rows_fixed = m_fixed.rows();
        auto cols_fixed = m_fixed.cols();

        // Dynamic
        simd::Matrix<double, simd::Dynamic, simd::Dynamic> m_dynamic(3, 3);
        m_dynamic(0, 0) = 1.0;
        auto rows_dynamic = m_dynamic.rows();
        auto cols_dynamic = m_dynamic.cols();

        CHECK(rows_fixed == rows_dynamic);
        CHECK(cols_fixed == cols_dynamic);
        CHECK(m_fixed(0, 0) == m_dynamic(0, 0));
    }
}
