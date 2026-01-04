#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>

namespace simd = optinum::simd;

TEST_CASE("Dynamic Vector - Basic operations") {
    SUBCASE("Runtime construction via owning storage") {
        std::size_t n = 5;
        dp::mat::Vector<double, simd::Dynamic> v_owned(n);
        simd::Vector<double, simd::Dynamic> v(v_owned);

        CHECK(v.size() == 5);
        CHECK_FALSE(v.empty());
    }

    SUBCASE("Construction with value via owning storage") {
        dp::mat::Vector<double, simd::Dynamic> v_owned(10, 3.14);
        simd::Vector<double, simd::Dynamic> v(v_owned);

        CHECK(v.size() == 10);
        for (std::size_t i = 0; i < 10; ++i) {
            CHECK(v[i] == doctest::Approx(3.14));
        }
    }

    SUBCASE("Resize on owning storage") {
        dp::mat::Vector<double, simd::Dynamic> v_owned(5);
        v_owned.resize(10);

        CHECK(v_owned.size() == 10);
    }

    SUBCASE("Element access") {
        dp::mat::Vector<double, simd::Dynamic> v_owned(3);
        simd::Vector<double, simd::Dynamic> v(v_owned);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;

        CHECK(v[0] == 1.0);
        CHECK(v[1] == 2.0);
        CHECK(v[2] == 3.0);
    }
}

TEST_CASE("Dynamic Matrix - Basic operations using owning type") {
    // Note: simd::Matrix is now a non-owning view, so we use dp::mat::matrix for owning storage
    // simd::Matrix can then be created as a view over dp::mat::matrix

    SUBCASE("Owning matrix construction and view creation") {
        std::size_t r = 3, c = 4;
        dp::mat::Matrix<double, simd::Dynamic, simd::Dynamic> m_owned(r, c);

        CHECK(m_owned.rows() == 3);
        CHECK(m_owned.cols() == 4);

        // Create a view over the owned data
        simd::Matrix<double, simd::Dynamic, simd::Dynamic> m_view(m_owned.data(), r, c);
        CHECK(m_view.rows() == 3);
        CHECK(m_view.cols() == 4);
    }

    SUBCASE("Element access through view") {
        dp::mat::Matrix<double, simd::Dynamic, simd::Dynamic> m_owned(2, 2);

        // Create a view and access elements
        simd::Matrix<double, simd::Dynamic, simd::Dynamic> m(m_owned.data(), 2, 2);
        m(0, 0) = 1.0;
        m(0, 1) = 2.0;
        m(1, 0) = 3.0;
        m(1, 1) = 4.0;

        CHECK(m(0, 0) == 1.0);
        CHECK(m(0, 1) == 2.0);
        CHECK(m(1, 0) == 3.0);
        CHECK(m(1, 1) == 4.0);

        // Changes should be visible in the owned matrix too
        CHECK(m_owned(0, 0) == 1.0);
        CHECK(m_owned(0, 1) == 2.0);
    }
}

TEST_CASE("Dynamic vs Fixed-Size API compatibility") {
    SUBCASE("Vector API is identical") {
        // Fixed-size (view over owned data)
        dp::mat::Vector<double, 5> v_fixed_owned{};
        simd::Vector<double, 5> v_fixed(v_fixed_owned);
        v_fixed[0] = 1.0;
        auto size_fixed = v_fixed.size();

        // Dynamic (view over owned data)
        dp::mat::Vector<double, simd::Dynamic> v_dynamic_owned(5);
        simd::Vector<double, simd::Dynamic> v_dynamic(v_dynamic_owned);
        v_dynamic[0] = 1.0;
        auto size_dynamic = v_dynamic.size();

        CHECK(size_fixed == size_dynamic);
        CHECK(v_fixed[0] == v_dynamic[0]);
    }

    SUBCASE("Matrix view API is identical to fixed-size") {
        // Fixed-size (view over owned data)
        dp::mat::Matrix<double, 3, 3> m_fixed_owned;
        simd::Matrix<double, 3, 3> m_fixed(m_fixed_owned);
        m_fixed(0, 0) = 1.0;
        auto rows_fixed = m_fixed.rows();
        auto cols_fixed = m_fixed.cols();

        // Dynamic (view over owned data)
        dp::mat::Matrix<double, simd::Dynamic, simd::Dynamic> m_dynamic_owned(3, 3);
        simd::Matrix<double, simd::Dynamic, simd::Dynamic> m_dynamic(m_dynamic_owned.data(), 3, 3);
        m_dynamic(0, 0) = 1.0;
        auto rows_dynamic = m_dynamic.rows();
        auto cols_dynamic = m_dynamic.cols();

        CHECK(rows_fixed == rows_dynamic);
        CHECK(cols_fixed == cols_dynamic);
        CHECK(m_fixed(0, 0) == m_dynamic(0, 0));
    }
}
