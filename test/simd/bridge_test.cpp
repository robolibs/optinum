// =============================================================================
// test/simd/bridge_test.cpp
// Tests for datapod -> SIMD view bridge
// =============================================================================

#include <doctest/doctest.h>

#include <datapod/datapod.hpp>
#include <optinum/simd/bridge.hpp>

namespace on = optinum;

// =============================================================================
// Scalar bridge tests
// =============================================================================

TEST_CASE("bridge - scalar<T> to scalar_view<T,W>") {
    using scalar_t = datapod::mat::Scalar<float>;
    using view_t = on::simd::scalar_view<float, 4>;

    scalar_t s{42.0f};

    SUBCASE("Explicit width") {
        auto v = on::simd::view<4>(s);
        static_assert(std::is_same_v<decltype(v), view_t>);
        CHECK(v.get() == doctest::Approx(42.0f));

        *v.data() = 100.0f;
        CHECK(s.value == doctest::Approx(100.0f));
    }

    SUBCASE("Auto-detect width") {
        auto v = on::simd::view(s);
        CHECK(v.get() == doctest::Approx(42.0f));

        *v.data() = 200.0f;
        CHECK(s.value == doctest::Approx(200.0f));
    }

    SUBCASE("Const view") {
        const scalar_t cs{99.0f};
        auto cv = on::simd::view<4>(cs);
        CHECK(cv.get_const() == doctest::Approx(99.0f));
    }
}

// =============================================================================
// Vector bridge tests
// =============================================================================

TEST_CASE("bridge - vector<T,N> to vector_view<T,W>") {
    using vector_t = datapod::mat::Vector<float, 8>;
    using view_t = on::simd::vector_view<float, 4>;

    alignas(32) vector_t v{{1, 2, 3, 4, 5, 6, 7, 8}};

    SUBCASE("Explicit width") {
        auto vv = on::simd::view<4>(v);
        static_assert(std::is_same_v<decltype(vv), view_t>);
        CHECK(vv.size() == 8);
        CHECK(vv[0] == doctest::Approx(1.0f));
        CHECK(vv[7] == doctest::Approx(8.0f));
    }

    SUBCASE("Auto-detect width") {
        auto vv = on::simd::view(v);
        CHECK(vv.size() == 8);
        CHECK(vv[0] == doctest::Approx(1.0f));
    }

    SUBCASE("Pack operations") {
        auto vv = on::simd::view<4>(v);
        auto p0 = vv.load_pack(0);
        CHECK(p0[0] == doctest::Approx(1.0f));
        CHECK(p0[3] == doctest::Approx(4.0f));

        auto p1 = vv.load_pack(1);
        CHECK(p1[0] == doctest::Approx(5.0f));
        CHECK(p1[3] == doctest::Approx(8.0f));
    }

    SUBCASE("Modifications") {
        auto vv = on::simd::view<4>(v);
        vv[0] = 100.0f;
        CHECK(v[0] == doctest::Approx(100.0f));
    }

    SUBCASE("Const view") {
        const vector_t cv{{10, 20, 30, 40, 50, 60, 70, 80}};
        auto cvv = on::simd::view<4>(cv);
        CHECK(cvv.at(0) == doctest::Approx(10.0f));
        CHECK(cvv.at(7) == doctest::Approx(80.0f));
    }
}

// =============================================================================
// Matrix bridge tests
// =============================================================================

TEST_CASE("bridge - matrix<T,R,C> to matrix_view<T,W>") {
    using matrix_t = datapod::mat::Matrix<float, 4, 3>;
    using view_t = on::simd::matrix_view<float, 4>;

    // 4x3 matrix, column-major storage
    alignas(32) matrix_t m;
    for (std::size_t i = 0; i < 12; ++i) {
        m[i] = static_cast<float>(i + 1);
    }

    SUBCASE("Explicit width") {
        auto mv = on::simd::view<4>(m);
        static_assert(std::is_same_v<decltype(mv), view_t>);
        CHECK(mv.rows() == 4);
        CHECK(mv.cols() == 3);
    }

    SUBCASE("Element access") {
        auto mv = on::simd::view<4>(m);
        CHECK(mv(0, 0) == doctest::Approx(1.0f));
        CHECK(mv(1, 0) == doctest::Approx(2.0f));
        CHECK(mv(0, 1) == doctest::Approx(5.0f));
        CHECK(mv(3, 2) == doctest::Approx(12.0f));
    }

    SUBCASE("Column views") {
        auto mv = on::simd::view<4>(m);
        auto col0 = mv.col(0);
        CHECK(col0.size() == 4);
        CHECK(col0[0] == doctest::Approx(1.0f));
        CHECK(col0[3] == doctest::Approx(4.0f));

        auto col2 = mv.col(2);
        CHECK(col2[0] == doctest::Approx(9.0f));
        CHECK(col2[3] == doctest::Approx(12.0f));
    }

    SUBCASE("Row views") {
        auto mv = on::simd::view<4>(m);
        auto row0 = mv.row(0);
        CHECK(row0.size() == 3);
        CHECK(row0[0] == doctest::Approx(1.0f));
        CHECK(row0[1] == doctest::Approx(5.0f));
        CHECK(row0[2] == doctest::Approx(9.0f));
    }

    SUBCASE("Modifications") {
        auto mv = on::simd::view<4>(m);
        mv(0, 0) = 999.0f;
        CHECK(m(0, 0) == doctest::Approx(999.0f));
    }

    SUBCASE("Const view") {
        matrix_t cm_tmp = m;
        const matrix_t &cm = cm_tmp;
        auto cmv = on::simd::view<4>(cm);
        CHECK(cmv.at(0, 0) == doctest::Approx(1.0f));
        CHECK(cmv.at(3, 2) == doctest::Approx(12.0f));
    }
}

// =============================================================================
// Tensor bridge tests
// =============================================================================

TEST_CASE("bridge - tensor<T,Dims...> to tensor_view<T,W,Rank>") {
    using tensor_t = datapod::mat::Tensor<float, 2, 3, 4>;
    using view_t = on::simd::tensor_view<float, 4, 3>;

    // 2x3x4 tensor, column-major storage
    alignas(32) tensor_t t;
    for (std::size_t i = 0; i < 24; ++i) {
        t[i] = static_cast<float>(i + 1);
    }

    SUBCASE("Explicit width") {
        auto tv = on::simd::view<4>(t);
        static_assert(std::is_same_v<decltype(tv), view_t>);
        CHECK(tv.size() == 24);
        CHECK(tv.extent(0) == 2);
        CHECK(tv.extent(1) == 3);
        CHECK(tv.extent(2) == 4);
    }

    SUBCASE("Auto-detect width") {
        auto tv = on::simd::view(t);
        CHECK(tv.size() == 24);
    }

    SUBCASE("Element access - column-major indexing") {
        auto tv = on::simd::view<4>(t);

        // Column-major: stride[0]=1, stride[1]=2, stride[2]=6
        // t(i,j,k) = data[i + j*2 + k*6]
        CHECK(tv(0, 0, 0) == doctest::Approx(1.0f));  // index 0
        CHECK(tv(1, 0, 0) == doctest::Approx(2.0f));  // index 1
        CHECK(tv(0, 1, 0) == doctest::Approx(3.0f));  // index 2
        CHECK(tv(1, 1, 0) == doctest::Approx(4.0f));  // index 3
        CHECK(tv(0, 2, 0) == doctest::Approx(5.0f));  // index 4
        CHECK(tv(1, 2, 0) == doctest::Approx(6.0f));  // index 5
        CHECK(tv(0, 0, 1) == doctest::Approx(7.0f));  // index 6
        CHECK(tv(1, 0, 1) == doctest::Approx(8.0f));  // index 7
        CHECK(tv(0, 0, 3) == doctest::Approx(19.0f)); // index 18
        CHECK(tv(1, 2, 3) == doctest::Approx(24.0f)); // index 23
    }

    SUBCASE("Pack operations") {
        auto tv = on::simd::view<4>(t);
        auto p0 = tv.load_pack(0);
        CHECK(p0[0] == doctest::Approx(1.0f));
        CHECK(p0[3] == doctest::Approx(4.0f));

        auto p5 = tv.load_pack(5);
        CHECK(p5[0] == doctest::Approx(21.0f));
        CHECK(p5[3] == doctest::Approx(24.0f));
    }

    SUBCASE("Modifications") {
        auto tv = on::simd::view<4>(t);
        tv(0, 0, 0) = 999.0f;
        CHECK(t[0] == doctest::Approx(999.0f));

        tv(1, 2, 3) = 888.0f;
        CHECK(t[23] == doctest::Approx(888.0f));
    }

    SUBCASE("Const view") {
        const tensor_t &ct = t;
        auto ctv = on::simd::view<4>(ct);
        CHECK(ctv.at(0, 0, 0) == doctest::Approx(1.0f));
        CHECK(ctv.at(1, 2, 3) == doctest::Approx(24.0f));
    }
}

// =============================================================================
// Cross-type integration tests
// =============================================================================

TEST_CASE("bridge - Mixed types and operations") {
    using vec_t = datapod::mat::Vector<float, 8>;
    using mat_t = datapod::mat::Matrix<float, 4, 2>;

    alignas(32) vec_t v{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) mat_t m;
    for (std::size_t i = 0; i < 8; ++i) {
        m[i] = static_cast<float>((i + 1) * 10);
    }

    SUBCASE("Vector and matrix views") {
        auto vv = on::simd::view<4>(v);
        auto mv = on::simd::view<4>(m);

        CHECK(vv.size() == 8);
        CHECK(mv.size() == 8);

        // Both should have same linear data layout
        auto vp = vv.load_pack(0);
        auto mp = mv.load_pack(0);

        CHECK(vp[0] == doctest::Approx(1.0f));
        CHECK(mp[0] == doctest::Approx(10.0f));
    }
}

TEST_CASE("bridge - Width auto-detection") {
    using vec_f32 = datapod::mat::Vector<float, 8>;
    using vec_f64 = datapod::mat::Vector<double, 4>;

    alignas(32) vec_f32 vf{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_f64 vd{{1.0, 2.0, 3.0, 4.0}};

    SUBCASE("float auto-width") {
        auto vv = on::simd::view(vf);
#if defined(__AVX__)
        static_assert(decltype(vv)::width == 8, "Expected AVX width=8 for float");
#elif defined(__SSE__)
        static_assert(decltype(vv)::width == 4, "Expected SSE width=4 for float");
#endif
    }

    SUBCASE("double auto-width") {
        auto vv = on::simd::view(vd);
#if defined(__AVX__)
        static_assert(decltype(vv)::width == 4, "Expected AVX width=4 for double");
#elif defined(__SSE2__)
        static_assert(decltype(vv)::width == 2, "Expected SSE2 width=2 for double");
#endif
    }
}
