// =============================================================================
// test/simd/pack/mask_test.cpp
// Tests for mask<T,W> - SIMD comparison and masking operations
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/isfinite.hpp>
#include <optinum/simd/math/isinf.hpp>
#include <optinum/simd/math/isnan.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/pack/sse.hpp>

#include <cmath>
#include <limits>

namespace on = optinum;

// =============================================================================
// Scalar Fallback Tests
// =============================================================================

TEST_CASE("mask<float, 2> - Scalar fallback") {
    using mask_t = on::simd::mask<float, 2>;
    using pack_t = on::simd::pack<float, 2>;

    SUBCASE("Factory functions") {
        auto all_t = mask_t::all_true();
        CHECK(all_t.all());
        CHECK(all_t.any());
        CHECK(!all_t.none());
        CHECK(all_t.popcount() == 2);

        auto all_f = mask_t::all_false();
        CHECK(!all_f.all());
        CHECK(!all_f.any());
        CHECK(all_f.none());
        CHECK(all_f.popcount() == 0);

        auto first = mask_t::first_n(1);
        CHECK(first[0] == true);
        CHECK(first[1] == false);
        CHECK(first.popcount() == 1);
    }

    SUBCASE("Boolean operations") {
        auto m1 = mask_t::first_n(1); // [true, false]
        auto m2 = mask_t::all_true(); // [true, true]

        auto and_result = m1 & m2;
        CHECK(and_result[0] == true);
        CHECK(and_result[1] == false);

        auto or_result = m1 | m2;
        CHECK(or_result.all());

        auto not_result = !m1;
        CHECK(not_result[0] == false);
        CHECK(not_result[1] == true);
    }

    SUBCASE("Comparisons") {
        pack_t a(1.0f);
        pack_t b(2.0f);

        auto eq = on::simd::cmp_eq(a, a);
        CHECK(eq.all());

        auto lt = on::simd::cmp_lt(a, b);
        CHECK(lt.all());

        auto gt = on::simd::cmp_gt(a, b);
        CHECK(gt.none());
    }

    SUBCASE("Masked operations") {
        pack_t a(1.0f);
        pack_t b(2.0f);
        auto m = mask_t::first_n(1); // [true, false]

        auto blended = on::simd::blend(a, b, m);
        CHECK(blended[0] == doctest::Approx(2.0f)); // m[0]=true, select b
        CHECK(blended[1] == doctest::Approx(1.0f)); // m[1]=false, select a
    }
}

// =============================================================================
// SSE Tests
// =============================================================================

#ifdef OPTINUM_HAS_SSE2

TEST_CASE("mask<float, 4> - SSE") {
    using mask_t = on::simd::mask<float, 4>;
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Factory functions") {
        auto all_t = mask_t::all_true();
        CHECK(all_t.all());
        CHECK(all_t.any());
        CHECK(!all_t.none());
        CHECK(all_t.popcount() == 4);

        auto all_f = mask_t::all_false();
        CHECK(!all_f.all());
        CHECK(!all_f.any());
        CHECK(all_f.none());
        CHECK(all_f.popcount() == 0);

        auto first2 = mask_t::first_n(2);
        CHECK(first2[0] == true);
        CHECK(first2[1] == true);
        CHECK(first2[2] == false);
        CHECK(first2[3] == false);
        CHECK(first2.popcount() == 2);
    }

    SUBCASE("Boolean operations") {
        auto m1 = mask_t::first_n(2); // [T, T, F, F]
        auto m2 = mask_t::first_n(3); // [T, T, T, F]

        auto and_result = m1 & m2; // [T, T, F, F]
        CHECK(and_result.popcount() == 2);

        auto or_result = m1 | m2; // [T, T, T, F]
        CHECK(or_result.popcount() == 3);

        auto xor_result = m1 ^ m2; // [F, F, T, F]
        CHECK(xor_result[2] == true);
        CHECK(xor_result.popcount() == 1);

        auto not_result = !m1; // [F, F, T, T]
        CHECK(not_result.popcount() == 2);
    }

    SUBCASE("Comparison functions") {
        alignas(16) float data1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        alignas(16) float data2[4] = {1.0f, 3.0f, 3.0f, 2.0f};
        auto p1 = pack_t::load(data1);
        auto p2 = pack_t::load(data2);

        auto eq = on::simd::cmp_eq(p1, p2); // [T, F, T, F]
        CHECK(eq[0] == true);
        CHECK(eq[1] == false);
        CHECK(eq[2] == true);
        CHECK(eq[3] == false);
        CHECK(eq.popcount() == 2);

        auto lt = on::simd::cmp_lt(p1, p2); // [F, T, F, F]
        CHECK(lt[1] == true);
        CHECK(lt.popcount() == 1);

        auto le = on::simd::cmp_le(p1, p2); // [T, T, T, F]
        CHECK(le.popcount() == 3);

        auto gt = on::simd::cmp_gt(p1, p2); // [F, F, F, T]
        CHECK(gt[3] == true);
        CHECK(gt.popcount() == 1);
    }

    SUBCASE("blend operation") {
        pack_t a(1.0f);
        pack_t b(2.0f);
        auto m = mask_t::first_n(2); // [T, T, F, F]

        auto result = on::simd::blend(a, b, m);
        CHECK(result[0] == doctest::Approx(2.0f)); // select b
        CHECK(result[1] == doctest::Approx(2.0f)); // select b
        CHECK(result[2] == doctest::Approx(1.0f)); // select a
        CHECK(result[3] == doctest::Approx(1.0f)); // select a
    }

    SUBCASE("maskload/maskstore") {
        alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto m = mask_t::first_n(2); // load first 2 elements

        auto loaded = on::simd::maskload(data, m);
        CHECK(loaded[0] == doctest::Approx(1.0f));
        CHECK(loaded[1] == doctest::Approx(2.0f));
        CHECK(loaded[2] == doctest::Approx(0.0f)); // masked out = 0
        CHECK(loaded[3] == doctest::Approx(0.0f));

        alignas(16) float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        pack_t values(5.0f);
        on::simd::maskstore(output, values, m);
        CHECK(output[0] == doctest::Approx(5.0f)); // stored
        CHECK(output[1] == doctest::Approx(5.0f)); // stored
        CHECK(output[2] == doctest::Approx(0.0f)); // not stored
        CHECK(output[3] == doctest::Approx(0.0f)); // not stored
    }
}

TEST_CASE("mask<double, 2> - SSE2") {
    using mask_t = on::simd::mask<double, 2>;
    using pack_t = on::simd::pack<double, 2>;

    SUBCASE("Factory functions") {
        auto all_t = mask_t::all_true();
        CHECK(all_t.all());
        CHECK(all_t.popcount() == 2);

        auto first1 = mask_t::first_n(1);
        CHECK(first1[0] == true);
        CHECK(first1[1] == false);
        CHECK(first1.popcount() == 1);
    }

    SUBCASE("Comparisons") {
        alignas(16) double data1[2] = {1.5, 2.5};
        alignas(16) double data2[2] = {1.5, 3.5};
        auto p1 = pack_t::load(data1);
        auto p2 = pack_t::load(data2);

        auto eq = on::simd::cmp_eq(p1, p2);
        CHECK(eq[0] == true);
        CHECK(eq[1] == false);

        auto lt = on::simd::cmp_lt(p1, p2);
        CHECK(lt[0] == false);
        CHECK(lt[1] == true);
    }

    SUBCASE("blend") {
        pack_t a(1.0);
        pack_t b(2.0);
        auto m = mask_t::first_n(1);

        auto result = on::simd::blend(a, b, m);
        CHECK(result[0] == doctest::Approx(2.0));
        CHECK(result[1] == doctest::Approx(1.0));
    }
}

#endif // OPTINUM_HAS_SSE2

// =============================================================================
// AVX Tests
// =============================================================================

#ifdef OPTINUM_HAS_AVX

TEST_CASE("mask<float, 8> - AVX") {
    using mask_t = on::simd::mask<float, 8>;
    using pack_t = on::simd::pack<float, 8>;

    SUBCASE("Factory functions") {
        auto all_t = mask_t::all_true();
        CHECK(all_t.all());
        CHECK(all_t.popcount() == 8);

        auto first4 = mask_t::first_n(4);
        for (int i = 0; i < 4; ++i)
            CHECK(first4[i] == true);
        for (int i = 4; i < 8; ++i)
            CHECK(first4[i] == false);
        CHECK(first4.popcount() == 4);
    }

    SUBCASE("Boolean operations") {
        auto m1 = mask_t::first_n(4); // [T,T,T,T,F,F,F,F]
        auto m2 = mask_t::first_n(6); // [T,T,T,T,T,T,F,F]

        auto and_result = m1 & m2;
        CHECK(and_result.popcount() == 4);

        auto or_result = m1 | m2;
        CHECK(or_result.popcount() == 6);

        auto not_result = !m1;
        CHECK(not_result.popcount() == 4);
    }

    SUBCASE("Comparisons") {
        alignas(32) float data1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        alignas(32) float data2[8] = {1, 3, 3, 5, 5, 7, 7, 9};
        auto p1 = pack_t::load(data1);
        auto p2 = pack_t::load(data2);

        auto eq = on::simd::cmp_eq(p1, p2);
        CHECK(eq[0] == true);
        CHECK(eq[2] == true);
        CHECK(eq[4] == true);
        CHECK(eq[6] == true);
        CHECK(eq.popcount() == 4); // indices 0, 2, 4, 6

        auto lt = on::simd::cmp_lt(p1, p2);
        CHECK(lt[1] == true); // 2 < 3
        CHECK(lt[3] == true); // 4 < 5
        CHECK(lt[5] == true); // 6 < 7
        CHECK(lt[7] == true); // 8 < 9
    }

    SUBCASE("blend") {
        pack_t a(1.0f);
        pack_t b(2.0f);
        auto m = mask_t::first_n(4);

        auto result = on::simd::blend(a, b, m);
        for (int i = 0; i < 4; ++i)
            CHECK(result[i] == doctest::Approx(2.0f));
        for (int i = 4; i < 8; ++i)
            CHECK(result[i] == doctest::Approx(1.0f));
    }

    SUBCASE("maskload/maskstore") {
        alignas(32) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        auto m = mask_t::first_n(5);

        auto loaded = on::simd::maskload(data, m);
        for (int i = 0; i < 5; ++i)
            CHECK(loaded[i] == doctest::Approx(static_cast<float>(i + 1)));
        for (int i = 5; i < 8; ++i)
            CHECK(loaded[i] == doctest::Approx(0.0f));

        alignas(32) float output[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        pack_t values(9.0f);
        on::simd::maskstore(output, values, m);
        for (int i = 0; i < 5; ++i)
            CHECK(output[i] == doctest::Approx(9.0f));
        for (int i = 5; i < 8; ++i)
            CHECK(output[i] == doctest::Approx(0.0f));
    }
}

TEST_CASE("mask<double, 4> - AVX") {
    using mask_t = on::simd::mask<double, 4>;
    using pack_t = on::simd::pack<double, 4>;

    SUBCASE("Factory functions") {
        auto all_t = mask_t::all_true();
        CHECK(all_t.all());
        CHECK(all_t.popcount() == 4);

        auto first2 = mask_t::first_n(2);
        CHECK(first2.popcount() == 2);
    }

    SUBCASE("Comparisons") {
        alignas(32) double data1[4] = {1.0, 2.0, 3.0, 4.0};
        alignas(32) double data2[4] = {1.0, 3.0, 3.0, 2.0};
        auto p1 = pack_t::load(data1);
        auto p2 = pack_t::load(data2);

        auto eq = on::simd::cmp_eq(p1, p2);
        CHECK(eq.popcount() == 2); // indices 0, 2

        auto lt = on::simd::cmp_lt(p1, p2);
        CHECK(lt[1] == true); // 2 < 3
        CHECK(lt.popcount() == 1);

        auto gt = on::simd::cmp_gt(p1, p2);
        CHECK(gt[3] == true); // 4 > 2
        CHECK(gt.popcount() == 1);
    }

    SUBCASE("blend") {
        pack_t a(1.0);
        pack_t b(2.0);
        auto m = mask_t::first_n(2);

        auto result = on::simd::blend(a, b, m);
        CHECK(result[0] == doctest::Approx(2.0));
        CHECK(result[1] == doctest::Approx(2.0));
        CHECK(result[2] == doctest::Approx(1.0));
        CHECK(result[3] == doctest::Approx(1.0));
    }

    SUBCASE("maskload/maskstore") {
        alignas(32) double data[4] = {1.0, 2.0, 3.0, 4.0};
        auto m = mask_t::first_n(3);

        auto loaded = on::simd::maskload(data, m);
        for (int i = 0; i < 3; ++i)
            CHECK(loaded[i] == doctest::Approx(static_cast<double>(i + 1)));
        CHECK(loaded[3] == doctest::Approx(0.0));

        alignas(32) double output[4] = {0, 0, 0, 0};
        pack_t values(7.0);
        on::simd::maskstore(output, values, m);
        for (int i = 0; i < 3; ++i)
            CHECK(output[i] == doctest::Approx(7.0));
        CHECK(output[3] == doctest::Approx(0.0));
    }
}

#endif // OPTINUM_HAS_AVX

// =============================================================================
// Boolean Test Functions (isinf, isnan, isfinite)
// =============================================================================

TEST_CASE("Boolean functions - isinf, isnan, isfinite") {
    using pack_f4 = on::simd::pack<float, 4>;
    using mask_f4 = on::simd::mask<float, 4>;

    SUBCASE("isinf - SSE float") {
        // Test positive infinity
        alignas(16) float data_inf[4] = {INFINITY, 1.0f, -INFINITY, 2.0f};
        auto p_inf = pack_f4::load(data_inf);
        auto m_inf = on::simd::isinf(p_inf);

        CHECK(m_inf[0] == true);  // +inf
        CHECK(m_inf[1] == false); // normal
        CHECK(m_inf[2] == true);  // -inf
        CHECK(m_inf[3] == false); // normal
        CHECK(m_inf.popcount() == 2);

        // Test all normal values
        pack_f4 normal(42.0f);
        auto m_normal = on::simd::isinf(normal);
        CHECK(m_normal.none());

        // Test all infinities
        pack_f4 all_inf(INFINITY);
        auto m_all_inf = on::simd::isinf(all_inf);
        CHECK(m_all_inf.all());
    }

    SUBCASE("isnan - SSE float") {
        // Test NaN values
        alignas(16) float data_nan[4] = {NAN, 1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f};
        auto p_nan = pack_f4::load(data_nan);
        auto m_nan = on::simd::isnan(p_nan);

        CHECK(m_nan[0] == true);  // NaN
        CHECK(m_nan[1] == false); // normal
        CHECK(m_nan[2] == true);  // NaN
        CHECK(m_nan[3] == false); // normal
        CHECK(m_nan.popcount() == 2);

        // Test all normal values
        pack_f4 normal(3.14f);
        auto m_normal = on::simd::isnan(normal);
        CHECK(m_normal.none());

        // Test infinity is not NaN
        pack_f4 inf(INFINITY);
        auto m_inf = on::simd::isnan(inf);
        CHECK(m_inf.none());
    }

    SUBCASE("isfinite - SSE float") {
        // Test mixed values
        alignas(16) float data_mixed[4] = {1.0f, INFINITY, NAN, -42.0f};
        auto p_mixed = pack_f4::load(data_mixed);
        auto m_finite = on::simd::isfinite(p_mixed);

        CHECK(m_finite[0] == true);  // normal
        CHECK(m_finite[1] == false); // infinity
        CHECK(m_finite[2] == false); // NaN
        CHECK(m_finite[3] == true);  // normal
        CHECK(m_finite.popcount() == 2);

        // Test all finite values
        pack_f4 all_finite(1.23f);
        auto m_all_finite = on::simd::isfinite(all_finite);
        CHECK(m_all_finite.all());

        // Test all infinities
        pack_f4 all_inf(INFINITY);
        auto m_all_inf = on::simd::isfinite(all_inf);
        CHECK(m_all_inf.none());
    }

#ifdef OPTINUM_HAS_SSE2
    SUBCASE("isinf - SSE double") {
        using pack_d2 = on::simd::pack<double, 2>;
        using mask_d2 = on::simd::mask<double, 2>;

        alignas(16) double data_inf[2] = {INFINITY, 1.5};
        auto p_inf = pack_d2::load(data_inf);
        auto m_inf = on::simd::isinf(p_inf);

        CHECK(m_inf[0] == true);
        CHECK(m_inf[1] == false);
        CHECK(m_inf.popcount() == 1);

        // Test negative infinity
        pack_d2 neg_inf(-INFINITY);
        auto m_neg_inf = on::simd::isinf(neg_inf);
        CHECK(m_neg_inf.all());
    }

    SUBCASE("isnan - SSE double") {
        using pack_d2 = on::simd::pack<double, 2>;
        using mask_d2 = on::simd::mask<double, 2>;

        alignas(16) double data_nan[2] = {std::numeric_limits<double>::quiet_NaN(), 2.5};
        auto p_nan = pack_d2::load(data_nan);
        auto m_nan = on::simd::isnan(p_nan);

        CHECK(m_nan[0] == true);
        CHECK(m_nan[1] == false);
        CHECK(m_nan.popcount() == 1);
    }

    SUBCASE("isfinite - SSE double") {
        using pack_d2 = on::simd::pack<double, 2>;
        using mask_d2 = on::simd::mask<double, 2>;

        alignas(16) double data_mixed[2] = {INFINITY, 3.14};
        auto p_mixed = pack_d2::load(data_mixed);
        auto m_finite = on::simd::isfinite(p_mixed);

        CHECK(m_finite[0] == false);
        CHECK(m_finite[1] == true);
        CHECK(m_finite.popcount() == 1);
    }
#endif // OPTINUM_HAS_SSE2

#ifdef OPTINUM_HAS_AVX
    SUBCASE("isinf - AVX float") {
        using pack_f8 = on::simd::pack<float, 8>;
        using mask_f8 = on::simd::mask<float, 8>;

        alignas(32) float data_inf[8] = {INFINITY, 1.0f, -INFINITY, 2.0f, 3.0f, INFINITY, 4.0f, -INFINITY};
        auto p_inf = pack_f8::load(data_inf);
        auto m_inf = on::simd::isinf(p_inf);

        CHECK(m_inf[0] == true);
        CHECK(m_inf[1] == false);
        CHECK(m_inf[2] == true);
        CHECK(m_inf[3] == false);
        CHECK(m_inf[4] == false);
        CHECK(m_inf[5] == true);
        CHECK(m_inf[6] == false);
        CHECK(m_inf[7] == true);
        CHECK(m_inf.popcount() == 4);
    }

    SUBCASE("isnan - AVX float") {
        using pack_f8 = on::simd::pack<float, 8>;
        using mask_f8 = on::simd::mask<float, 8>;

        alignas(32) float data_nan[8] = {NAN, 1.0f, 2.0f, NAN, 3.0f, 4.0f, NAN, 5.0f};
        auto p_nan = pack_f8::load(data_nan);
        auto m_nan = on::simd::isnan(p_nan);

        CHECK(m_nan[0] == true);
        CHECK(m_nan[3] == true);
        CHECK(m_nan[6] == true);
        CHECK(m_nan.popcount() == 3);
    }

    SUBCASE("isfinite - AVX float") {
        using pack_f8 = on::simd::pack<float, 8>;
        using mask_f8 = on::simd::mask<float, 8>;

        alignas(32) float data_mixed[8] = {1.0f, INFINITY, NAN, 2.0f, 3.0f, -INFINITY, 4.0f, NAN};
        auto p_mixed = pack_f8::load(data_mixed);
        auto m_finite = on::simd::isfinite(p_mixed);

        CHECK(m_finite[0] == true);
        CHECK(m_finite[1] == false);
        CHECK(m_finite[2] == false);
        CHECK(m_finite[3] == true);
        CHECK(m_finite[4] == true);
        CHECK(m_finite[5] == false);
        CHECK(m_finite[6] == true);
        CHECK(m_finite[7] == false);
        CHECK(m_finite.popcount() == 4);
    }

    SUBCASE("isinf - AVX double") {
        using pack_d4 = on::simd::pack<double, 4>;
        using mask_d4 = on::simd::mask<double, 4>;

        alignas(32) double data_inf[4] = {INFINITY, 1.5, -INFINITY, 2.5};
        auto p_inf = pack_d4::load(data_inf);
        auto m_inf = on::simd::isinf(p_inf);

        CHECK(m_inf[0] == true);
        CHECK(m_inf[1] == false);
        CHECK(m_inf[2] == true);
        CHECK(m_inf[3] == false);
        CHECK(m_inf.popcount() == 2);
    }

    SUBCASE("isnan - AVX double") {
        using pack_d4 = on::simd::pack<double, 4>;
        using mask_d4 = on::simd::mask<double, 4>;

        alignas(32) double data_nan[4] = {std::numeric_limits<double>::quiet_NaN(), 1.5, 2.5, NAN};
        auto p_nan = pack_d4::load(data_nan);
        auto m_nan = on::simd::isnan(p_nan);

        CHECK(m_nan[0] == true);
        CHECK(m_nan[1] == false);
        CHECK(m_nan[2] == false);
        CHECK(m_nan[3] == true);
        CHECK(m_nan.popcount() == 2);
    }

    SUBCASE("isfinite - AVX double") {
        using pack_d4 = on::simd::pack<double, 4>;
        using mask_d4 = on::simd::mask<double, 4>;

        alignas(32) double data_mixed[4] = {1.5, INFINITY, NAN, 2.5};
        auto p_mixed = pack_d4::load(data_mixed);
        auto m_finite = on::simd::isfinite(p_mixed);

        CHECK(m_finite[0] == true);
        CHECK(m_finite[1] == false);
        CHECK(m_finite[2] == false);
        CHECK(m_finite[3] == true);
        CHECK(m_finite.popcount() == 2);
    }
#endif // OPTINUM_HAS_AVX
}
