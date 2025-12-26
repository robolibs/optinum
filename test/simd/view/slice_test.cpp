// =============================================================================
// test/simd/view/slice_test.cpp
// Tests for view slicing: seq(), fseq<>(), all, fix<N>
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/view/slice.hpp>
#include <optinum/simd/view/vector_view.hpp>
#include <vector>

using namespace optinum::simd;

TEST_CASE("Slicing types - seq") {
    SUBCASE("seq(stop)") {
        seq s(10);
        CHECK(s.start == 0);
        CHECK(s.stop == 10);
        CHECK(s.step == 1);
        CHECK(s.size() == 10);
        CHECK(s[0] == 0);
        CHECK(s[5] == 5);
        CHECK(s[9] == 9);
    }

    SUBCASE("seq(start, stop)") {
        seq s(5, 15);
        CHECK(s.start == 5);
        CHECK(s.stop == 15);
        CHECK(s.step == 1);
        CHECK(s.size() == 10);
        CHECK(s[0] == 5);
        CHECK(s[5] == 10);
        CHECK(s[9] == 14);
    }

    SUBCASE("seq(start, stop, step)") {
        seq s(0, 10, 2);
        CHECK(s.start == 0);
        CHECK(s.stop == 10);
        CHECK(s.step == 2);
        CHECK(s.size() == 5);
        CHECK(s[0] == 0);
        CHECK(s[1] == 2);
        CHECK(s[2] == 4);
        CHECK(s[3] == 6);
        CHECK(s[4] == 8);
    }

    SUBCASE("seq with step=3") {
        seq s(1, 10, 3);
        CHECK(s.size() == 3);
        CHECK(s[0] == 1);
        CHECK(s[1] == 4);
        CHECK(s[2] == 7);
    }

    SUBCASE("Empty seq") {
        seq s(10, 5);
        CHECK(s.size() == 0);
    }
}

TEST_CASE("Slicing types - fseq") {
    SUBCASE("fseq<0, 10>") {
        fseq<0, 10> s;
        CHECK(s.start == 0);
        CHECK(s.stop == 10);
        CHECK(s.step == 1);
        CHECK(s.size() == 10);
        CHECK(s[0] == 0);
        CHECK(s[9] == 9);
    }

    SUBCASE("fseq<5, 15>") {
        fseq<5, 15> s;
        CHECK(s.size() == 10);
        CHECK(s[0] == 5);
        CHECK(s[5] == 10);
    }

    SUBCASE("fseq<0, 10, 2> with step") {
        fseq<0, 10, 2> s;
        CHECK(s.size() == 5);
        CHECK(s[0] == 0);
        CHECK(s[1] == 2);
        CHECK(s[4] == 8);
    }
}

TEST_CASE("Slicing types - all_t") {
    SUBCASE("all binds to dimension") {
        all_t a;
        seq s = a.bind(20);
        CHECK(s.start == 0);
        CHECK(s.stop == 20);
        CHECK(s.step == 1);
        CHECK(s.size() == 20);
    }
}

TEST_CASE("Slicing types - fix<N>") {
    SUBCASE("fix<5>") {
        fix<5> f;
        CHECK(f() == 5);
        CHECK(f.index == 5);
    }
}

TEST_CASE("resolve_slice") {
    SUBCASE("resolve seq") {
        seq s(5, 15);
        seq resolved = resolve_slice(s, 100);
        CHECK(resolved.start == 5);
        CHECK(resolved.stop == 15);
    }

    SUBCASE("resolve fseq") {
        fseq<10, 20> f;
        seq resolved = resolve_slice(f, 100);
        CHECK(resolved.start == 10);
        CHECK(resolved.stop == 20);
        CHECK(resolved.step == 1);
    }

    SUBCASE("resolve all") {
        seq resolved = resolve_slice(all, 50);
        CHECK(resolved.start == 0);
        CHECK(resolved.stop == 50);
        CHECK(resolved.step == 1);
    }

    SUBCASE("resolve fix<N>") {
        fix<7> f;
        seq resolved = resolve_slice(f, 100);
        CHECK(resolved.start == 7);
        CHECK(resolved.stop == 8);
        CHECK(resolved.size() == 1);
    }
}

TEST_CASE("vector_view slicing") {
    // Create test data
    std::vector<float> data(20);
    for (int i = 0; i < 20; i++) {
        data[i] = static_cast<float>(i);
    }

    vector_view<float, 4> v(data.data(), 20);

    SUBCASE("slice with seq()") {
        auto s = v.slice(seq(5, 15));
        CHECK(s.size() == 10);
        CHECK(s[0] == 5.0f);
        CHECK(s[5] == 10.0f);
        CHECK(s[9] == 14.0f);
    }

    SUBCASE("slice with seq(stop)") {
        auto s = v.slice(seq(10));
        CHECK(s.size() == 10);
        CHECK(s[0] == 0.0f);
        CHECK(s[9] == 9.0f);
    }

    SUBCASE("slice with all") {
        auto s = v.slice(all);
        CHECK(s.size() == 20);
        CHECK(s[0] == 0.0f);
        CHECK(s[19] == 19.0f);
    }

    SUBCASE("slice with fseq<>") {
        auto s = v.slice(fseq<3, 13>());
        CHECK(s.size() == 10);
        CHECK(s[0] == 3.0f);
        CHECK(s[9] == 12.0f);
    }

    SUBCASE("slice with fix<N>") {
        auto s = v.slice(fix<7>());
        CHECK(s.size() == 1);
        CHECK(s[0] == 7.0f);
    }

    SUBCASE("chained slicing") {
        auto s1 = v.slice(seq(5, 15)); // 5..14
        auto s2 = s1.slice(seq(2, 8)); // 7..12 (relative to s1)
        CHECK(s2.size() == 6);
        CHECK(s2[0] == 7.0f);
        CHECK(s2[5] == 12.0f);
    }
}

TEST_CASE("Type traits") {
    SUBCASE("is_slice_v") {
        CHECK(is_slice_v<seq> == true);
        CHECK(is_slice_v<fseq<0, 10>> == true);
        CHECK(is_slice_v<all_t> == true);
        CHECK(is_slice_v<int> == false);
        CHECK(is_slice_v<fix<5>> == false);
    }

    SUBCASE("is_fixed_index_v") {
        CHECK(is_fixed_index_v<fix<0>> == true);
        CHECK(is_fixed_index_v<fix<10>> == true);
        CHECK(is_fixed_index_v<seq> == false);
        CHECK(is_fixed_index_v<int> == false);
    }

    SUBCASE("is_static_slice_v") {
        CHECK(is_static_slice_v<fseq<0, 10>> == true);
        CHECK(is_static_slice_v<fix<5>> == true);
        CHECK(is_static_slice_v<seq> == false);
        CHECK(is_static_slice_v<all_t> == false);
    }
}
