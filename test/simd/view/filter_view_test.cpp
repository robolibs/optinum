// =============================================================================
// test/simd/view/filter_view_test.cpp
// Tests for filter_view - non-owning view over masked/filtered elements
// =============================================================================

#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/simd/pack/sse.hpp>
#include <optinum/simd/vector.hpp>
#include <optinum/simd/view/filter_view.hpp>
#include <optinum/simd/view/vector_view.hpp>

namespace on = optinum;

// =============================================================================
// Basic Filter View Tests
// =============================================================================

TEST_CASE("filter_view - Basic construction and access") {
    using namespace on::simd;

    alignas(16) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    bool mask[8] = {true, false, true, false, true, false, true, false};

    SUBCASE("Construction from mask array") {
        filter_view<float, 4> fv(data, mask, 8);

        // Should have 4 true elements
        CHECK(fv.size() == 4);
        CHECK(!fv.empty());

        // Selected elements: indices 0, 2, 4, 6 -> values 1, 3, 5, 7
        CHECK(fv[0] == doctest::Approx(1.0f));
        CHECK(fv[1] == doctest::Approx(3.0f));
        CHECK(fv[2] == doctest::Approx(5.0f));
        CHECK(fv[3] == doctest::Approx(7.0f));

        // Check original indices
        CHECK(fv.index(0) == 0);
        CHECK(fv.index(1) == 2);
        CHECK(fv.index(2) == 4);
        CHECK(fv.index(3) == 6);
    }

    SUBCASE("Empty filter") {
        bool all_false[8] = {false, false, false, false, false, false, false, false};
        filter_view<float, 4> fv(data, all_false, 8);

        CHECK(fv.size() == 0);
        CHECK(fv.empty());
    }

    SUBCASE("Full filter (all true)") {
        bool all_true[8] = {true, true, true, true, true, true, true, true};
        filter_view<float, 4> fv(data, all_true, 8);

        CHECK(fv.size() == 8);

        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(fv[i] == doctest::Approx(static_cast<float>(i + 1)));
        }
    }
}

TEST_CASE("filter_view - Write operations") {
    using namespace on::simd;

    alignas(16) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    bool mask[8] = {true, false, true, false, true, false, true, false};

    SUBCASE("Modify filtered elements") {
        filter_view<float, 4> fv(data, mask, 8);

        fv[0] = 100.0f;
        fv[1] = 200.0f;
        fv[2] = 300.0f;
        fv[3] = 400.0f;

        // Verify changes in original data
        CHECK(data[0] == doctest::Approx(100.0f)); // index 0
        CHECK(data[1] == doctest::Approx(2.0f));   // not selected
        CHECK(data[2] == doctest::Approx(200.0f)); // index 2
        CHECK(data[3] == doctest::Approx(4.0f));   // not selected
        CHECK(data[4] == doctest::Approx(300.0f)); // index 4
        CHECK(data[5] == doctest::Approx(6.0f));   // not selected
        CHECK(data[6] == doctest::Approx(400.0f)); // index 6
        CHECK(data[7] == doctest::Approx(8.0f));   // not selected
    }
}

TEST_CASE("filter_view - Pack operations (gather/scatter)") {
    using namespace on::simd;
    using pack_t = pack<float, 4>;

    alignas(16) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    bool mask[8] = {true, false, true, false, true, false, true, false};

    SUBCASE("Gather pack") {
        filter_view<float, 4> fv(data, mask, 8);

        CHECK(fv.num_packs() == 1);
        CHECK(fv.tail_size() == 4);

        pack_t p = fv.load_pack_gather(0);

        // Should gather elements at indices 0, 2, 4, 6
        CHECK(p[0] == doctest::Approx(1.0f));
        CHECK(p[1] == doctest::Approx(3.0f));
        CHECK(p[2] == doctest::Approx(5.0f));
        CHECK(p[3] == doctest::Approx(7.0f));
    }

    SUBCASE("Scatter pack") {
        filter_view<float, 4> fv(data, mask, 8);

        pack_t p = pack_t::set(10.0f, 20.0f, 30.0f, 40.0f);
        fv.store_pack_scatter(0, p);

        // Should scatter to indices 0, 2, 4, 6
        CHECK(data[0] == doctest::Approx(10.0f));
        CHECK(data[1] == doctest::Approx(2.0f)); // unchanged
        CHECK(data[2] == doctest::Approx(20.0f));
        CHECK(data[3] == doctest::Approx(4.0f)); // unchanged
        CHECK(data[4] == doctest::Approx(30.0f));
        CHECK(data[5] == doctest::Approx(6.0f)); // unchanged
        CHECK(data[6] == doctest::Approx(40.0f));
        CHECK(data[7] == doctest::Approx(8.0f)); // unchanged
    }

    SUBCASE("Tail-safe gather/scatter") {
        bool mask_tail[8] = {true, false, true, false, true, false, false, false};
        filter_view<float, 4> fv(data, mask_tail, 8);

        CHECK(fv.size() == 3);
        CHECK(fv.num_packs() == 1);
        CHECK(fv.tail_size() == 3);

        pack_t p = fv.load_pack_gather_tail(0);
        CHECK(p[0] == doctest::Approx(1.0f));
        CHECK(p[1] == doctest::Approx(3.0f));
        CHECK(p[2] == doctest::Approx(5.0f));
        // p[3] is padding

        pack_t new_p = pack_t::set(99.0f, 88.0f, 77.0f, 66.0f);
        fv.store_pack_scatter_tail(0, new_p);

        CHECK(data[0] == doctest::Approx(99.0f));
        CHECK(data[2] == doctest::Approx(88.0f));
        CHECK(data[4] == doctest::Approx(77.0f));
        // data[6] should be unchanged (tail safe)
        CHECK(data[6] == doctest::Approx(7.0f));
    }
}

TEST_CASE("filter_view - Helper functions") {
    using namespace on::simd;

    alignas(16) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    bool mask[8] = {true, false, true, false, true, false, true, false};

    SUBCASE("filter() helper") {
        auto fv = filter<float, 4>(data, mask, 8);

        CHECK(fv.size() == 4);
        CHECK(fv[0] == doctest::Approx(1.0f));
        CHECK(fv[1] == doctest::Approx(3.0f));
    }

    SUBCASE("filter() from vector_view") {
        vector_view<float, 4> vv(data, 8);
        auto fv = filter(vv, mask);

        CHECK(fv.size() == 4);
        CHECK(fv[0] == doctest::Approx(1.0f));
        CHECK(fv[1] == doctest::Approx(3.0f));
    }
}

TEST_CASE("filter_view - Predicate-based filtering") {
    using namespace on::simd;

    alignas(16) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    SUBCASE("Filter even values") {
        auto fv = filter_if<float, 4>(data, 8, [](float x) { return static_cast<int>(x) % 2 == 0; });

        CHECK(fv.size() == 4);
        // Even values: 2, 4, 6, 8
        CHECK(fv[0] == doctest::Approx(2.0f));
        CHECK(fv[1] == doctest::Approx(4.0f));
        CHECK(fv[2] == doctest::Approx(6.0f));
        CHECK(fv[3] == doctest::Approx(8.0f));
    }

    SUBCASE("Filter values greater than 5") {
        auto fv = filter_if<float, 4>(data, 8, [](float x) { return x > 5.0f; });

        CHECK(fv.size() == 3);
        // Values > 5: 6, 7, 8
        CHECK(fv[0] == doctest::Approx(6.0f));
        CHECK(fv[1] == doctest::Approx(7.0f));
        CHECK(fv[2] == doctest::Approx(8.0f));
    }

    SUBCASE("Filter from vector_view with predicate") {
        vector_view<float, 4> vv(data, 8);
        auto fv = filter_if(vv, [](float x) { return x <= 3.0f; });

        CHECK(fv.size() == 3);
        // Values <= 3: 1, 2, 3
        CHECK(fv[0] == doctest::Approx(1.0f));
        CHECK(fv[1] == doctest::Approx(2.0f));
        CHECK(fv[2] == doctest::Approx(3.0f));
    }
}

TEST_CASE("filter_view - Multiple packs") {
    using namespace on::simd;
    using pack_t = pack<float, 4>;

    alignas(32) float data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    bool mask[16] = {true, true, false, false, true, true,  false, false,
                     true, true, false, false, true, false, false, false};

    SUBCASE("Filter with multiple packs") {
        filter_view<float, 4> fv(data, mask, 16);

        // Selected indices: 0, 1, 4, 5, 8, 9, 12
        CHECK(fv.size() == 7);
        CHECK(fv.num_packs() == 2); // ceil(7 / 4) = 2

        // First pack
        CHECK(fv[0] == doctest::Approx(1.0f));
        CHECK(fv[1] == doctest::Approx(2.0f));
        CHECK(fv[2] == doctest::Approx(5.0f));
        CHECK(fv[3] == doctest::Approx(6.0f));

        // Second pack (partial)
        CHECK(fv[4] == doctest::Approx(9.0f));
        CHECK(fv[5] == doctest::Approx(10.0f));
        CHECK(fv[6] == doctest::Approx(13.0f));

        CHECK(fv.tail_size() == 3);
    }

    SUBCASE("Gather multiple packs") {
        filter_view<float, 4> fv(data, mask, 16);

        pack_t p0 = fv.load_pack_gather(0);
        CHECK(p0[0] == doctest::Approx(1.0f));
        CHECK(p0[1] == doctest::Approx(2.0f));
        CHECK(p0[2] == doctest::Approx(5.0f));
        CHECK(p0[3] == doctest::Approx(6.0f));

        pack_t p1 = fv.load_pack_gather_tail(1);
        CHECK(p1[0] == doctest::Approx(9.0f));
        CHECK(p1[1] == doctest::Approx(10.0f));
        CHECK(p1[2] == doctest::Approx(13.0f));
        // p1[3] is padding
    }
}

TEST_CASE("filter_view - Integration with Vector") {
    using namespace on::simd;

    // Create backing storage and view
    datapod::mat::Vector<float, 8> v_storage;
    on::simd::Vector<float, 8> v(v_storage);
    for (std::size_t i = 0; i < 8; ++i) {
        v[i] = static_cast<float>(i + 1);
    }

    SUBCASE("Filter and modify Vector elements") {
        bool mask[8] = {false, true, false, true, false, true, false, true};
        vector_view<float, 4> vv(v.data(), 8);
        auto fv = filter(vv, mask);

        CHECK(fv.size() == 4);

        // Multiply filtered elements by 10
        for (std::size_t i = 0; i < fv.size(); ++i) {
            fv[i] *= 10.0f;
        }

        // Verify changes
        CHECK(v[0] == doctest::Approx(1.0f));  // not selected
        CHECK(v[1] == doctest::Approx(20.0f)); // selected
        CHECK(v[2] == doctest::Approx(3.0f));  // not selected
        CHECK(v[3] == doctest::Approx(40.0f)); // selected
        CHECK(v[4] == doctest::Approx(5.0f));  // not selected
        CHECK(v[5] == doctest::Approx(60.0f)); // selected
        CHECK(v[6] == doctest::Approx(7.0f));  // not selected
        CHECK(v[7] == doctest::Approx(80.0f)); // selected
    }

    SUBCASE("Predicate-based filtering on Vector") {
        vector_view<float, 4> vv(v.data(), 8);
        auto fv = filter_if(vv, [](float x) { return x > 4.0f; });

        CHECK(fv.size() == 4);
        CHECK(fv[0] == doctest::Approx(5.0f));
        CHECK(fv[1] == doctest::Approx(6.0f));
        CHECK(fv[2] == doctest::Approx(7.0f));
        CHECK(fv[3] == doctest::Approx(8.0f));
    }
}

TEST_CASE("filter_view - const access") {
    using namespace on::simd;

    alignas(16) const float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    bool mask[8] = {true, false, true, false, true, false, true, false};

    SUBCASE("const filter view") {
        filter_view<const float, 4> fv(data, mask, 8);

        CHECK(fv.size() == 4);
        CHECK(fv.at(0) == doctest::Approx(1.0f));
        CHECK(fv.at(1) == doctest::Approx(3.0f));
        CHECK(fv.at(2) == doctest::Approx(5.0f));
        CHECK(fv.at(3) == doctest::Approx(7.0f));

        // Gather should still work
        auto p = fv.load_pack_gather(0);
        CHECK(p[0] == doctest::Approx(1.0f));
    }
}

TEST_CASE("filter_view - Edge cases") {
    using namespace on::simd;

    SUBCASE("Single element filter") {
        alignas(16) float data[1] = {42.0f};
        bool mask[1] = {true};

        filter_view<float, 4> fv(data, mask, 1);
        CHECK(fv.size() == 1);
        CHECK(fv[0] == doctest::Approx(42.0f));
    }

    SUBCASE("Large sparse filter") {
        constexpr std::size_t N = 100;
        alignas(64) float data[N];
        bool mask[N];

        for (std::size_t i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i);
            mask[i] = (i % 10 == 0); // Every 10th element
        }

        filter_view<float, 4> fv(data, mask, N);
        CHECK(fv.size() == 10); // 0, 10, 20, ..., 90

        for (std::size_t i = 0; i < fv.size(); ++i) {
            CHECK(fv[i] == doctest::Approx(static_cast<float>(i * 10)));
        }
    }
}
