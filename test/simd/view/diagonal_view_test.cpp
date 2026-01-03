// =============================================================================
// test/simd/view/diagonal_view_test.cpp
// Tests for diagonal_view - non-owning view over matrix diagonals
// =============================================================================

#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/pack/sse.hpp>
#include <optinum/simd/view/diagonal_view.hpp>
#include <optinum/simd/view/matrix_view.hpp>

namespace on = optinum;

// =============================================================================
// Basic Diagonal View Tests
// =============================================================================

TEST_CASE("diagonal_view - Main diagonal") {
    using namespace on::simd;

    // 4x4 matrix, column-major
    alignas(16) float data[16] = {1,  2,  3,  4,   // col 0
                                  5,  6,  7,  8,   // col 1
                                  9,  10, 11, 12,  // col 2
                                  13, 14, 15, 16}; // col 3

    SUBCASE("Main diagonal (k=0)") {
        diagonal_view<float, 4> diag(data, 4, 4, 0);

        CHECK(diag.size() == 4);
        CHECK(diag.diagonal_offset() == 0);
        CHECK(diag.is_contiguous() == false);

        // Main diagonal: (0,0)=1, (1,1)=6, (2,2)=11, (3,3)=16
        CHECK(diag[0] == doctest::Approx(1.0f));
        CHECK(diag[1] == doctest::Approx(6.0f));
        CHECK(diag[2] == doctest::Approx(11.0f));
        CHECK(diag[3] == doctest::Approx(16.0f));
    }

    SUBCASE("Main diagonal - default constructor") {
        diagonal_view<float, 4> diag(data, 4, 4);

        CHECK(diag.size() == 4);
        CHECK(diag.diagonal_offset() == 0);

        CHECK(diag[0] == doctest::Approx(1.0f));
        CHECK(diag[1] == doctest::Approx(6.0f));
        CHECK(diag[2] == doctest::Approx(11.0f));
        CHECK(diag[3] == doctest::Approx(16.0f));
    }
}

TEST_CASE("diagonal_view - Upper diagonals") {
    using namespace on::simd;

    // 4x4 matrix
    alignas(16) float data[16] = {1,  2,  3,  4,   // col 0
                                  5,  6,  7,  8,   // col 1
                                  9,  10, 11, 12,  // col 2
                                  13, 14, 15, 16}; // col 3

    SUBCASE("First upper diagonal (k=1)") {
        diagonal_view<float, 4> diag(data, 4, 4, 1);

        CHECK(diag.size() == 3); // min(4, 4-1) = 3
        CHECK(diag.diagonal_offset() == 1);

        // First upper: (0,1)=5, (1,2)=10, (2,3)=15
        CHECK(diag[0] == doctest::Approx(5.0f));
        CHECK(diag[1] == doctest::Approx(10.0f));
        CHECK(diag[2] == doctest::Approx(15.0f));
    }

    SUBCASE("Second upper diagonal (k=2)") {
        diagonal_view<float, 4> diag(data, 4, 4, 2);

        CHECK(diag.size() == 2); // min(4, 4-2) = 2

        // Second upper: (0,2)=9, (1,3)=14
        CHECK(diag[0] == doctest::Approx(9.0f));
        CHECK(diag[1] == doctest::Approx(14.0f));
    }

    SUBCASE("Third upper diagonal (k=3)") {
        diagonal_view<float, 4> diag(data, 4, 4, 3);

        CHECK(diag.size() == 1); // min(4, 4-3) = 1

        // Third upper: (0,3)=13
        CHECK(diag[0] == doctest::Approx(13.0f));
    }

    SUBCASE("Out of bounds upper diagonal (k=4)") {
        diagonal_view<float, 4> diag(data, 4, 4, 4);

        CHECK(diag.size() == 0); // Out of bounds
    }
}

TEST_CASE("diagonal_view - Lower diagonals") {
    using namespace on::simd;

    // 4x4 matrix
    alignas(16) float data[16] = {1,  2,  3,  4,   // col 0
                                  5,  6,  7,  8,   // col 1
                                  9,  10, 11, 12,  // col 2
                                  13, 14, 15, 16}; // col 3

    SUBCASE("First lower diagonal (k=-1)") {
        diagonal_view<float, 4> diag(data, 4, 4, -1);

        CHECK(diag.size() == 3); // min(4-1, 4) = 3
        CHECK(diag.diagonal_offset() == -1);

        // First lower: (1,0)=2, (2,1)=7, (3,2)=12
        CHECK(diag[0] == doctest::Approx(2.0f));
        CHECK(diag[1] == doctest::Approx(7.0f));
        CHECK(diag[2] == doctest::Approx(12.0f));
    }

    SUBCASE("Second lower diagonal (k=-2)") {
        diagonal_view<float, 4> diag(data, 4, 4, -2);

        CHECK(diag.size() == 2); // min(4-2, 4) = 2

        // Second lower: (2,0)=3, (3,1)=8
        CHECK(diag[0] == doctest::Approx(3.0f));
        CHECK(diag[1] == doctest::Approx(8.0f));
    }

    SUBCASE("Third lower diagonal (k=-3)") {
        diagonal_view<float, 4> diag(data, 4, 4, -3);

        CHECK(diag.size() == 1); // min(4-3, 4) = 1

        // Third lower: (3,0)=4
        CHECK(diag[0] == doctest::Approx(4.0f));
    }

    SUBCASE("Out of bounds lower diagonal (k=-4)") {
        diagonal_view<float, 4> diag(data, 4, 4, -4);

        CHECK(diag.size() == 0); // Out of bounds
    }
}

TEST_CASE("diagonal_view - Non-square matrices") {
    using namespace on::simd;

    SUBCASE("3x5 matrix (more columns than rows)") {
        // 3x5 matrix
        alignas(16) float data[15] = {1,  2,  3,   // col 0
                                      4,  5,  6,   // col 1
                                      7,  8,  9,   // col 2
                                      10, 11, 12,  // col 3
                                      13, 14, 15}; // col 4

        diagonal_view<float, 4> diag(data, 3, 5, 0);
        CHECK(diag.size() == 3); // min(3, 5) = 3

        // Main diagonal: (0,0)=1, (1,1)=5, (2,2)=9
        CHECK(diag[0] == doctest::Approx(1.0f));
        CHECK(diag[1] == doctest::Approx(5.0f));
        CHECK(diag[2] == doctest::Approx(9.0f));

        // Upper diagonal (k=1)
        diagonal_view<float, 4> upper(data, 3, 5, 1);
        CHECK(upper.size() == 3); // min(3, 5-1) = 3

        // Upper: (0,1)=4, (1,2)=8, (2,3)=12
        CHECK(upper[0] == doctest::Approx(4.0f));
        CHECK(upper[1] == doctest::Approx(8.0f));
        CHECK(upper[2] == doctest::Approx(12.0f));

        // Upper diagonal (k=2)
        diagonal_view<float, 4> upper2(data, 3, 5, 2);
        CHECK(upper2.size() == 3); // min(3, 5-2) = 3
    }

    SUBCASE("5x3 matrix (more rows than columns)") {
        // 5x3 matrix
        alignas(32) float data[15] = {1,  2,  3,  4,  5,   // col 0
                                      6,  7,  8,  9,  10,  // col 1
                                      11, 12, 13, 14, 15}; // col 2

        diagonal_view<float, 4> diag(data, 5, 3, 0);
        CHECK(diag.size() == 3); // min(5, 3) = 3

        // Main diagonal: (0,0)=1, (1,1)=7, (2,2)=13
        CHECK(diag[0] == doctest::Approx(1.0f));
        CHECK(diag[1] == doctest::Approx(7.0f));
        CHECK(diag[2] == doctest::Approx(13.0f));

        // Lower diagonal (k=-1)
        diagonal_view<float, 4> lower(data, 5, 3, -1);
        CHECK(lower.size() == 3); // min(5-1, 3) = 3

        // Lower: (1,0)=2, (2,1)=8, (3,2)=14
        CHECK(lower[0] == doctest::Approx(2.0f));
        CHECK(lower[1] == doctest::Approx(8.0f));
        CHECK(lower[2] == doctest::Approx(14.0f));

        // Lower diagonal (k=-2)
        diagonal_view<float, 4> lower2(data, 5, 3, -2);
        CHECK(lower2.size() == 3); // min(5-2, 3) = 3
    }
}

TEST_CASE("diagonal_view - Write operations") {
    using namespace on::simd;

    alignas(16) float data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    SUBCASE("Modify main diagonal") {
        diagonal_view<float, 4> diag(data, 4, 4, 0);

        diag[0] = 100.0f;
        diag[1] = 200.0f;
        diag[2] = 300.0f;
        diag[3] = 400.0f;

        // Verify changes in original data
        CHECK(data[0] == doctest::Approx(100.0f));  // (0,0)
        CHECK(data[5] == doctest::Approx(200.0f));  // (1,1)
        CHECK(data[10] == doctest::Approx(300.0f)); // (2,2)
        CHECK(data[15] == doctest::Approx(400.0f)); // (3,3)
    }

    SUBCASE("Modify upper diagonal") {
        diagonal_view<float, 4> diag(data, 4, 4, 1);

        diag[0] = 10.0f;
        diag[1] = 20.0f;
        diag[2] = 30.0f;

        // Verify changes: (0,1)=5, (1,2)=10, (2,3)=15
        CHECK(data[4] == doctest::Approx(10.0f));  // (0,1)
        CHECK(data[9] == doctest::Approx(20.0f));  // (1,2)
        CHECK(data[14] == doctest::Approx(30.0f)); // (2,3)
    }
}

TEST_CASE("diagonal_view - Pack operations") {
    using namespace on::simd;
    using pack_t = pack<float, 4>;

    alignas(16) float data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    SUBCASE("Load pack from main diagonal") {
        diagonal_view<float, 4> diag(data, 4, 4, 0);

        CHECK(diag.num_packs() == 1);
        CHECK(diag.tail_size() == 4);

        pack_t p = diag.load_pack(0);
        CHECK(p[0] == doctest::Approx(1.0f));
        CHECK(p[1] == doctest::Approx(6.0f));
        CHECK(p[2] == doctest::Approx(11.0f));
        CHECK(p[3] == doctest::Approx(16.0f));
    }

    SUBCASE("Store pack to main diagonal") {
        diagonal_view<float, 4> diag(data, 4, 4, 0);

        pack_t p = pack_t::set(100.0f, 200.0f, 300.0f, 400.0f);
        diag.store_pack(0, p);

        CHECK(data[0] == doctest::Approx(100.0f));  // (0,0)
        CHECK(data[5] == doctest::Approx(200.0f));  // (1,1)
        CHECK(data[10] == doctest::Approx(300.0f)); // (2,2)
        CHECK(data[15] == doctest::Approx(400.0f)); // (3,3)
    }

    SUBCASE("Tail-safe pack operations") {
        diagonal_view<float, 4> diag(data, 4, 4, 1); // size = 3

        CHECK(diag.num_packs() == 1);
        CHECK(diag.tail_size() == 3);

        // Load with tail handling
        pack_t p = diag.load_pack_tail(0);
        CHECK(p[0] == doctest::Approx(5.0f));
        CHECK(p[1] == doctest::Approx(10.0f));
        CHECK(p[2] == doctest::Approx(15.0f));
        // p[3] is undefined (tail element)
    }
}

TEST_CASE("diagonal_view - Helper functions") {
    using namespace on::simd;

    alignas(16) float data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    SUBCASE("diagonal() - main diagonal") {
        auto diag = diagonal<float, 4>(data, 4, 4);

        CHECK(diag.size() == 4);
        CHECK(diag[0] == doctest::Approx(1.0f));
        CHECK(diag[1] == doctest::Approx(6.0f));
    }

    SUBCASE("diagonal() - with offset") {
        auto diag = diagonal<float, 4>(data, 4, 4, 1);

        CHECK(diag.size() == 3);
        CHECK(diag[0] == doctest::Approx(5.0f));
        CHECK(diag[1] == doctest::Approx(10.0f));
    }

    SUBCASE("diagonal() - from matrix_view") {
        matrix_view<float, 4> m(data, 4, 4);
        auto diag = diagonal(m, 0);

        CHECK(diag.size() == 4);
        CHECK(diag[0] == doctest::Approx(1.0f));
        CHECK(diag[1] == doctest::Approx(6.0f));

        auto upper = diagonal(m, 1);
        CHECK(upper.size() == 3);
        CHECK(upper[0] == doctest::Approx(5.0f));
    }
}

TEST_CASE("diagonal_view - Integration with Matrix") {
    using namespace on::simd;

    // Create backing storage and view
    datapod::mat::Matrix<float, 4, 4> m_storage;
    on::simd::Matrix<float, 4, 4> m(m_storage);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            m(i, j) = static_cast<float>(i * 4 + j + 1);
        }
    }

    SUBCASE("Read diagonal from Matrix") {
        matrix_view<float, 4> mv(m.data(), 4, 4);
        auto diag = diagonal(mv, 0);

        // Matrix is column-major, so (0,0)=1, (1,1)=6, (2,2)=11, (3,3)=16
        CHECK(diag[0] == doctest::Approx(1.0f));
        CHECK(diag[1] == doctest::Approx(6.0f));
        CHECK(diag[2] == doctest::Approx(11.0f));
        CHECK(diag[3] == doctest::Approx(16.0f));
    }

    SUBCASE("Modify diagonal of Matrix") {
        matrix_view<float, 4> mv(m.data(), 4, 4);
        auto diag = diagonal(mv, 0);

        diag[0] = 99.0f;
        diag[1] = 88.0f;
        diag[2] = 77.0f;
        diag[3] = 66.0f;

        // Verify changes
        CHECK(m(0, 0) == doctest::Approx(99.0f));
        CHECK(m(1, 1) == doctest::Approx(88.0f));
        CHECK(m(2, 2) == doctest::Approx(77.0f));
        CHECK(m(3, 3) == doctest::Approx(66.0f));
    }
}

TEST_CASE("diagonal_view - const access") {
    using namespace on::simd;

    alignas(16) const float data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    SUBCASE("const diagonal view") {
        diagonal_view<const float, 4> diag(data, 4, 4, 0);

        CHECK(diag.size() == 4);
        CHECK(diag.at(0) == doctest::Approx(1.0f));
        CHECK(diag.at(1) == doctest::Approx(6.0f));
        CHECK(diag.at(2) == doctest::Approx(11.0f));
        CHECK(diag.at(3) == doctest::Approx(16.0f));
    }
}
