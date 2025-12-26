// =============================================================================
// test/simd/view/matrix_slice_test.cpp
// Tests for matrix_view::slice() - 2D slicing with seq/fseq/all/fix
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/view/matrix_view.hpp>
#include <optinum/simd/view/slice.hpp>
#include <vector>

using namespace optinum::simd;

TEST_CASE("Matrix slicing - basic row/column slices") {
    // Create a 4x5 matrix (column-major)
    //   0  4  8 12 16
    //   1  5  9 13 17
    //   2  6 10 14 18
    //   3  7 11 15 19
    std::vector<float> data(20);
    for (std::size_t i = 0; i < 20; ++i) {
        data[i] = static_cast<float>(i);
    }

    matrix_view<float, 4> m(data.data(), 4, 5);

    SUBCASE("Slice rows with seq") {
        auto sliced = m.slice(seq(1, 3), all);
        CHECK(sliced.rows() == 2);
        CHECK(sliced.cols() == 5);
        CHECK(sliced(0, 0) == 1.0f);
        CHECK(sliced(1, 0) == 2.0f);
        CHECK(sliced(0, 1) == 5.0f);
        CHECK(sliced(1, 4) == 18.0f);
    }

    SUBCASE("Slice columns with seq") {
        auto sliced = m.slice(all, seq(1, 4));
        CHECK(sliced.rows() == 4);
        CHECK(sliced.cols() == 3);
        CHECK(sliced(0, 0) == 4.0f);
        CHECK(sliced(1, 0) == 5.0f);
        CHECK(sliced(3, 2) == 15.0f);
    }

    SUBCASE("Slice both dimensions") {
        auto sliced = m.slice(seq(1, 3), seq(2, 5));
        CHECK(sliced.rows() == 2);
        CHECK(sliced.cols() == 3);
        CHECK(sliced(0, 0) == 9.0f);
        CHECK(sliced(1, 0) == 10.0f);
        CHECK(sliced(0, 1) == 13.0f);
        CHECK(sliced(1, 2) == 18.0f);
    }

    SUBCASE("Single row with all columns") {
        auto sliced = m.slice(seq(2, 3), all);
        CHECK(sliced.rows() == 1);
        CHECK(sliced.cols() == 5);
        CHECK(sliced(0, 0) == 2.0f);
        CHECK(sliced(0, 1) == 6.0f);
        CHECK(sliced(0, 4) == 18.0f);
    }

    SUBCASE("Single column with all rows") {
        auto sliced = m.slice(all, seq(3, 4));
        CHECK(sliced.rows() == 4);
        CHECK(sliced.cols() == 1);
        CHECK(sliced(0, 0) == 12.0f);
        CHECK(sliced(1, 0) == 13.0f);
        CHECK(sliced(3, 0) == 15.0f);
    }
}

TEST_CASE("Matrix slicing - compile-time fseq") {
    std::vector<double> data(12);
    for (std::size_t i = 0; i < 12; ++i) {
        data[i] = static_cast<double>(i * 10);
    }

    matrix_view<double, 2> m(data.data(), 3, 4);

    SUBCASE("fseq for rows") {
        auto sliced = m.slice(fseq<0, 2>(), all);
        CHECK(sliced.rows() == 2);
        CHECK(sliced.cols() == 4);
        CHECK(sliced(0, 0) == 0.0);
        CHECK(sliced(1, 0) == 10.0);
        CHECK(sliced(0, 3) == 90.0);
    }

    SUBCASE("fseq for columns") {
        auto sliced = m.slice(all, fseq<1, 3>());
        CHECK(sliced.rows() == 3);
        CHECK(sliced.cols() == 2);
        CHECK(sliced(0, 0) == 30.0);
        CHECK(sliced(1, 0) == 40.0);
        CHECK(sliced(2, 1) == 80.0);
    }

    SUBCASE("fseq for both dimensions") {
        auto sliced = m.slice(fseq<1, 3>(), fseq<0, 2>());
        CHECK(sliced.rows() == 2);
        CHECK(sliced.cols() == 2);
        CHECK(sliced(0, 0) == 10.0);
        CHECK(sliced(1, 0) == 20.0);
        CHECK(sliced(0, 1) == 40.0);
        CHECK(sliced(1, 1) == 50.0);
    }
}

TEST_CASE("Matrix slicing - fixed indices with fix<N>") {
    std::vector<float> data(16);
    // Create 4x4 matrix with values r*10 + c
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            data[r + c * 4] = static_cast<float>(r * 10 + c);
        }
    }

    matrix_view<float, 4> m(data.data(), 4, 4);

    SUBCASE("Single row (fix<N> for row)") {
        auto row_vec = m.slice(fix<2>(), all);
        // Should return vector_view with 4 elements
        CHECK(row_vec.size() == 4);
        CHECK(row_vec[0] == 20.0f);
        CHECK(row_vec[1] == 21.0f);
        CHECK(row_vec[2] == 22.0f);
        CHECK(row_vec[3] == 23.0f);
    }

    SUBCASE("Single column (fix<N> for column)") {
        auto col_vec = m.slice(all, fix<1>());
        // Should return vector_view with 4 elements
        CHECK(col_vec.size() == 4);
        CHECK(col_vec[0] == 1.0f);
        CHECK(col_vec[1] == 11.0f);
        CHECK(col_vec[2] == 21.0f);
        CHECK(col_vec[3] == 31.0f);
    }

    SUBCASE("Single element (fix<N> for both)") {
        auto elem_mat = m.slice(fix<2>(), fix<3>());
        // Currently returns 1x1 matrix
        CHECK(elem_mat.rows() == 1);
        CHECK(elem_mat.cols() == 1);
        CHECK(elem_mat(0, 0) == 23.0f);
    }
}

TEST_CASE("Matrix slicing - strided slices") {
    std::vector<float> data(36);
    for (std::size_t i = 0; i < 36; ++i) {
        data[i] = static_cast<float>(i);
    }

    matrix_view<float, 4> m(data.data(), 6, 6);

    SUBCASE("Every other row") {
        auto sliced = m.slice(seq(0, 6, 2), all);
        CHECK(sliced.rows() == 3);
        CHECK(sliced.cols() == 6);
        CHECK(sliced(0, 0) == 0.0f);
        CHECK(sliced(1, 0) == 2.0f);
        CHECK(sliced(2, 0) == 4.0f);
        CHECK(sliced(0, 1) == 6.0f);
        CHECK(sliced(2, 5) == 34.0f);
    }

    SUBCASE("Every other column") {
        auto sliced = m.slice(all, seq(0, 6, 2));
        CHECK(sliced.rows() == 6);
        CHECK(sliced.cols() == 3);
        CHECK(sliced(0, 0) == 0.0f);
        CHECK(sliced(0, 1) == 12.0f);
        CHECK(sliced(0, 2) == 24.0f);
        CHECK(sliced(5, 0) == 5.0f);
        CHECK(sliced(5, 2) == 29.0f);
    }

    SUBCASE("Every 2nd row and 3rd column") {
        auto sliced = m.slice(seq(1, 6, 2), seq(0, 6, 3));
        CHECK(sliced.rows() == 3);
        CHECK(sliced.cols() == 2);
        CHECK(sliced(0, 0) == 1.0f);
        CHECK(sliced(1, 0) == 3.0f);
        CHECK(sliced(2, 0) == 5.0f);
        CHECK(sliced(0, 1) == 19.0f);
        CHECK(sliced(2, 1) == 23.0f);
    }
}

TEST_CASE("Matrix slicing - edge cases") {
    std::vector<double> data(25);
    for (std::size_t i = 0; i < 25; ++i) {
        data[i] = static_cast<double>(i);
    }

    matrix_view<double, 2> m(data.data(), 5, 5);

    SUBCASE("First row") {
        auto sliced = m.slice(seq(0, 1), all);
        CHECK(sliced.rows() == 1);
        CHECK(sliced.cols() == 5);
        CHECK(sliced(0, 0) == 0.0);
        CHECK(sliced(0, 4) == 20.0);
    }

    SUBCASE("Last row") {
        auto sliced = m.slice(seq(4, 5), all);
        CHECK(sliced.rows() == 1);
        CHECK(sliced.cols() == 5);
        CHECK(sliced(0, 0) == 4.0);
        CHECK(sliced(0, 4) == 24.0);
    }

    SUBCASE("First column") {
        auto sliced = m.slice(all, seq(0, 1));
        CHECK(sliced.rows() == 5);
        CHECK(sliced.cols() == 1);
        CHECK(sliced(0, 0) == 0.0);
        CHECK(sliced(4, 0) == 4.0);
    }

    SUBCASE("Last column") {
        auto sliced = m.slice(all, seq(4, 5));
        CHECK(sliced.rows() == 5);
        CHECK(sliced.cols() == 1);
        CHECK(sliced(0, 0) == 20.0);
        CHECK(sliced(4, 0) == 24.0);
    }

    SUBCASE("1x1 slice from center") {
        auto sliced = m.slice(seq(2, 3), seq(2, 3));
        CHECK(sliced.rows() == 1);
        CHECK(sliced.cols() == 1);
        CHECK(sliced(0, 0) == 12.0);
    }
}

TEST_CASE("Matrix slicing - mutability") {
    std::vector<float> data(9);
    for (std::size_t i = 0; i < 9; ++i) {
        data[i] = static_cast<float>(i);
    }

    matrix_view<float, 4> m(data.data(), 3, 3);

    SUBCASE("Modify through slice") {
        auto sliced = m.slice(seq(0, 2), seq(0, 2));
        sliced(0, 0) = 100.0f;
        sliced(1, 1) = 200.0f;

        // Check original data was modified
        CHECK(data[0] == 100.0f);
        CHECK(data[4] == 200.0f);
        CHECK(data[3] == 3.0f); // unchanged
    }

    SUBCASE("Modify vector slice") {
        auto row = m.slice(fix<1>(), all);
        row[0] = 99.0f;
        row[2] = 88.0f;

        // row 1, col 0 -> index 1
        CHECK(data[1] == 99.0f);
        // row 1, col 2 -> index 1 + 2*3 = 7
        CHECK(data[7] == 88.0f);
    }
}
