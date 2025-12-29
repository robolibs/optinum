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

// =============================================================================
// Tests for new matrix_view operations (arithmetic, fill, trace, etc.)
// =============================================================================

TEST_CASE("matrix_view - fill operations") {
    std::vector<float> data(12);
    matrix_view<float, 4> m(data.data(), 3, 4);

    SUBCASE("fill with constant") {
        m.fill(3.14f);
        for (std::size_t i = 0; i < 12; ++i) {
            CHECK(data[i] == doctest::Approx(3.14f));
        }
    }

    SUBCASE("set_identity for square matrix") {
        std::vector<float> sq_data(9);
        matrix_view<float, 4> sq(sq_data.data(), 3, 3);
        sq.set_identity();

        // Check diagonal is 1
        CHECK(sq_data[0] == 1.0f); // (0,0)
        CHECK(sq_data[4] == 1.0f); // (1,1)
        CHECK(sq_data[8] == 1.0f); // (2,2)

        // Check off-diagonal is 0
        CHECK(sq_data[1] == 0.0f);
        CHECK(sq_data[2] == 0.0f);
        CHECK(sq_data[3] == 0.0f);
        CHECK(sq_data[5] == 0.0f);
        CHECK(sq_data[6] == 0.0f);
        CHECK(sq_data[7] == 0.0f);
    }

    SUBCASE("set_identity for non-square matrix") {
        m.set_identity();
        // 3x4 matrix, diagonal has min(3,4)=3 elements
        CHECK(data[0] == 1.0f);  // (0,0)
        CHECK(data[4] == 1.0f);  // (1,1)
        CHECK(data[8] == 1.0f);  // (2,2)
        CHECK(data[1] == 0.0f);  // off-diagonal
        CHECK(data[9] == 0.0f);  // (0,3)
        CHECK(data[10] == 0.0f); // (1,3)
        CHECK(data[11] == 0.0f); // (2,3)
    }
}

TEST_CASE("matrix_view - static factory functions") {
    std::vector<double> data(6);

    SUBCASE("zeros") {
        matrix_view<double, 4>::zeros(data.data(), 2, 3);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(data[i] == 0.0);
        }
    }

    SUBCASE("ones") {
        matrix_view<double, 4>::ones(data.data(), 2, 3);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(data[i] == 1.0);
        }
    }

    SUBCASE("identity") {
        std::vector<double> sq_data(9);
        matrix_view<double, 4>::identity(sq_data.data(), 3, 3);
        CHECK(sq_data[0] == 1.0);
        CHECK(sq_data[4] == 1.0);
        CHECK(sq_data[8] == 1.0);
        CHECK(sq_data[1] == 0.0);
        CHECK(sq_data[3] == 0.0);
    }
}

TEST_CASE("matrix_view - compound assignment operators") {
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {10, 20, 30, 40, 50, 60};

    matrix_view<float, 4> a(a_data.data(), 2, 3);
    matrix_view<float, 4> b(b_data.data(), 2, 3);

    SUBCASE("operator+=") {
        a += b;
        CHECK(a_data[0] == doctest::Approx(11.0f));
        CHECK(a_data[1] == doctest::Approx(22.0f));
        CHECK(a_data[5] == doctest::Approx(66.0f));
    }

    SUBCASE("operator-=") {
        a -= b;
        CHECK(a_data[0] == doctest::Approx(-9.0f));
        CHECK(a_data[1] == doctest::Approx(-18.0f));
        CHECK(a_data[5] == doctest::Approx(-54.0f));
    }

    SUBCASE("operator*= (scalar)") {
        a *= 2.0f;
        CHECK(a_data[0] == doctest::Approx(2.0f));
        CHECK(a_data[1] == doctest::Approx(4.0f));
        CHECK(a_data[5] == doctest::Approx(12.0f));
    }

    SUBCASE("operator/= (scalar)") {
        a /= 2.0f;
        CHECK(a_data[0] == doctest::Approx(0.5f));
        CHECK(a_data[1] == doctest::Approx(1.0f));
        CHECK(a_data[5] == doctest::Approx(3.0f));
    }
}

TEST_CASE("matrix_view - element-wise operations") {
    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {2, 4, 6, 8};

    matrix_view<float, 4> a(a_data.data(), 2, 2);
    matrix_view<float, 4> b(b_data.data(), 2, 2);

    SUBCASE("hadamard_inplace") {
        a.hadamard_inplace(b);
        CHECK(a_data[0] == doctest::Approx(2.0f));
        CHECK(a_data[1] == doctest::Approx(8.0f));
        CHECK(a_data[2] == doctest::Approx(18.0f));
        CHECK(a_data[3] == doctest::Approx(32.0f));
    }

    SUBCASE("div_inplace") {
        a.div_inplace(b);
        CHECK(a_data[0] == doctest::Approx(0.5f));
        CHECK(a_data[1] == doctest::Approx(0.5f));
        CHECK(a_data[2] == doctest::Approx(0.5f));
        CHECK(a_data[3] == doctest::Approx(0.5f));
    }
}

TEST_CASE("matrix_view - reductions") {
    // 2x3 matrix:
    //   1 3 5
    //   2 4 6
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    matrix_view<double, 4> m(data.data(), 2, 3);

    SUBCASE("trace") {
        // Trace is sum of diagonal: 1 + 4 = 5 (min(2,3)=2 diagonal elements)
        CHECK(m.trace() == doctest::Approx(5.0));
    }

    SUBCASE("sum") { CHECK(m.sum() == doctest::Approx(21.0)); }

    SUBCASE("frobenius_norm") {
        // sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91)
        CHECK(m.frobenius_norm() == doctest::Approx(std::sqrt(91.0)));
    }
}

TEST_CASE("matrix_view - free function operations") {
    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {10, 20, 30, 40};
    std::vector<float> out_data(4);

    matrix_view<float, 4> a(a_data.data(), 2, 2);
    matrix_view<float, 4> b(b_data.data(), 2, 2);

    SUBCASE("add") {
        add(out_data.data(), a, b);
        CHECK(out_data[0] == doctest::Approx(11.0f));
        CHECK(out_data[1] == doctest::Approx(22.0f));
        CHECK(out_data[3] == doctest::Approx(44.0f));
    }

    SUBCASE("sub") {
        sub(out_data.data(), a, b);
        CHECK(out_data[0] == doctest::Approx(-9.0f));
        CHECK(out_data[1] == doctest::Approx(-18.0f));
        CHECK(out_data[3] == doctest::Approx(-36.0f));
    }

    SUBCASE("hadamard") {
        hadamard(out_data.data(), a, b);
        CHECK(out_data[0] == doctest::Approx(10.0f));
        CHECK(out_data[1] == doctest::Approx(40.0f));
        CHECK(out_data[3] == doctest::Approx(160.0f));
    }

    SUBCASE("mul_scalar") {
        mul_scalar(out_data.data(), a, 3.0f);
        CHECK(out_data[0] == doctest::Approx(3.0f));
        CHECK(out_data[1] == doctest::Approx(6.0f));
        CHECK(out_data[3] == doctest::Approx(12.0f));
    }

    SUBCASE("negate") {
        negate(out_data.data(), a);
        CHECK(out_data[0] == doctest::Approx(-1.0f));
        CHECK(out_data[1] == doctest::Approx(-2.0f));
        CHECK(out_data[3] == doctest::Approx(-4.0f));
    }
}

TEST_CASE("matrix_view - transpose") {
    // 2x3 matrix (column-major):
    //   1 3 5
    //   2 4 6
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> out_data(6);

    matrix_view<float, 4> a(a_data.data(), 2, 3);

    transpose(out_data.data(), a);

    // Result should be 3x2 (column-major):
    //   1 2
    //   3 4
    //   5 6
    matrix_view<float, 4> out(out_data.data(), 3, 2);
    CHECK(out.at(0, 0) == doctest::Approx(1.0f));
    CHECK(out.at(1, 0) == doctest::Approx(3.0f));
    CHECK(out.at(2, 0) == doctest::Approx(5.0f));
    CHECK(out.at(0, 1) == doctest::Approx(2.0f));
    CHECK(out.at(1, 1) == doctest::Approx(4.0f));
    CHECK(out.at(2, 1) == doctest::Approx(6.0f));
}

TEST_CASE("matrix_view - matmul") {
    // A = 2x3 (column-major):
    //   1 3 5
    //   2 4 6
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};

    // B = 3x2 (column-major):
    //   1 4
    //   2 5
    //   3 6
    std::vector<float> b_data = {1, 2, 3, 4, 5, 6};

    // Result C = A * B = 2x2
    std::vector<float> c_data(4);

    matrix_view<float, 4> a(a_data.data(), 2, 3);
    matrix_view<float, 4> b(b_data.data(), 3, 2);

    matmul(c_data.data(), a, b);

    // C[0,0] = 1*1 + 3*2 + 5*3 = 1 + 6 + 15 = 22
    // C[1,0] = 2*1 + 4*2 + 6*3 = 2 + 8 + 18 = 28
    // C[0,1] = 1*4 + 3*5 + 5*6 = 4 + 15 + 30 = 49
    // C[1,1] = 2*4 + 4*5 + 6*6 = 8 + 20 + 36 = 64
    matrix_view<float, 4> c(c_data.data(), 2, 2);
    CHECK(c.at(0, 0) == doctest::Approx(22.0f));
    CHECK(c.at(1, 0) == doctest::Approx(28.0f));
    CHECK(c.at(0, 1) == doctest::Approx(49.0f));
    CHECK(c.at(1, 1) == doctest::Approx(64.0f));
}

TEST_CASE("matrix_view - matvec") {
    // A = 2x3 (column-major):
    //   1 3 5
    //   2 4 6
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};

    // x = [1, 2, 3]
    std::vector<float> x_data = {1, 2, 3};

    // Result y = A * x = [1*1 + 3*2 + 5*3, 2*1 + 4*2 + 6*3] = [22, 28]
    std::vector<float> y_data(2);

    matrix_view<float, 4> a(a_data.data(), 2, 3);
    vector_view<float, 4> x(x_data.data(), 3);

    matvec(y_data.data(), a, x);

    CHECK(y_data[0] == doctest::Approx(22.0f));
    CHECK(y_data[1] == doctest::Approx(28.0f));
}

TEST_CASE("matrix_view - comparison operators") {
    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {1, 2, 3, 4};
    std::vector<float> c_data = {1, 2, 3, 5};

    matrix_view<float, 4> a(a_data.data(), 2, 2);
    matrix_view<float, 4> b(b_data.data(), 2, 2);
    matrix_view<float, 4> c(c_data.data(), 2, 2);

    CHECK(a == b);
    CHECK_FALSE(a != b);
    CHECK_FALSE(a == c);
    CHECK(a != c);
}

TEST_CASE("matrix_view - copy") {
    std::vector<float> src_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> dst_data(6);

    matrix_view<float, 4> src(src_data.data(), 2, 3);
    matrix_view<float, 4> dst(dst_data.data(), 2, 3);

    copy(dst, src);

    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(dst_data[i] == doctest::Approx(src_data[i]));
    }
}

TEST_CASE("matrix_view - 1D indexing") {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    matrix_view<float, 4> m(data.data(), 2, 3);

    SUBCASE("at_linear read") {
        CHECK(m.at_linear(0) == 1.0f);
        CHECK(m.at_linear(1) == 2.0f);
        CHECK(m.at_linear(5) == 6.0f);
    }

    SUBCASE("operator[] write") {
        m[0] = 100.0f;
        m[5] = 200.0f;
        CHECK(data[0] == 100.0f);
        CHECK(data[5] == 200.0f);
    }
}

TEST_CASE("matrix_view - trace free function") {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix_view<double, 4> m(data.data(), 3, 3);

    // Diagonal: 1, 5, 9 (indices 0, 4, 8 in column-major)
    CHECK(trace(m) == doctest::Approx(15.0));
}

TEST_CASE("matrix_view - frobenius_norm free function") {
    std::vector<float> data = {1, 2, 3, 4};
    matrix_view<float, 4> m(data.data(), 2, 2);

    // sqrt(1 + 4 + 9 + 16) = sqrt(30)
    CHECK(frobenius_norm(m) == doctest::Approx(std::sqrt(30.0f)));
}
