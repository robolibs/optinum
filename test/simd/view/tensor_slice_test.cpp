// =============================================================================
// test/simd/view/tensor_slice_test.cpp
// Tests for tensor_view::slice() - N-dimensional slicing with seq/fseq/all/fix
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/view/slice.hpp>
#include <optinum/simd/view/tensor_view.hpp>
#include <vector>

using namespace optinum::simd;

TEST_CASE("Tensor slicing - 3D basic slicing") {
    // Create a 3x4x5 tensor (60 elements)
    // Shape: (3 rows, 4 columns, 5 depth)
    std::vector<float> data(60);
    for (std::size_t i = 0; i < 60; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Row-major strides for simplicity: depth varies fastest
    // stride[0] = 20 (row), stride[1] = 5 (col), stride[2] = 1 (depth)
    tensor_view<float, 4, 3> t(data.data(), {3, 4, 5}, {20, 5, 1});

    SUBCASE("Slice first dimension only") {
        auto sliced = t.slice(seq(1, 3), all, all);
        CHECK(sliced.extent(0) == 2); // rows 1..2
        CHECK(sliced.extent(1) == 4); // all columns
        CHECK(sliced.extent(2) == 5); // all depth

        // Element at (0,0,0) of sliced should be element (1,0,0) of original
        // Original (1,0,0) -> offset = 1*20 + 0*5 + 0*1 = 20
        CHECK(sliced(0, 0, 0) == 20.0f);
        CHECK(sliced(1, 0, 0) == 40.0f); // row 2
    }

    SUBCASE("Slice second dimension only") {
        auto sliced = t.slice(all, seq(1, 3), all);
        CHECK(sliced.extent(0) == 3); // all rows
        CHECK(sliced.extent(1) == 2); // cols 1..2
        CHECK(sliced.extent(2) == 5); // all depth

        // Element at (0,0,0) of sliced should be element (0,1,0) of original
        // Original (0,1,0) -> offset = 0*20 + 1*5 + 0*1 = 5
        CHECK(sliced(0, 0, 0) == 5.0f);
    }

    SUBCASE("Slice third dimension only") {
        auto sliced = t.slice(all, all, seq(2, 5));
        CHECK(sliced.extent(0) == 3); // all rows
        CHECK(sliced.extent(1) == 4); // all columns
        CHECK(sliced.extent(2) == 3); // depth 2..4

        // Element at (0,0,0) of sliced should be element (0,0,2) of original
        // Original (0,0,2) -> offset = 0*20 + 0*5 + 2*1 = 2
        CHECK(sliced(0, 0, 0) == 2.0f);
    }

    SUBCASE("Slice all three dimensions") {
        auto sliced = t.slice(seq(0, 2), seq(1, 3), seq(1, 4));
        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 2);
        CHECK(sliced.extent(2) == 3);

        // Element at (0,0,0) of sliced should be element (0,1,1) of original
        // Original (0,1,1) -> offset = 0*20 + 1*5 + 1*1 = 6
        CHECK(sliced(0, 0, 0) == 6.0f);

        // Element at (1,1,2) of sliced should be element (1,2,3) of original
        // Original (1,2,3) -> offset = 1*20 + 2*5 + 3*1 = 33
        CHECK(sliced(1, 1, 2) == 33.0f);
    }
}

TEST_CASE("Tensor slicing - compile-time fseq") {
    std::vector<double> data(24);
    for (std::size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<double>(i * 10);
    }

    // 2x3x4 tensor
    tensor_view<double, 2, 3> t(data.data(), {2, 3, 4}, {12, 4, 1});

    SUBCASE("fseq for first dimension") {
        auto sliced = t.slice(fseq<0, 1>(), all, all);
        CHECK(sliced.extent(0) == 1);
        CHECK(sliced.extent(1) == 3);
        CHECK(sliced.extent(2) == 4);
        CHECK(sliced(0, 0, 0) == 0.0);
    }

    SUBCASE("fseq for second dimension") {
        auto sliced = t.slice(all, fseq<1, 3>(), all);
        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 2);
        CHECK(sliced.extent(2) == 4);
        // (0,1,0) -> offset = 0*12 + 1*4 + 0*1 = 4
        CHECK(sliced(0, 0, 0) == 40.0);
    }

    SUBCASE("fseq for all dimensions") {
        auto sliced = t.slice(fseq<0, 2>(), fseq<1, 3>(), fseq<0, 2>());
        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 2);
        CHECK(sliced.extent(2) == 2);
        // (0,0,0) in sliced is (0,1,0) in original -> offset = 0*12 + 1*4 + 0*1 = 4
        CHECK(sliced(0, 0, 0) == 40.0);
        // (1,1,1) in sliced is (1,2,1) in original -> offset = 1*12 + 2*4 + 1*1 = 21
        CHECK(sliced(1, 1, 1) == 210.0);
    }
}

TEST_CASE("Tensor slicing - strided slicing") {
    std::vector<float> data(60);
    for (std::size_t i = 0; i < 60; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 3x4x5 tensor
    tensor_view<float, 4, 3> t(data.data(), {3, 4, 5}, {20, 5, 1});

    SUBCASE("Every other element in first dimension") {
        auto sliced = t.slice(seq(0, 3, 2), all, all);
        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 4);
        CHECK(sliced.extent(2) == 5);

        CHECK(sliced(0, 0, 0) == 0.0f);
        CHECK(sliced(1, 0, 0) == 40.0f); // skip row 1, go to row 2
    }

    SUBCASE("Every other element in second dimension") {
        auto sliced = t.slice(all, seq(0, 4, 2), all);
        CHECK(sliced.extent(0) == 3);
        CHECK(sliced.extent(1) == 2);
        CHECK(sliced.extent(2) == 5);

        CHECK(sliced(0, 0, 0) == 0.0f);
        CHECK(sliced(0, 1, 0) == 10.0f); // skip col 1, go to col 2
    }

    SUBCASE("Strided on all dimensions") {
        auto sliced = t.slice(seq(0, 3, 2), seq(0, 4, 2), seq(0, 5, 2));
        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 2);
        CHECK(sliced.extent(2) == 3);

        // (0,0,0) -> (0,0,0) = 0
        CHECK(sliced(0, 0, 0) == 0.0f);
        // (1,1,1) -> (2,2,2) = 2*20 + 2*5 + 2 = 52
        CHECK(sliced(1, 1, 1) == 52.0f);
    }
}

TEST_CASE("Tensor slicing - edge cases") {
    std::vector<double> data(60);
    for (std::size_t i = 0; i < 60; ++i) {
        data[i] = static_cast<double>(i);
    }

    // 3x4x5 tensor
    tensor_view<double, 2, 3> t(data.data(), {3, 4, 5}, {20, 5, 1});

    SUBCASE("Single element slice in each dimension") {
        auto sliced = t.slice(seq(1, 2), seq(2, 3), seq(3, 4));
        CHECK(sliced.extent(0) == 1);
        CHECK(sliced.extent(1) == 1);
        CHECK(sliced.extent(2) == 1);

        // (1,2,3) -> offset = 1*20 + 2*5 + 3*1 = 33
        CHECK(sliced(0, 0, 0) == 33.0);
    }

    SUBCASE("First slice along all dimensions") {
        auto sliced = t.slice(seq(0, 1), seq(0, 1), seq(0, 1));
        CHECK(sliced.extent(0) == 1);
        CHECK(sliced.extent(1) == 1);
        CHECK(sliced.extent(2) == 1);
        CHECK(sliced(0, 0, 0) == 0.0);
    }

    SUBCASE("Last slice along all dimensions") {
        auto sliced = t.slice(seq(2, 3), seq(3, 4), seq(4, 5));
        CHECK(sliced.extent(0) == 1);
        CHECK(sliced.extent(1) == 1);
        CHECK(sliced.extent(2) == 1);

        // (2,3,4) -> offset = 2*20 + 3*5 + 4*1 = 59
        CHECK(sliced(0, 0, 0) == 59.0);
    }
}

TEST_CASE("Tensor slicing - mutability") {
    std::vector<float> data(24);
    for (std::size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 2x3x4 tensor
    tensor_view<float, 4, 3> t(data.data(), {2, 3, 4}, {12, 4, 1});

    SUBCASE("Modify through slice") {
        auto sliced = t.slice(seq(0, 2), seq(0, 2), seq(0, 2));

        sliced(0, 0, 0) = 100.0f;
        sliced(1, 1, 1) = 200.0f;

        // Check original data was modified
        CHECK(data[0] == 100.0f);
        // (1,1,1) -> offset = 1*12 + 1*4 + 1*1 = 17
        CHECK(data[17] == 200.0f);
    }

    SUBCASE("Modify strided slice") {
        auto sliced = t.slice(all, seq(0, 3, 2), all);

        sliced(0, 0, 0) = 99.0f;
        sliced(0, 1, 0) = 88.0f;

        CHECK(data[0] == 99.0f);
        // (0,1,0) with step 2 in dim 1 -> (0,2,0) = 0*12 + 2*4 + 0 = 8
        CHECK(data[8] == 88.0f);
    }
}

TEST_CASE("Tensor slicing - 4D tensor") {
    // 2x2x2x2 tensor (16 elements)
    std::vector<float> data(16);
    for (std::size_t i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i);
    }

    tensor_view<float, 4, 4> t(data.data(), {2, 2, 2, 2}, {8, 4, 2, 1});

    SUBCASE("Slice all dimensions") {
        auto sliced = t.slice(all, seq(0, 1), all, seq(1, 2));
        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 1);
        CHECK(sliced.extent(2) == 2);
        CHECK(sliced.extent(3) == 1);

        // (0,0,0,1) -> offset = 0*8 + 0*4 + 0*2 + 1*1 = 1
        CHECK(sliced(0, 0, 0, 0) == 1.0f);
        // (1,0,1,0) in sliced is (1,0,1,1) in original
        // offset = 1*8 + 0*4 + 1*2 + 1*1 = 11
        CHECK(sliced(1, 0, 1, 0) == 11.0f);
    }

    SUBCASE("Strided 4D slice") {
        auto sliced = t.slice(seq(0, 2, 1), seq(0, 2, 1), seq(0, 2, 2), seq(0, 2, 1));
        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 2);
        CHECK(sliced.extent(2) == 1); // step 2 over size 2 gives 1 element
        CHECK(sliced.extent(3) == 2);

        CHECK(sliced(0, 0, 0, 0) == 0.0f);
        // (1,1,0,1) in sliced is (1,1,0,1) in original (step 2 in dim2 means only idx 0)
        // offset = 1*8 + 1*4 + 0*2 + 1*1 = 13
        CHECK(sliced(1, 1, 0, 1) == 13.0f);
    }
}

TEST_CASE("Tensor slicing - dimensionality reduction to matrix") {
    std::vector<float> data(24);
    for (std::size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 2x3x4 tensor
    tensor_view<float, 4, 3> t(data.data(), {2, 3, 4}, {12, 4, 1});

    SUBCASE("Fix first dimension -> 2D matrix") {
        auto mat = t.slice(fix<0>(), all, all);
        // Should return matrix_view (rank reduced from 3 to 2)
        CHECK(mat.rows() == 3);
        CHECK(mat.cols() == 4);
        // (0,0,0) in original tensor
        CHECK(mat(0, 0) == 0.0f);
        CHECK(mat(2, 3) == 11.0f);
    }

    SUBCASE("Fix middle dimension -> 2D matrix") {
        auto mat = t.slice(all, fix<1>(), all);
        CHECK(mat.rows() == 2);
        CHECK(mat.cols() == 4);
        // (0,1,0) in original -> offset = 0*12 + 1*4 + 0 = 4
        CHECK(mat(0, 0) == 4.0f);
        // (1,1,3) in original -> offset = 1*12 + 1*4 + 3 = 19
        CHECK(mat(1, 3) == 19.0f);
    }

    SUBCASE("Fix last dimension -> 2D matrix") {
        auto mat = t.slice(all, all, fix<2>());
        CHECK(mat.rows() == 2);
        CHECK(mat.cols() == 3);
        // (0,0,2) in original -> offset = 0*12 + 0*4 + 2 = 2
        CHECK(mat(0, 0) == 2.0f);
        // (1,2,2) in original -> offset = 1*12 + 2*4 + 2 = 22
        CHECK(mat(1, 2) == 22.0f);
    }
}

TEST_CASE("Tensor slicing - dimensionality reduction to vector") {
    std::vector<float> data(24);
    for (std::size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 2x3x4 tensor
    tensor_view<float, 4, 3> t(data.data(), {2, 3, 4}, {12, 4, 1});

    SUBCASE("Fix first two dimensions -> 1D vector") {
        auto vec = t.slice(fix<0>(), fix<1>(), all);
        // Should return vector_view (rank reduced from 3 to 1)
        CHECK(vec.size() == 4);
        // (0,1,0) in original -> offset = 0*12 + 1*4 + 0 = 4
        CHECK(vec[0] == 4.0f);
        CHECK(vec[1] == 5.0f);
        CHECK(vec[3] == 7.0f);
    }

    SUBCASE("Fix first and last dimensions -> 1D vector") {
        auto vec = t.slice(fix<1>(), all, fix<2>());
        CHECK(vec.size() == 3);
        // (1,0,2) in original -> offset = 1*12 + 0*4 + 2 = 14
        CHECK(vec[0] == 14.0f);
        // (1,1,2) in original -> offset = 1*12 + 1*4 + 2 = 18
        CHECK(vec[1] == 18.0f);
        // (1,2,2) in original -> offset = 1*12 + 2*4 + 2 = 22
        CHECK(vec[2] == 22.0f);
    }

    SUBCASE("Fix last two dimensions -> 1D vector") {
        auto vec = t.slice(all, fix<0>(), fix<3>());
        CHECK(vec.size() == 2);
        // (0,0,3) in original -> offset = 0*12 + 0*4 + 3 = 3
        CHECK(vec[0] == 3.0f);
        // (1,0,3) in original -> offset = 1*12 + 0*4 + 3 = 15
        CHECK(vec[1] == 15.0f);
    }
}

TEST_CASE("Tensor slicing - dimensionality reduction to scalar") {
    std::vector<double> data(24);
    for (std::size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<double>(i * 10);
    }

    // 2x3x4 tensor
    tensor_view<double, 2, 3> t(data.data(), {2, 3, 4}, {12, 4, 1});

    SUBCASE("Fix all dimensions -> scalar (as 1x1x1 tensor)") {
        auto scalar_tensor = t.slice(fix<1>(), fix<2>(), fix<3>());
        // Currently returns 1x1x1 tensor_view
        CHECK(scalar_tensor.extent(0) == 1);
        CHECK(scalar_tensor.extent(1) == 1);
        CHECK(scalar_tensor.extent(2) == 1);
        // (1,2,3) in original -> offset = 1*12 + 2*4 + 3 = 23
        CHECK(scalar_tensor(0, 0, 0) == 230.0);
    }
}
