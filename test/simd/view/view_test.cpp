// =============================================================================
// test/simd/view/view_test.cpp
// Tests for SIMD view types (kernel, scalar_view, vector_view, matrix_view)
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/kernel.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/sse.hpp>
#include <optinum/simd/view/view.hpp>

namespace on = optinum;

// =============================================================================
// Kernel Tests
// =============================================================================

TEST_CASE("Kernel<T,W,1> - 1D memory layout") {
    using kernel_t = on::simd::Kernel<float, 4, 1>;

    alignas(16) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    kernel_t k(data, {8}, {1});

    SUBCASE("Metadata") {
        CHECK(k.extent(0) == 8);
        CHECK(k.stride(0) == 1);
        CHECK(k.linear_size() == 8);
        CHECK(k.num_packs() == 2); // 8 / 4 = 2
        CHECK(k.tail_size() == 4); // 8 % 4 = 0, so full pack
        CHECK(k.is_contiguous());
    }

    SUBCASE("Pack access") {
        auto p0 = k.load_pack(0);
        CHECK(p0[0] == doctest::Approx(1.0f));
        CHECK(p0[3] == doctest::Approx(4.0f));

        auto p1 = k.load_pack(1);
        CHECK(p1[0] == doctest::Approx(5.0f));
        CHECK(p1[3] == doctest::Approx(8.0f));
    }

    SUBCASE("Scalar access") {
        CHECK(k.at_linear(0) == doctest::Approx(1.0f));
        CHECK(k.at_linear(7) == doctest::Approx(8.0f));
    }
}

TEST_CASE("Kernel<T,W,2> - 2D column-major") {
    using kernel_t = on::simd::Kernel<float, 4, 2>;

    // 3x2 matrix, column-major
    alignas(16) float data[6] = {
        1, 2, 3, // column 0
        4, 5, 6  // column 1
    };
    kernel_t k(data, {3, 2}, {1, 3}); // rows=3, cols=2, stride[0]=1, stride[1]=3

    SUBCASE("Metadata") {
        CHECK(k.extent(0) == 3);
        CHECK(k.extent(1) == 2);
        CHECK(k.stride(0) == 1);
        CHECK(k.stride(1) == 3);
        CHECK(k.linear_size() == 6);
        CHECK(k.is_contiguous());
    }

    SUBCASE("Multi-dimensional indexing") {
        CHECK(k.at(0, 0) == doctest::Approx(1.0f));
        CHECK(k.at(1, 0) == doctest::Approx(2.0f));
        CHECK(k.at(2, 0) == doctest::Approx(3.0f));
        CHECK(k.at(0, 1) == doctest::Approx(4.0f));
        CHECK(k.at(1, 1) == doctest::Approx(5.0f));
        CHECK(k.at(2, 1) == doctest::Approx(6.0f));
    }
}

// =============================================================================
// scalar_view Tests
// =============================================================================

TEST_CASE("scalar_view<T,W>") {
    using view_t = on::simd::scalar_view<float, 4>;

    float value = 3.14f;
    view_t v(&value);

    SUBCASE("Access") {
        CHECK(v.get() == doctest::Approx(3.14f));
        CHECK(v.get_const() == doctest::Approx(3.14f));
    }

    SUBCASE("Assignment") {
        v = 2.71f;
        CHECK(value == doctest::Approx(2.71f));
    }

    SUBCASE("Conversion") {
        float &ref = v;
        CHECK(ref == doctest::Approx(3.14f));
    }
}

// =============================================================================
// vector_view Tests
// =============================================================================

TEST_CASE("vector_view<T,W> - Basic operations") {
    using view_t = on::simd::vector_view<float, 4>;

    alignas(16) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    view_t v(data, 8);

    SUBCASE("Size queries") {
        CHECK(v.size() == 8);
        CHECK(v.num_packs() == 2);
        CHECK(v.tail_size() == 4);
        CHECK(v.is_contiguous());
    }

    SUBCASE("Element access") {
        CHECK(v[0] == doctest::Approx(1.0f));
        CHECK(v[7] == doctest::Approx(8.0f));
        CHECK(v.at(3) == doctest::Approx(4.0f));
    }

    SUBCASE("Pack access") {
        auto p0 = v.load_pack(0);
        CHECK(p0[0] == doctest::Approx(1.0f));
        CHECK(p0[3] == doctest::Approx(4.0f));

        auto p1 = v.load_pack(1);
        CHECK(p1[0] == doctest::Approx(5.0f));
    }

    SUBCASE("Pack store") {
        on::simd::pack<float, 4> p(10.0f);
        v.store_pack(0, p);

        CHECK(data[0] == doctest::Approx(10.0f));
        CHECK(data[1] == doctest::Approx(10.0f));
        CHECK(data[2] == doctest::Approx(10.0f));
        CHECK(data[3] == doctest::Approx(10.0f));
        CHECK(data[4] == doctest::Approx(5.0f)); // not overwritten
    }
}

TEST_CASE("vector_view<T,W> - Tail handling") {
    using view_t = on::simd::vector_view<float, 4>;

    alignas(16) float data[5] = {1, 2, 3, 4, 5};
    view_t v(data, 5);

    SUBCASE("Tail size") {
        CHECK(v.num_packs() == 2); // (5 + 3) / 4 = 2
        CHECK(v.tail_size() == 1); // 5 % 4 = 1
    }

    SUBCASE("Tail load") {
        auto p1 = v.load_pack_tail(1); // last pack, only 1 valid element
        CHECK(p1[0] == doctest::Approx(5.0f));
        // Other elements are zero from maskload
    }

    SUBCASE("Tail store") {
        on::simd::pack<float, 4> p(99.0f);
        v.store_pack_tail(1, p); // only stores to index 4

        CHECK(data[4] == doctest::Approx(99.0f));
        // Should not write beyond array
    }
}

TEST_CASE("vector_view<T,W> - Subview") {
    using view_t = on::simd::vector_view<float, 4>;

    alignas(16) float data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    view_t v(data, 10);

    auto sub = v.subview(2, 5); // elements [2..6]
    CHECK(sub.size() == 5);
    CHECK(sub[0] == doctest::Approx(2.0f));
    CHECK(sub[4] == doctest::Approx(6.0f));
}

// =============================================================================
// matrix_view Tests
// =============================================================================

TEST_CASE("matrix_view<T,W> - Column-major layout") {
    using view_t = on::simd::matrix_view<float, 4>;

    // 3x2 matrix, column-major storage
    alignas(16) float data[6] = {
        1, 2, 3, // column 0
        4, 5, 6  // column 1
    };
    view_t m(data, 3, 2);

    SUBCASE("Size queries") {
        CHECK(m.rows() == 3);
        CHECK(m.cols() == 2);
        CHECK(m.size() == 6);
        CHECK(m.is_contiguous());
    }

    SUBCASE("Element access") {
        CHECK(m(0, 0) == doctest::Approx(1.0f));
        CHECK(m(1, 0) == doctest::Approx(2.0f));
        CHECK(m(2, 0) == doctest::Approx(3.0f));
        CHECK(m(0, 1) == doctest::Approx(4.0f));
        CHECK(m(1, 1) == doctest::Approx(5.0f));
        CHECK(m(2, 1) == doctest::Approx(6.0f));
    }

    SUBCASE("Column view") {
        auto col0 = m.col(0);
        CHECK(col0.size() == 3);
        CHECK(col0[0] == doctest::Approx(1.0f));
        CHECK(col0[1] == doctest::Approx(2.0f));
        CHECK(col0[2] == doctest::Approx(3.0f));

        auto col1 = m.col(1);
        CHECK(col1[0] == doctest::Approx(4.0f));
    }

    SUBCASE("Row view") {
        auto row0 = m.row(0);
        CHECK(row0.size() == 2);
        CHECK(row0[0] == doctest::Approx(1.0f));
        CHECK(row0[1] == doctest::Approx(4.0f));
    }
}

TEST_CASE("matrix_view<T,W> - Pack operations") {
    using view_t = on::simd::matrix_view<float, 4>;

    alignas(16) float data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    view_t m(data, 4, 3); // 4x3 matrix

    SUBCASE("Pack load") {
        auto p0 = m.load_pack(0); // first 4 elements
        CHECK(p0[0] == doctest::Approx(1.0f));
        CHECK(p0[3] == doctest::Approx(4.0f));
    }

    SUBCASE("Pack store") {
        on::simd::pack<float, 4> p(42.0f);
        m.store_pack(0, p);

        CHECK(data[0] == doctest::Approx(42.0f));
        CHECK(data[1] == doctest::Approx(42.0f));
        CHECK(data[2] == doctest::Approx(42.0f));
        CHECK(data[3] == doctest::Approx(42.0f));
        CHECK(data[4] == doctest::Approx(5.0f)); // not touched
    }
}

TEST_CASE("matrix_view<T,W> - Subview (block)") {
    using view_t = on::simd::matrix_view<float, 4>;

    // 4x4 matrix
    alignas(16) float data[16];
    for (int i = 0; i < 16; ++i)
        data[i] = static_cast<float>(i);
    view_t m(data, 4, 4);

    // Extract 2x2 block starting at (1,1)
    auto block = m.subview(1, 1, 2, 2);
    CHECK(block.rows() == 2);
    CHECK(block.cols() == 2);
    CHECK(block(0, 0) == doctest::Approx(5.0f));  // data[1 + 1*4] = data[5]
    CHECK(block(1, 0) == doctest::Approx(6.0f));  // data[2 + 1*4] = data[6]
    CHECK(block(0, 1) == doctest::Approx(9.0f));  // data[1 + 2*4] = data[9]
    CHECK(block(1, 1) == doctest::Approx(10.0f)); // data[2 + 2*4] = data[10]
}

// =============================================================================
// tensor_view Tests
// =============================================================================

TEST_CASE("tensor_view<T,W,3> - 3D array") {
    using view_t = on::simd::tensor_view<float, 4, 3>;

    // 2x3x2 tensor (contiguous)
    alignas(16) float data[12];
    for (int i = 0; i < 12; ++i)
        data[i] = static_cast<float>(i);

    view_t t(data, {2, 3, 2}, {1, 2, 6});

    SUBCASE("Size queries") {
        CHECK(t.extent(0) == 2);
        CHECK(t.extent(1) == 3);
        CHECK(t.extent(2) == 2);
        CHECK(t.size() == 12);
    }

    SUBCASE("Multi-dimensional indexing") {
        CHECK(t(0, 0, 0) == doctest::Approx(0.0f));
        CHECK(t(1, 0, 0) == doctest::Approx(1.0f));
        CHECK(t(0, 1, 0) == doctest::Approx(2.0f));
        CHECK(t(0, 0, 1) == doctest::Approx(6.0f));
    }
}
