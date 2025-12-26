// =============================================================================
// examples/slicing_usage.cpp
// Demonstrates view slicing with seq(), fseq<>(), all, fix<N>
// =============================================================================

#include <iostream>
#include <optinum/simd/simd.hpp>
#include <vector>

using namespace optinum::simd;

int main() {
    // Create test data
    std::vector<float> data(20);
    for (int i = 0; i < 20; i++) {
        data[i] = static_cast<float>(i);
    }

    // Create a SIMD view
    vector_view<float, 4> v(data.data(), 20);

    std::cout << "=== View Slicing Examples ===\n\n";

    // Example 1: Runtime slicing with seq()
    std::cout << "1. Runtime slicing: v.slice(seq(5, 15))\n";
    auto s1 = v.slice(seq(5, 15)); // Elements 5..14
    std::cout << "   Size: " << s1.size() << " elements\n";
    std::cout << "   First 5: ";
    for (int i = 0; i < 5; i++) {
        std::cout << s1[i] << " ";
    }
    std::cout << "\n\n";

    // Example 2: Stride slicing
    std::cout << "2. Strided slicing: v.slice(seq(0, 20, 3))\n";
    auto s2 = v.slice(seq(0, 20, 3)); // Every 3rd element
    std::cout << "   Size: " << s2.size() << " elements\n";
    std::cout << "   Values: ";
    for (std::size_t i = 0; i < s2.size(); i++) {
        std::cout << s2[i] << " ";
    }
    std::cout << "\n\n";

    // Example 3: Compile-time slicing with fseq<>
    std::cout << "3. Compile-time slicing: v.slice(fseq<10, 15>())\n";
    auto s3 = v.slice(fseq<10, 15>()); // Elements 10..14 (known at compile time)
    std::cout << "   Size: " << s3.size() << " elements\n";
    std::cout << "   Values: ";
    for (std::size_t i = 0; i < s3.size(); i++) {
        std::cout << s3[i] << " ";
    }
    std::cout << "\n\n";

    // Example 4: Select all elements
    std::cout << "4. Select all: v.slice(all)\n";
    auto s4 = v.slice(all);
    std::cout << "   Size: " << s4.size() << " elements\n";
    std::cout << "   First 10: ";
    for (int i = 0; i < 10; i++) {
        std::cout << s4[i] << " ";
    }
    std::cout << "...\n\n";

    // Example 5: Single element selection with fix<N>
    std::cout << "5. Single element: v.slice(fix<7>())\n";
    auto s5 = v.slice(fix<7>());
    std::cout << "   Size: " << s5.size() << " element\n";
    std::cout << "   Value: " << s5[0] << "\n\n";

    // Example 6: Chained slicing
    std::cout << "6. Chained slicing: v.slice(seq(5, 15)).slice(seq(2, 8))\n";
    auto s6 = v.slice(seq(5, 15)).slice(seq(2, 8)); // 7..12
    std::cout << "   Size: " << s6.size() << " elements\n";
    std::cout << "   Values: ";
    for (std::size_t i = 0; i < s6.size(); i++) {
        std::cout << s6[i] << " ";
    }
    std::cout << "\n\n";

    // Example 7: SIMD operations on sliced views
    std::cout << "7. SIMD operations on slices\n";
    auto slice = v.slice(seq(0, 12)); // First 12 elements
    std::cout << "   Computing sum of first 12 elements using SIMD...\n";
    float sum = 0.0f;
    for (std::size_t i = 0; i < slice.num_packs(); i++) {
        auto p = slice.load_pack(i);
        for (std::size_t j = 0; j < 4; j++) {
            sum += p[j];
        }
    }
    std::cout << "   Sum: " << sum << " (expected: " << (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11) << ")\n";

    std::cout << "\n=== Matrix Slicing Examples ===\n\n";

    // Create matrix data (4x5 column-major)
    std::vector<float> mat_data(20);
    for (int i = 0; i < 20; i++) {
        mat_data[i] = static_cast<float>(i);
    }
    matrix_view<float, 4> m(mat_data.data(), 4, 5);

    std::cout << "Original 4x5 matrix (column-major):\n";
    for (std::size_t r = 0; r < 4; r++) {
        std::cout << "   ";
        for (std::size_t c = 0; c < 5; c++) {
            std::cout << m(r, c) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Example 8: Slice rows
    std::cout << "8. Row slicing: m.slice(seq(1, 3), all)\n";
    auto m1 = m.slice(seq(1, 3), all);
    std::cout << "   Result: " << m1.rows() << "x" << m1.cols() << " matrix\n";
    for (std::size_t r = 0; r < m1.rows(); r++) {
        std::cout << "   ";
        for (std::size_t c = 0; c < m1.cols(); c++) {
            std::cout << m1(r, c) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Example 9: Slice columns
    std::cout << "9. Column slicing: m.slice(all, seq(2, 5))\n";
    auto m2 = m.slice(all, seq(2, 5));
    std::cout << "   Result: " << m2.rows() << "x" << m2.cols() << " matrix\n";
    for (std::size_t r = 0; r < m2.rows(); r++) {
        std::cout << "   ";
        for (std::size_t c = 0; c < m2.cols(); c++) {
            std::cout << m2(r, c) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Example 10: Extract a single row as vector
    std::cout << "10. Single row extraction: m.slice(fix<2>(), all)\n";
    auto row = m.slice(fix<2>(), all);
    std::cout << "   Result: vector of size " << row.size() << "\n";
    std::cout << "   Values: ";
    for (std::size_t i = 0; i < row.size(); i++) {
        std::cout << row[i] << " ";
    }
    std::cout << "\n\n";

    // Example 11: Extract a single column as vector
    std::cout << "11. Single column extraction: m.slice(all, fix<3>())\n";
    auto col = m.slice(all, fix<3>());
    std::cout << "   Result: vector of size " << col.size() << "\n";
    std::cout << "   Values: ";
    for (std::size_t i = 0; i < col.size(); i++) {
        std::cout << col[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "\n=== Tensor Slicing Examples ===\n\n";

    // Create 3D tensor data (2x3x4)
    std::vector<float> tensor_data(24);
    for (int i = 0; i < 24; i++) {
        tensor_data[i] = static_cast<float>(i);
    }
    tensor_view<float, 4, 3> t(tensor_data.data(), {2, 3, 4}, {12, 4, 1});

    std::cout << "Original 2x3x4 tensor:\n";
    for (std::size_t d0 = 0; d0 < 2; d0++) {
        std::cout << "   Slice " << d0 << ":\n";
        for (std::size_t d1 = 0; d1 < 3; d1++) {
            std::cout << "      ";
            for (std::size_t d2 = 0; d2 < 4; d2++) {
                std::cout << t(d0, d1, d2) << " ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    // Example 12: Slice first dimension
    std::cout << "12. First dimension slicing: t.slice(seq(0, 1), all, all)\n";
    auto t1 = t.slice(seq(0, 1), all, all);
    std::cout << "   Result: " << t1.extent(0) << "x" << t1.extent(1) << "x" << t1.extent(2) << " tensor\n\n";

    // Example 13: Multi-dimensional slicing
    std::cout << "13. Multi-dim slicing: t.slice(all, seq(1, 3), seq(0, 3))\n";
    auto t2 = t.slice(all, seq(1, 3), seq(0, 3));
    std::cout << "   Result: " << t2.extent(0) << "x" << t2.extent(1) << "x" << t2.extent(2) << " tensor\n";
    std::cout << "   Sample values: t2(0,0,0)=" << t2(0, 0, 0) << ", t2(1,1,2)=" << t2(1, 1, 2) << "\n\n";

    // Example 14: Strided tensor slicing
    std::cout << "14. Strided slicing: t.slice(all, all, seq(0, 4, 2))\n";
    auto t3 = t.slice(all, all, seq(0, 4, 2));
    std::cout << "   Result: " << t3.extent(0) << "x" << t3.extent(1) << "x" << t3.extent(2) << " tensor\n";
    std::cout << "   (every 2nd element in last dimension)\n\n";

    std::cout << "\n=== All Slicing Examples Complete ===\n";

    return 0;
}
