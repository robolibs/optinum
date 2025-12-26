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

    std::cout << "\n=== View Slicing Complete ===\n";

    return 0;
}
