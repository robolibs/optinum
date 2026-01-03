// =============================================================================
// examples/factory_usage.cpp
// Demonstrates factory functions and utility methods for Vector/Matrix
// =============================================================================

#include <iomanip>
#include <iostream>
#include <optinum/optinum.hpp>

namespace dp = datapod;
using namespace optinum;

void print_vector(const char *label, auto &v) {
    std::cout << label << ": [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i < v.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

void print_matrix(const char *label, auto &m) {
    std::cout << label << ":\n";
    for (std::size_t r = 0; r < m.rows(); ++r) {
        std::cout << "  [";
        for (std::size_t c = 0; c < m.cols(); ++c) {
            std::cout << std::setw(6) << std::setprecision(3) << m(r, c);
            if (c < m.cols() - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

int main() {
    std::cout << "=== FACTORY FUNCTIONS & UTILITY METHODS ===\n\n";

    // =========================================================================
    // Part 1: Static Factories (now free functions returning dp::mat types)
    // =========================================================================
    std::cout << "PART 1: Static Factory Methods\n";
    std::cout << std::string(50, '-') << "\n\n";

    // zeros() - Create vector/matrix filled with zeros
    std::cout << "1. zeros() - Create filled with zeros\n";
    auto v_zeros = simd::zeros<float, 5>();
    print_vector("   simd::zeros<float, 5>()", v_zeros);

    auto m_zeros = simd::zeros_matrix<double, 2, 3>();
    print_matrix("   simd::zeros_matrix<double, 2, 3>()", m_zeros);
    std::cout << "\n";

    // ones() - Create vector/matrix filled with ones
    std::cout << "2. ones() - Create filled with ones\n";
    auto v_ones = simd::ones<int, 4>();
    print_vector("   simd::ones<int, 4>()", v_ones);

    auto m_ones = simd::ones_matrix<float, 3, 2>();
    print_matrix("   simd::ones_matrix<float, 3, 2>()", m_ones);
    std::cout << "\n";

    // arange() - Create with sequential values
    std::cout << "3. arange() - Create with sequential values\n";

    auto v_arange1 = simd::arange<int, 6>();
    print_vector("   simd::arange<int, 6>()", v_arange1);

    auto v_arange2 = simd::arange<float, 5>(10.0f);
    print_vector("   simd::arange<float, 5>(10.0f)", v_arange2);

    auto v_arange3 = simd::arange<double, 4>(0.0, 0.5);
    print_vector("   simd::arange<double, 4>(0.0, 0.5)", v_arange3);

    // For matrix arange, we use iota on a zero-initialized matrix
    dp::mat::Matrix<int, 2, 4> m_arange;
    simd::Matrix<int, 2, 4>(m_arange).iota();
    print_matrix("   Matrix<int, 2, 4> with iota()", m_arange);
    std::cout << "\n";

    // =========================================================================
    // Part 2: Instance Methods - fill()
    // Using dp::mat types with simd::Vector views
    // =========================================================================
    std::cout << "PART 2: fill() - Fill existing container\n";
    std::cout << std::string(50, '-') << "\n\n";

    dp::mat::Vector<float, 4> v_storage;
    simd::Vector<float, 4> v(v_storage);
    v.fill(3.14f);
    print_vector("v.fill(3.14f)", v_storage);

    dp::mat::Matrix<double, 2, 2> m_storage;
    simd::Matrix<double, 2, 2> m(m_storage);
    m.fill(2.71);
    print_matrix("m.fill(2.71)", m_storage);
    std::cout << "\n";

    // =========================================================================
    // Part 3: iota() - Sequential filling
    // =========================================================================
    std::cout << "PART 3: iota() - Fill with sequential values\n";
    std::cout << std::string(50, '-') << "\n\n";

    dp::mat::Vector<int, 5> v1_storage;
    simd::Vector<int, 5> v1(v1_storage);
    v1.iota();
    print_vector("v.iota() - defaults to 0, 1, 2, ...", v1_storage);

    dp::mat::Vector<float, 4> v2_storage;
    simd::Vector<float, 4> v2(v2_storage);
    v2.iota(10.0f);
    print_vector("v.iota(10.0f) - start from 10", v2_storage);

    dp::mat::Vector<double, 5> v3_storage;
    simd::Vector<double, 5> v3(v3_storage);
    v3.iota(0.0, 2.5);
    print_vector("v.iota(0.0, 2.5) - step by 2.5", v3_storage);

    dp::mat::Matrix<int, 3, 3> m1_storage;
    simd::Matrix<int, 3, 3> m1(m1_storage);
    m1.iota();
    print_matrix("m.iota() - fill matrix sequentially", m1_storage);
    std::cout << "\n";

    // =========================================================================
    // Part 4: reverse() - Reverse element order
    // =========================================================================
    std::cout << "PART 4: reverse() - Reverse element order\n";
    std::cout << std::string(50, '-') << "\n\n";

    dp::mat::Vector<int, 6> v_rev_storage;
    simd::Vector<int, 6> v_rev(v_rev_storage);
    v_rev.iota();
    print_vector("Before reverse", v_rev_storage);
    v_rev.reverse();
    print_vector("After reverse", v_rev_storage);

    dp::mat::Matrix<int, 2, 3> m_rev_storage;
    simd::Matrix<int, 2, 3> m_rev(m_rev_storage);
    m_rev.iota();
    print_matrix("Before reverse", m_rev_storage);
    m_rev.reverse();
    print_matrix("After reverse (linear order)", m_rev_storage);
    std::cout << "\n";

    // =========================================================================
    // Part 5: random() - Random values
    // =========================================================================
    std::cout << "PART 5: random() - Fill with random values\n";
    std::cout << std::string(50, '-') << "\n\n";

    auto v_rand_storage = simd::random<float, 8>();
    print_vector("simd::random<float, 8>() - uniform [0, 1)", v_rand_storage);

    // For matrices, we can use the algo random fill functions
    std::cout << "   (Use simd::random_uniform_fill for matrices)\n";
    std::cout << "\n";

    // =========================================================================
    // Part 7: Method chaining
    // =========================================================================
    std::cout << "PART 7: Method Chaining\n";
    std::cout << std::string(50, '-') << "\n\n";

    dp::mat::Vector<float, 5> v_chain_storage;
    simd::Vector<float, 5> v_chain(v_chain_storage);
    v_chain.iota(1.0f, 2.0f).reverse();
    print_vector("v.iota(1.0f, 2.0f).reverse()", v_chain_storage);

    dp::mat::Matrix<int, 2, 3> m_chain_storage;
    simd::Matrix<int, 2, 3> m_chain(m_chain_storage);
    m_chain.fill(0).iota(100);
    print_matrix("m.fill(0).iota(100)", m_chain_storage);
    std::cout << "\n";

    // =========================================================================
    // Part 8: Practical Examples
    // =========================================================================
    std::cout << "PART 8: Practical Examples\n";
    std::cout << std::string(50, '-') << "\n\n";

    // Create coordinate grid
    std::cout << "Example 1: Create coordinate grid\n";
    auto x_coords = simd::arange<double, 5>(0.0, 0.25);
    print_vector("   x coordinates (0, 0.25, 0.5, ...)", x_coords);

    // Initialize weights for ML
    std::cout << "\nExample 2: Initialize neural network weights\n";
    auto weights_storage = simd::random<float, 12>(); // 3x4 = 12 elements
    dp::mat::Matrix<float, 3, 4> weights_matrix;
    for (std::size_t i = 0; i < 12; ++i) {
        weights_matrix[i] = weights_storage[i];
    }
    std::cout << "   Weights initialized with random values:\n";
    print_matrix("   ", weights_matrix);

    // Create test data
    std::cout << "\nExample 3: Create test vector\n";
    auto test_data = simd::arange<int, 10>(1);
    print_vector("   Test indices (1, 2, 3, ...)", test_data);

    std::cout << "\n=== END ===\n";

    return 0;
}
