// =============================================================================
// examples/factory_usage.cpp
// Demonstrates factory functions and utility methods for Vector/Matrix
// =============================================================================

#include <iomanip>
#include <iostream>
#include <optinum/simd/simd.hpp>

using namespace optinum::simd;

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
    // Part 1: Static Factories
    // =========================================================================
    std::cout << "PART 1: Static Factory Methods\n";
    std::cout << std::string(50, '-') << "\n\n";

    // zeros() - Create vector/matrix filled with zeros
    std::cout << "1. zeros() - Create filled with zeros\n";
    auto v_zeros = Vector<float, 5>::zeros();
    print_vector("   Vector<float, 5>::zeros()", v_zeros);

    auto m_zeros = Matrix<double, 2, 3>::zeros();
    print_matrix("   Matrix<double, 2, 3>::zeros()", m_zeros);
    std::cout << "\n";

    // ones() - Create vector/matrix filled with ones
    std::cout << "2. ones() - Create filled with ones\n";
    auto v_ones = Vector<int, 4>::ones();
    print_vector("   Vector<int, 4>::ones()", v_ones);

    auto m_ones = Matrix<float, 3, 2>::ones();
    print_matrix("   Matrix<float, 3, 2>::ones()", m_ones);
    std::cout << "\n";

    // arange() - Create with sequential values
    std::cout << "3. arange() - Create with sequential values\n";

    auto v_arange1 = Vector<int, 6>::arange();
    print_vector("   Vector<int, 6>::arange()", v_arange1);

    auto v_arange2 = Vector<float, 5>::arange(10.0f);
    print_vector("   Vector<float, 5>::arange(10.0f)", v_arange2);

    auto v_arange3 = Vector<double, 4>::arange(0.0, 0.5);
    print_vector("   Vector<double, 4>::arange(0.0, 0.5)", v_arange3);

    auto m_arange = Matrix<int, 2, 4>::arange();
    print_matrix("   Matrix<int, 2, 4>::arange()", m_arange);
    std::cout << "\n";

    // =========================================================================
    // Part 2: Instance Methods - fill()
    // =========================================================================
    std::cout << "PART 2: fill() - Fill existing container\n";
    std::cout << std::string(50, '-') << "\n\n";

    Vector<float, 4> v;
    v.fill(3.14f);
    print_vector("v.fill(3.14f)", v);

    Matrix<double, 2, 2> m;
    m.fill(2.71);
    print_matrix("m.fill(2.71)", m);
    std::cout << "\n";

    // =========================================================================
    // Part 3: iota() - Sequential filling
    // =========================================================================
    std::cout << "PART 3: iota() - Fill with sequential values\n";
    std::cout << std::string(50, '-') << "\n\n";

    Vector<int, 5> v1;
    v1.iota();
    print_vector("v.iota() - defaults to 0, 1, 2, ...", v1);

    Vector<float, 4> v2;
    v2.iota(10.0f);
    print_vector("v.iota(10.0f) - start from 10", v2);

    Vector<double, 5> v3;
    v3.iota(0.0, 2.5);
    print_vector("v.iota(0.0, 2.5) - step by 2.5", v3);

    Matrix<int, 3, 3> m1;
    m1.iota();
    print_matrix("m.iota() - fill matrix sequentially", m1);
    std::cout << "\n";

    // =========================================================================
    // Part 4: reverse() - Reverse element order
    // =========================================================================
    std::cout << "PART 4: reverse() - Reverse element order\n";
    std::cout << std::string(50, '-') << "\n\n";

    Vector<int, 6> v_rev;
    v_rev.iota();
    print_vector("Before reverse", v_rev);
    v_rev.reverse();
    print_vector("After reverse", v_rev);

    Matrix<int, 2, 3> m_rev;
    m_rev.iota();
    print_matrix("Before reverse", m_rev);
    m_rev.reverse();
    print_matrix("After reverse (linear order)", m_rev);
    std::cout << "\n";

    // =========================================================================
    // Part 5: random() - Random values
    // =========================================================================
    std::cout << "PART 5: random() - Fill with random values\n";
    std::cout << std::string(50, '-') << "\n\n";

    Vector<float, 8> v_rand;
    v_rand.random();
    print_vector("v.random() - uniform [0, 1)", v_rand);

    Matrix<double, 3, 3> m_rand;
    m_rand.random();
    print_matrix("m.random() - uniform [0, 1)", m_rand);
    std::cout << "\n";

    // =========================================================================
    // Part 6: randint() - Random integers
    // =========================================================================
    std::cout << "PART 6: randint() - Fill with random integers\n";
    std::cout << std::string(50, '-') << "\n\n";

    Vector<int, 10> v_randint;
    v_randint.randint(1, 10);
    print_vector("v.randint(1, 10) - integers in [1, 10]", v_randint);

    Matrix<int, 3, 4> m_randint;
    m_randint.randint(0, 100);
    print_matrix("m.randint(0, 100) - integers in [0, 100]", m_randint);
    std::cout << "\n";

    // =========================================================================
    // Part 7: Method chaining
    // =========================================================================
    std::cout << "PART 7: Method Chaining\n";
    std::cout << std::string(50, '-') << "\n\n";

    Vector<float, 5> v_chain;
    v_chain.iota(1.0f, 2.0f).reverse();
    print_vector("v.iota(1.0f, 2.0f).reverse()", v_chain);

    Matrix<int, 2, 3> m_chain;
    m_chain.fill(0).iota(100);
    print_matrix("m.fill(0).iota(100)", m_chain);
    std::cout << "\n";

    // =========================================================================
    // Part 8: Practical Examples
    // =========================================================================
    std::cout << "PART 8: Practical Examples\n";
    std::cout << std::string(50, '-') << "\n\n";

    // Create coordinate grid
    std::cout << "Example 1: Create coordinate grid\n";
    auto x_coords = Vector<double, 5>::arange(0.0, 0.25);
    print_vector("   x coordinates (0, 0.25, 0.5, ...)", x_coords);

    // Initialize weights for ML
    std::cout << "\nExample 2: Initialize neural network weights\n";
    Matrix<float, 3, 4> weights;
    weights.random();
    std::cout << "   Weights initialized with random values:\n";
    print_matrix("   ", weights);

    // Create test data
    std::cout << "\nExample 3: Create test vector\n";
    auto test_data = Vector<int, 10>::arange(1);
    print_vector("   Test indices (1, 2, 3, ...)", test_data);

    std::cout << "\n=== END ===\n";

    return 0;
}
