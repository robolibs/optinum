#include <iostream>
#include <optinum/lina/lina.hpp>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace lina = optinum::lina;
namespace simd = optinum::simd;

int main() {
    // Create vectors using dp::mat::vector
    dp::mat::Vector<float, 3> a{1.0f, 2.0f, 3.0f};

    dp::mat::Vector<float, 3> b;
    b.fill(2.0f);

    // Element-wise operations using SIMD backend
    dp::mat::Vector<float, 3> c;
    simd::backend::add<float, 3>(c.data(), a.data(), b.data());

    dp::mat::Vector<float, 3> d;
    simd::backend::mul<float, 3>(d.data(), a.data(), b.data());

    std::cout << "a = [" << a[0] << ", " << a[1] << ", " << a[2] << "]\n";
    std::cout << "b = [" << b[0] << ", " << b[1] << ", " << b[2] << "]\n";
    std::cout << "a + b = [" << c[0] << ", " << c[1] << ", " << c[2] << "]\n";
    std::cout << "a * b = [" << d[0] << ", " << d[1] << ", " << d[2] << "]\n";

    // Scalar operations using SIMD backend
    dp::mat::Vector<float, 3> scaled;
    simd::backend::mul_scalar<float, 3>(scaled.data(), a.data(), 3.0f);
    std::cout << "a * 3 = [" << scaled[0] << ", " << scaled[1] << ", " << scaled[2] << "]\n";

    // Dot product and norm using SIMD backend
    std::cout << "dot(a, b) = " << simd::backend::dot<float, 3>(a.data(), b.data()) << "\n";
    std::cout << "sum(a) = " << simd::backend::reduce_sum<float, 3>(a.data()) << "\n";
    std::cout << "norm(a) = " << simd::backend::norm_l2<float, 3>(a.data()) << "\n";

    // Normalized using SIMD backend
    dp::mat::Vector<float, 3> n;
    simd::backend::normalize<float, 3>(n.data(), a.data());
    std::cout << "normalized(a) = [" << n[0] << ", " << n[1] << ", " << n[2] << "]\n";
    std::cout << "norm(normalized(a)) = " << simd::backend::norm_l2<float, 3>(n.data()) << "\n";

    // Direct access to dp::mat::vector
    std::cout << "a[0] = " << a[0] << "\n";

    // Create vector with initializer list
    dp::mat::Vector<double, 4> raw{1.0, 2.0, 3.0, 4.0};
    std::cout << "raw = [" << raw[0] << ", " << raw[1] << ", " << raw[2] << ", " << raw[3] << "]\n";

    // Iteration
    std::cout << "iterating a: ";
    for (auto val : a) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    return 0;
}
