#include <iostream>
#include <optinum/optinum.hpp>

int main() {
    // Create tensors
    optinum::simd::Vector<float, 3> a;
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    optinum::simd::Vector<float, 3> b;
    b.fill(2.0f);

    // Element-wise operations
    auto c = a + b;
    auto d = a * b;

    std::cout << "a = [" << a[0] << ", " << a[1] << ", " << a[2] << "]\n";
    std::cout << "b = [" << b[0] << ", " << b[1] << ", " << b[2] << "]\n";
    std::cout << "a + b = [" << c[0] << ", " << c[1] << ", " << c[2] << "]\n";
    std::cout << "a * b = [" << d[0] << ", " << d[1] << ", " << d[2] << "]\n";

    // Scalar operations
    auto scaled = a * 3.0f;
    std::cout << "a * 3 = [" << scaled[0] << ", " << scaled[1] << ", " << scaled[2] << "]\n";

    // Dot product and norm
    std::cout << "dot(a, b) = " << optinum::simd::dot(a, b) << "\n";
    std::cout << "sum(a) = " << optinum::simd::sum(a) << "\n";
    std::cout << "norm(a) = " << optinum::simd::norm(a) << "\n";

    // Normalized
    auto n = optinum::simd::normalized(a);
    std::cout << "normalized(a) = [" << n[0] << ", " << n[1] << ", " << n[2] << "]\n";
    std::cout << "norm(normalized(a)) = " << optinum::simd::norm(n) << "\n";

    // Access underlying datapod type
    datapod::mat::vector<float, 3> &pod = a.pod();
    std::cout << "pod[0] = " << pod[0] << "\n";

    // Create from datapod
    datapod::mat::vector<double, 4> raw{1.0, 2.0, 3.0, 4.0};
    optinum::simd::Vector<double, 4> wrapped(raw);
    std::cout << "wrapped = [" << wrapped[0] << ", " << wrapped[1] << ", " << wrapped[2] << ", " << wrapped[3] << "]\n";

    // Iteration
    std::cout << "iterating a: ";
    for (auto val : a) {
        std::cout << val << " ";
    }
    std::cout << "\n";

#if defined(SHORT_NAMESPACE)
    // Short namespace
    on::simd::Vector3f v;
    v[0] = 1.0f;
    v[1] = 0.0f;
    v[2] = 0.0f;
    std::cout << "on::simd::Vector3f norm = " << on::simd::norm(v) << "\n";
#endif

    return 0;
}
