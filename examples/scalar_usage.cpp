#include <iostream>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace simd = optinum::simd;

int main() {
    // Create scalars using dp::mat::scalar (owning storage)
    dp::mat::scalar<float> a{3.14f};
    dp::mat::scalar<float> b{2.0f};

    // Create non-owning views using simd::Scalar
    simd::Scalar<float> view_a(a);
    simd::Scalar<float> view_b(b);

    // Arithmetic using views (implicit conversion to T)
    dp::mat::scalar<float> c{static_cast<float>(view_a) + static_cast<float>(view_b)};
    dp::mat::scalar<float> d{static_cast<float>(view_a) * static_cast<float>(view_b)};

    std::cout << "a = " << view_a.get() << "\n";
    std::cout << "b = " << view_b.get() << "\n";
    std::cout << "a + b = " << c.value << "\n";
    std::cout << "a * b = " << d.value << "\n";

    // Direct access via view
    std::cout << "view_a.get() = " << view_a.get() << "\n";

    // Modify via view
    view_a = 5.0f;
    std::cout << "After view_a = 5.0f: a.value = " << a.value << "\n";

    // Compound assignment via view
    view_a += 2.0f;
    std::cout << "After view_a += 2.0f: a.value = " << a.value << "\n";

    // View from raw pointer
    float raw_val = 42.0f;
    simd::Scalar<float> view_raw(&raw_val);
    std::cout << "view_raw.get() = " << view_raw.get() << "\n";

    // Create double scalar with initializer
    dp::mat::scalar<double> raw{42.0};
    simd::Scalar<double> view_double(raw);
    std::cout << "raw = " << view_double.get() << "\n";

    // Comparison operators
    dp::mat::scalar<float> x{10.0f};
    dp::mat::scalar<float> y{20.0f};
    simd::Scalar<float> view_x(x);
    simd::Scalar<float> view_y(y);

    std::cout << "x < y: " << (view_x < view_y ? "true" : "false") << "\n";
    std::cout << "x == y: " << (view_x == view_y ? "true" : "false") << "\n";

    return 0;
}
