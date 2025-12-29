#include <datapod/matrix/scalar.hpp>
#include <iostream>

namespace dp = datapod;

int main() {
    // Create scalars using dp::mat::scalar
    dp::mat::scalar<float> a{3.14f};
    dp::mat::scalar<float> b{2.0f};

    // Arithmetic using value member
    dp::mat::scalar<float> c{a.value + b.value};
    dp::mat::scalar<float> d{a.value * b.value};

    std::cout << "a = " << a.value << "\n";
    std::cout << "b = " << b.value << "\n";
    std::cout << "a + b = " << c.value << "\n";
    std::cout << "a * b = " << d.value << "\n";

    // Direct access to value
    std::cout << "a.value = " << a.value << "\n";

    // Create with initializer
    dp::mat::scalar<double> raw{42.0};
    std::cout << "raw = " << raw.value << "\n";

    return 0;
}
