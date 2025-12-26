#include <iostream>
#include <optinum/optinum.hpp>

int main() {
    // Create from value
    optinum::Scalar<float> a(3.14f);
    optinum::Scalar<float> b(2.0f);

    // Arithmetic
    auto c = a + b;
    auto d = a * b;

    std::cout << "a = " << a.get() << "\n";
    std::cout << "b = " << b.get() << "\n";
    std::cout << "a + b = " << c.get() << "\n";
    std::cout << "a * b = " << d.get() << "\n";

    // Access underlying datapod type
    datapod::mat::scalar<float> &pod = a.pod();
    std::cout << "pod.value = " << pod.value << "\n";

    // Create from datapod
    datapod::mat::scalar<double> raw{42.0};
    optinum::Scalar<double> wrapped(raw);
    std::cout << "wrapped = " << wrapped.get() << "\n";

#if defined(SHORT_NAMESPACE)
    // Short namespace usage
    on::Scalar<int> x(10);
    on::Scalar<int> y(20);
    std::cout << "x + y = " << (x + y).get() << "\n";
#endif

    return 0;
}
