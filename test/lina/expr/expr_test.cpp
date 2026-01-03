#include <doctest/doctest.h>
#include <optinum/lina/expr/expr.hpp>

namespace expr = optinum::lina::expr;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

TEST_CASE("lina::expr vector expression eval") {
    dp::mat::Vector<float, 3> a;
    a[0] = 1.f;
    a[1] = 2.f;
    a[2] = 3.f;
    dp::mat::Vector<float, 3> b;
    b[0] = 4.f;
    b[1] = 5.f;
    b[2] = 6.f;

    // Create views that outlive the expression
    Vector<float, 3> a_view(a);
    Vector<float, 3> b_view(b);
    const auto e = expr::add(expr::ref(a_view), expr::scale(expr::ref(b_view), 2.f)); // a + 2*b
    const auto r = expr::eval(e);
    CHECK(r[0] == doctest::Approx(9.f));
    CHECK(r[1] == doctest::Approx(12.f));
    CHECK(r[2] == doctest::Approx(15.f));
}

TEST_CASE("lina::expr matrix expression eval") {
    dp::mat::Matrix<int, 2, 2> a;
    a(0, 0) = 1;
    a(1, 0) = 3;
    a(0, 1) = 2;
    a(1, 1) = 4;

    dp::mat::Matrix<int, 2, 2> b;
    b(0, 0) = 5;
    b(1, 0) = 7;
    b(0, 1) = 6;
    b(1, 1) = 8;

    // Create views that outlive the expression
    Matrix<int, 2, 2> a_view(a);
    Matrix<int, 2, 2> b_view(b);
    const auto e = expr::add(expr::scale(expr::ref(a_view), 2), expr::ref(b_view)); // 2*a + b
    const auto r = expr::eval(e);
    CHECK(r(0, 0) == 7);
    CHECK(r(1, 1) == 16);
}
