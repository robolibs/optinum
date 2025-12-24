#include <doctest/doctest.h>
#include <optinum/lina/expr/expr.hpp>

using optinum::lina::expr::add;
using optinum::lina::expr::eval;
using optinum::lina::expr::ref;
using optinum::lina::expr::scale;
using optinum::simd::Matrix;
using optinum::simd::Vector;

TEST_CASE("lina::expr vector expression eval") {
    Vector<float, 3> a;
    a[0] = 1.f;
    a[1] = 2.f;
    a[2] = 3.f;
    Vector<float, 3> b;
    b[0] = 4.f;
    b[1] = 5.f;
    b[2] = 6.f;

    const auto e = add(ref(a), scale(ref(b), 2.f)); // a + 2*b
    const auto r = eval(e);
    CHECK(r[0] == doctest::Approx(9.f));
    CHECK(r[1] == doctest::Approx(12.f));
    CHECK(r[2] == doctest::Approx(15.f));
}

TEST_CASE("lina::expr matrix expression eval") {
    Matrix<int, 2, 2> a;
    a(0, 0) = 1;
    a(1, 0) = 3;
    a(0, 1) = 2;
    a(1, 1) = 4;

    Matrix<int, 2, 2> b;
    b(0, 0) = 5;
    b(1, 0) = 7;
    b(0, 1) = 6;
    b(1, 1) = 8;

    const auto e = add(scale(ref(a), 2), ref(b)); // 2*a + b
    const auto r = eval(e);
    CHECK(r(0, 0) == 7);
    CHECK(r(1, 1) == 16);
}

