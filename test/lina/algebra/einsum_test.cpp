#include <doctest/doctest.h>
#include <optinum/lina/algebra/einsum.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;


TEST_CASE("lina::einsum ij,jk->ik") {
    dp::mat::Matrix<float, 2, 2> a;
    a(0, 0) = 1.f;
    a(1, 0) = 3.f;
    a(0, 1) = 2.f;
    a(1, 1) = 4.f;

    dp::mat::Matrix<float, 2, 2> b;
    b(0, 0) = 5.f;
    b(1, 0) = 7.f;
    b(0, 1) = 6.f;
    b(1, 1) = 8.f;

    const auto c = lina::einsum<"ij,jk->ik">(Matrix<float, 2, 2>(a), Matrix<float, 2, 2>(b));
    CHECK(c(0, 0) == doctest::Approx(19.f));
    CHECK(c(1, 1) == doctest::Approx(50.f));
}

TEST_CASE("lina::einsum ij,j->i") {
    dp::mat::Matrix<float, 2, 3> a;
    a(0, 0) = 1.f;
    a(1, 0) = 4.f;
    a(0, 1) = 2.f;
    a(1, 1) = 5.f;
    a(0, 2) = 3.f;
    a(1, 2) = 6.f;

    dp::mat::Vector<float, 3> x;
    x[0] = 1.f;
    x[1] = 2.f;
    x[2] = 3.f;

    const auto y = lina::einsum<"ij,j->i">(Matrix<float, 2, 3>(a), Vector<float, 3>(x));
    CHECK(y[0] == doctest::Approx(14.f));
    CHECK(y[1] == doctest::Approx(32.f));
}

TEST_CASE("lina::einsum i,i->") {
    dp::mat::Vector<float, 3> a;
    a[0] = 1.f;
    a[1] = 2.f;
    a[2] = 3.f;
    dp::mat::Vector<float, 3> b;
    b[0] = 4.f;
    b[1] = 5.f;
    b[2] = 6.f;

    const auto d = lina::einsum<"i,i->">(Vector<float, 3>(a), Vector<float, 3>(b));
    CHECK(d == doctest::Approx(32.f));
}

TEST_CASE("lina::einsum ij->ji") {
    dp::mat::Matrix<int, 2, 3> a;
    a(0, 0) = 1;
    a(1, 0) = 4;
    a(0, 1) = 2;
    a(1, 1) = 5;
    a(0, 2) = 3;
    a(1, 2) = 6;

    const auto at = lina::einsum<"ij->ji">(Matrix<int, 2, 3>(a));
    CHECK(at(0, 0) == 1);
    CHECK(at(2, 1) == 6);
}
