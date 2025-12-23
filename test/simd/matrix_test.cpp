#include <doctest/doctest.h>
#include <optinum/simd/matrix.hpp>

using optinum::simd::Matrix;
using optinum::simd::Tensor;

TEST_CASE("Matrix construction") {
    SUBCASE("default construction") {
        Matrix<float, 3, 3> m;
        CHECK(m.rows() == 3);
        CHECK(m.cols() == 3);
        CHECK(m.size() == 9);
    }

    SUBCASE("from datapod") {
        datapod::matrix<float, 2, 2> pod{};
        pod(0, 0) = 1.0f;
        pod(0, 1) = 2.0f;
        pod(1, 0) = 3.0f;
        pod(1, 1) = 4.0f;

        Matrix<float, 2, 2> m(pod);
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 2.0f);
        CHECK(m(1, 0) == 3.0f);
        CHECK(m(1, 1) == 4.0f);
    }
}

TEST_CASE("Matrix element access") {
    Matrix<float, 2, 3> m;
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(0, 2) = 3.0f;
    m(1, 0) = 4.0f;
    m(1, 1) = 5.0f;
    m(1, 2) = 6.0f;

    CHECK(m(0, 0) == 1.0f);
    CHECK(m(1, 2) == 6.0f);

    // Linear access
    CHECK(m[0] == 1.0f); // column-major
}

TEST_CASE("Matrix fill") {
    Matrix<double, 3, 3> m;
    m.fill(2.5);

    for (std::size_t i = 0; i < m.size(); ++i) {
        CHECK(m[i] == 2.5);
    }
}

TEST_CASE("Matrix set_identity") {
    Matrix<float, 3, 3> m;
    m.fill(99.0f);
    m.set_identity();

    CHECK(m(0, 0) == 1.0f);
    CHECK(m(1, 1) == 1.0f);
    CHECK(m(2, 2) == 1.0f);
    CHECK(m(0, 1) == 0.0f);
    CHECK(m(1, 0) == 0.0f);
}

TEST_CASE("Matrix element-wise arithmetic") {
    Matrix<float, 2, 2> a;
    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(1, 0) = 3.0f;
    a(1, 1) = 4.0f;

    Matrix<float, 2, 2> b;
    b(0, 0) = 5.0f;
    b(0, 1) = 6.0f;
    b(1, 0) = 7.0f;
    b(1, 1) = 8.0f;

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(c(0, 0) == 6.0f);
        CHECK(c(1, 1) == 12.0f);
    }

    SUBCASE("subtraction") {
        auto c = b - a;
        CHECK(c(0, 0) == 4.0f);
        CHECK(c(1, 1) == 4.0f);
    }

    SUBCASE("scalar multiplication") {
        auto c = a * 2.0f;
        CHECK(c(0, 0) == 2.0f);
        CHECK(c(1, 1) == 8.0f);
    }
}

TEST_CASE("Matrix multiplication") {
    Matrix<float, 2, 2> a;
    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(1, 0) = 3.0f;
    a(1, 1) = 4.0f;

    Matrix<float, 2, 2> b;
    b(0, 0) = 5.0f;
    b(0, 1) = 6.0f;
    b(1, 0) = 7.0f;
    b(1, 1) = 8.0f;

    auto c = a * b;
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    CHECK(c(0, 0) == 19.0f);
    CHECK(c(0, 1) == 22.0f);
    CHECK(c(1, 0) == 43.0f);
    CHECK(c(1, 1) == 50.0f);
}

TEST_CASE("Matrix multiplication with identity") {
    Matrix<float, 3, 3> a;
    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(0, 2) = 3.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 5.0f;
    a(1, 2) = 6.0f;
    a(2, 0) = 7.0f;
    a(2, 1) = 8.0f;
    a(2, 2) = 9.0f;

    auto I = optinum::simd::identity<float, 3>();
    auto b = a * I;

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(b(i, j) == a(i, j));
        }
    }
}

TEST_CASE("Matrix-vector multiplication") {
    Matrix<float, 2, 3> m;
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(0, 2) = 3.0f;
    m(1, 0) = 4.0f;
    m(1, 1) = 5.0f;
    m(1, 2) = 6.0f;

    Tensor<float, 3> v;
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;

    auto r = m * v;
    // [1 2 3] * [1]   [1*1+2*2+3*3]   [14]
    // [4 5 6]   [2] = [4*1+5*2+6*3] = [32]
    //           [3]
    CHECK(r[0] == 14.0f);
    CHECK(r[1] == 32.0f);
}

TEST_CASE("Matrix transpose") {
    Matrix<float, 2, 3> m;
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(0, 2) = 3.0f;
    m(1, 0) = 4.0f;
    m(1, 1) = 5.0f;
    m(1, 2) = 6.0f;

    auto mt = optinum::simd::transpose(m);

    CHECK(mt.rows() == 3);
    CHECK(mt.cols() == 2);
    CHECK(mt(0, 0) == 1.0f);
    CHECK(mt(0, 1) == 4.0f);
    CHECK(mt(1, 0) == 2.0f);
    CHECK(mt(2, 1) == 6.0f);
}

TEST_CASE("Matrix trace") {
    Matrix<float, 3, 3> m;
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(0, 2) = 3.0f;
    m(1, 0) = 4.0f;
    m(1, 1) = 5.0f;
    m(1, 2) = 6.0f;
    m(2, 0) = 7.0f;
    m(2, 1) = 8.0f;
    m(2, 2) = 9.0f;

    CHECK(optinum::simd::trace(m) == 15.0f);
}

TEST_CASE("Matrix frobenius_norm") {
    Matrix<float, 2, 2> m;
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(1, 0) = 3.0f;
    m(1, 1) = 4.0f;

    // sqrt(1 + 4 + 9 + 16) = sqrt(30)
    CHECK(optinum::simd::frobenius_norm(m) == doctest::Approx(std::sqrt(30.0f)));
}

TEST_CASE("Matrix identity factory") {
    auto I = optinum::simd::identity<double, 4>();

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (i == j) {
                CHECK(I(i, j) == 1.0);
            } else {
                CHECK(I(i, j) == 0.0);
            }
        }
    }
}

TEST_CASE("Matrix comparison") {
    Matrix<int, 2, 2> a;
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    Matrix<int, 2, 2> b;
    b(0, 0) = 1;
    b(0, 1) = 2;
    b(1, 0) = 3;
    b(1, 1) = 4;

    Matrix<int, 2, 2> c;
    c(0, 0) = 1;
    c(0, 1) = 2;
    c(1, 0) = 3;
    c(1, 1) = 5;

    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("Matrix pod access") {
    Matrix<float, 2, 2> m;
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(1, 0) = 3.0f;
    m(1, 1) = 4.0f;

    datapod::matrix<float, 2, 2> &pod = m.pod();
    CHECK(pod(0, 0) == 1.0f);

    pod(0, 0) = 99.0f;
    CHECK(m(0, 0) == 99.0f);
}
