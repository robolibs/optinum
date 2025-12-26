#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>
#include <sstream>

using namespace optinum::simd;

// =============================================================================
// Vector I/O Tests
// =============================================================================

TEST_CASE("Vector stream output") {
    SUBCASE("float vector") {
        Vector<float, 3> v;
        v[0] = 1.0f;
        v[1] = 2.0f;
        v[2] = 3.0f;

        std::ostringstream oss;
        oss << v;
        CHECK(oss.str() == "[1, 2, 3]");
    }

    SUBCASE("int vector") {
        Vector<int, 4> v;
        v.iota();

        std::ostringstream oss;
        oss << v;
        CHECK(oss.str() == "[0, 1, 2, 3]");
    }

    SUBCASE("double vector") {
        Vector<double, 2> v;
        v[0] = 3.14;
        v[1] = 2.71;

        std::ostringstream oss;
        oss << v;
        std::string result = oss.str();
        CHECK(result.find("3.14") != std::string::npos);
        CHECK(result.find("2.71") != std::string::npos);
    }
}

// =============================================================================
// Vector Cast Tests
// =============================================================================

TEST_CASE("Vector type conversion") {
    SUBCASE("int to float") {
        Vector<int, 4> vi;
        vi.iota();

        auto vf = cast<float>(vi);
        CHECK(vf[0] == 0.0f);
        CHECK(vf[1] == 1.0f);
        CHECK(vf[2] == 2.0f);
        CHECK(vf[3] == 3.0f);
    }

    SUBCASE("float to double") {
        Vector<float, 3> vf;
        vf[0] = 1.5f;
        vf[1] = 2.5f;
        vf[2] = 3.5f;

        auto vd = cast<double>(vf);
        CHECK(vd[0] == doctest::Approx(1.5));
        CHECK(vd[1] == doctest::Approx(2.5));
        CHECK(vd[2] == doctest::Approx(3.5));
    }

    SUBCASE("double to int (truncation)") {
        Vector<double, 3> vd;
        vd[0] = 1.9;
        vd[1] = 2.5;
        vd[2] = 3.1;

        auto vi = cast<int>(vd);
        CHECK(vi[0] == 1);
        CHECK(vi[1] == 2);
        CHECK(vi[2] == 3);
    }

    SUBCASE("float to int to float (round trip)") {
        Vector<float, 4> v1;
        v1.iota(10.0f);

        auto vi = cast<int>(v1);
        auto v2 = cast<float>(vi);

        CHECK(v2[0] == 10.0f);
        CHECK(v2[1] == 11.0f);
        CHECK(v2[2] == 12.0f);
        CHECK(v2[3] == 13.0f);
    }
}

// =============================================================================
// Matrix I/O Tests
// =============================================================================

TEST_CASE("Matrix stream output") {
    SUBCASE("2x2 float matrix") {
        Matrix<float, 2, 2> m;
        m(0, 0) = 1.0f;
        m(0, 1) = 2.0f;
        m(1, 0) = 3.0f;
        m(1, 1) = 4.0f;

        std::ostringstream oss;
        oss << m;
        // Column-major: first column [1,3], second column [2,4]
        CHECK(oss.str() == "[[1, 3]\n [2, 4]]");
    }

    SUBCASE("3x3 int matrix") {
        Matrix<int, 3, 3> m;
        m.iota();

        std::ostringstream oss;
        oss << m;
        // Column-major: storage [0,1,2,3,4,5,6,7,8] = col0[0,1,2], col1[3,4,5], col2[6,7,8]
        CHECK(oss.str() == "[[0, 1, 2]\n [3, 4, 5]\n [6, 7, 8]]");
    }

    SUBCASE("single row matrix") {
        Matrix<int, 1, 4> m;
        m.iota();

        std::ostringstream oss;
        oss << m;
        // 1 row, 4 columns: each column has 1 element
        CHECK(oss.str() == "[[0]\n [1]\n [2]\n [3]]");
    }

    SUBCASE("single column matrix") {
        Matrix<int, 3, 1> m;
        m.iota();

        std::ostringstream oss;
        oss << m;
        // 3 rows, 1 column: single column with 3 elements
        CHECK(oss.str() == "[[0, 1, 2]]");
    }
}

// =============================================================================
// Matrix Cast Tests
// =============================================================================

TEST_CASE("Matrix type conversion") {
    SUBCASE("int to float") {
        Matrix<int, 2, 3> mi;
        mi.iota();

        auto mf = cast<float>(mi);
        // Linear indexing (column-major storage)
        CHECK(mf[0] == 0.0f);
        CHECK(mf[1] == 1.0f);
        CHECK(mf[2] == 2.0f);
        CHECK(mf[3] == 3.0f);
        CHECK(mf[4] == 4.0f);
        CHECK(mf[5] == 5.0f);
    }

    SUBCASE("float to double") {
        Matrix<float, 2, 2> mf;
        mf.fill(3.14f);

        auto md = cast<double>(mf);
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(md[i] == doctest::Approx(3.14));
        }
    }

    SUBCASE("preserves matrix structure") {
        Matrix<int, 3, 2> mi;
        mi.iota();

        auto mf = cast<float>(mi);
        CHECK(mf.rows() == 3);
        CHECK(mf.cols() == 2);
        CHECK(mf.size() == 6);
    }
}

// =============================================================================
// Matrix Flatten Tests
// =============================================================================

TEST_CASE("Matrix flatten") {
    SUBCASE("2x3 matrix to vector") {
        Matrix<int, 2, 3> m;
        m.iota();

        auto v = m.flatten();
        CHECK(v.size() == 6);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(v[i] == static_cast<int>(i));
        }
    }

    SUBCASE("square matrix") {
        Matrix<float, 3, 3> m;
        m.iota(1.0f);

        auto v = m.flatten();
        CHECK(v.size() == 9);
        CHECK(v[0] == 1.0f);
        CHECK(v[8] == 9.0f);
    }

    SUBCASE("single row becomes vector") {
        Matrix<double, 1, 4> m;
        m[0] = 1.1;
        m[1] = 2.2;
        m[2] = 3.3;
        m[3] = 4.4;

        auto v = m.flatten();
        CHECK(v.size() == 4);
        CHECK(v[0] == doctest::Approx(1.1));
        CHECK(v[3] == doctest::Approx(4.4));
    }

    SUBCASE("flatten preserves values") {
        Matrix<int, 4, 2> m;
        for (std::size_t i = 0; i < 8; ++i) {
            m[i] = static_cast<int>(i * 10);
        }

        auto v = m.flatten();
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(v[i] == static_cast<int>(i * 10));
        }
    }

    SUBCASE("constexpr flatten") {
        constexpr auto test = []() {
            Matrix<int, 2, 2> m;
            m[0] = 1;
            m[1] = 2;
            m[2] = 3;
            m[3] = 4;
            auto v = m.flatten();
            return v[0] + v[3];
        };
        constexpr int result = test();
        CHECK(result == 5);
    }
}

// =============================================================================
// Tensor I/O Tests
// =============================================================================

TEST_CASE("Tensor stream output") {
    SUBCASE("2x2x2 tensor") {
        Tensor<int, 2, 2, 2> t;
        t.fill(42);

        std::ostringstream oss;
        oss << t;
        std::string result = oss.str();

        // Check format: Tensor<2x2x2>[...]
        CHECK(result.find("Tensor<2x2x2>") != std::string::npos);
        CHECK(result.find("[") != std::string::npos);
        CHECK(result.find("42") != std::string::npos);
    }

    SUBCASE("3x3x3 tensor") {
        Tensor<float, 3, 3, 3> t;
        t.fill(1.5f);

        std::ostringstream oss;
        oss << t;
        std::string result = oss.str();

        CHECK(result.find("Tensor<3x3x3>") != std::string::npos);
    }

    SUBCASE("4D tensor") {
        Tensor<int, 2, 2, 2, 2> t;
        t.fill(7);

        std::ostringstream oss;
        oss << t;
        std::string result = oss.str();

        CHECK(result.find("Tensor<2x2x2x2>") != std::string::npos);
        CHECK(result.find("7") != std::string::npos);
    }
}

// =============================================================================
// Tensor Cast Tests
// =============================================================================

TEST_CASE("Tensor type conversion") {
    SUBCASE("int to float") {
        Tensor<int, 2, 2, 2> ti;
        for (std::size_t i = 0; i < 8; ++i) {
            ti[i] = static_cast<int>(i);
        }

        auto tf = cast<float>(ti);
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(tf[i] == static_cast<float>(i));
        }
    }

    SUBCASE("float to double") {
        Tensor<float, 2, 2, 2> tf;
        tf.fill(2.5f);

        auto td = cast<double>(tf);
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(td[i] == doctest::Approx(2.5));
        }
    }

    SUBCASE("preserves tensor shape") {
        Tensor<int, 3, 4, 5> ti;
        ti.fill(1);

        auto tf = cast<float>(ti);
        CHECK(tf.size() == 60);
        CHECK(tf.rank == 3);
        auto shape = tf.shape();
        CHECK(shape[0] == 3);
        CHECK(shape[1] == 4);
        CHECK(shape[2] == 5);
    }

    SUBCASE("4D tensor conversion") {
        Tensor<double, 2, 3, 2, 2> td;
        for (std::size_t i = 0; i < 24; ++i) {
            td[i] = static_cast<double>(i) * 0.5;
        }

        auto ti = cast<int>(td);
        CHECK(ti.size() == 24);
        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(ti[i] == static_cast<int>(i * 0.5));
        }
    }
}

// =============================================================================
// Combined Operations Tests
// =============================================================================

TEST_CASE("Combined I/O and cast operations") {
    SUBCASE("cast then print vector") {
        Vector<int, 3> vi;
        vi.iota();

        auto vf = cast<float>(vi);
        std::ostringstream oss;
        oss << vf;
        CHECK(oss.str() == "[0, 1, 2]");
    }

    SUBCASE("cast then print matrix") {
        Matrix<int, 2, 2> mi;
        mi.iota();

        auto md = cast<double>(mi);
        std::ostringstream oss;
        oss << md;
        std::string result = oss.str();
        CHECK(result.find("[[") != std::string::npos);
    }

    SUBCASE("flatten then cast") {
        Matrix<int, 2, 3> m;
        m.iota(10);

        auto v_int = m.flatten();
        auto v_float = cast<float>(v_int);

        CHECK(v_float[0] == 10.0f);
        CHECK(v_float[5] == 15.0f);
    }

    SUBCASE("multiple type conversions") {
        Vector<int, 4> v1;
        v1.iota();

        auto v2 = cast<float>(v1);
        auto v3 = cast<double>(v2);
        auto v4 = cast<int>(v3);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(v4[i] == v1[i]);
        }
    }
}
