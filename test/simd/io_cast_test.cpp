#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>
#include <sstream>

namespace dp = datapod;
namespace simd = optinum::simd;

// =============================================================================
// Vector I/O Tests
// =============================================================================

TEST_CASE("Vector stream output") {
    SUBCASE("float vector") {
        dp::mat::Vector<float, 3> storage;
        simd::Vector<float, 3> v(storage);
        v[0] = 1.0f;
        v[1] = 2.0f;
        v[2] = 3.0f;

        std::ostringstream oss;
        oss << v;
        CHECK(oss.str() == "[1, 2, 3]");
    }

    SUBCASE("int vector") {
        dp::mat::Vector<int, 4> storage;
        simd::Vector<int, 4> v(storage);
        v.iota();

        std::ostringstream oss;
        oss << v;
        CHECK(oss.str() == "[0, 1, 2, 3]");
    }

    SUBCASE("double vector") {
        dp::mat::Vector<double, 2> storage;
        simd::Vector<double, 2> v(storage);
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
    SUBCASE("int to float using cast_to") {
        dp::mat::Vector<int, 4> storage;
        simd::Vector<int, 4> vi(storage);
        vi.iota();

        dp::mat::Vector<float, 4> result_storage;
        simd::cast_to(result_storage.data(), vi);

        CHECK(result_storage[0] == 0.0f);
        CHECK(result_storage[1] == 1.0f);
        CHECK(result_storage[2] == 2.0f);
        CHECK(result_storage[3] == 3.0f);
    }

    SUBCASE("float to double using cast_to") {
        dp::mat::Vector<float, 3> storage;
        simd::Vector<float, 3> vf(storage);
        vf[0] = 1.5f;
        vf[1] = 2.5f;
        vf[2] = 3.5f;

        dp::mat::Vector<double, 3> result_storage;
        simd::cast_to(result_storage.data(), vf);

        CHECK(result_storage[0] == doctest::Approx(1.5));
        CHECK(result_storage[1] == doctest::Approx(2.5));
        CHECK(result_storage[2] == doctest::Approx(3.5));
    }

    SUBCASE("double to int (truncation) using cast_to") {
        dp::mat::Vector<double, 3> storage;
        simd::Vector<double, 3> vd(storage);
        vd[0] = 1.9;
        vd[1] = 2.5;
        vd[2] = 3.1;

        dp::mat::Vector<int, 3> result_storage;
        simd::cast_to(result_storage.data(), vd);

        CHECK(result_storage[0] == 1);
        CHECK(result_storage[1] == 2);
        CHECK(result_storage[2] == 3);
    }

    SUBCASE("float to int to float (round trip)") {
        dp::mat::Vector<float, 4> storage;
        simd::Vector<float, 4> v1(storage);
        v1.iota(10.0f);

        dp::mat::Vector<int, 4> int_storage;
        simd::cast_to(int_storage.data(), v1);

        dp::mat::Vector<float, 4> float_storage;
        simd::Vector<int, 4> vi(int_storage);
        simd::cast_to(float_storage.data(), vi);

        CHECK(float_storage[0] == 10.0f);
        CHECK(float_storage[1] == 11.0f);
        CHECK(float_storage[2] == 12.0f);
        CHECK(float_storage[3] == 13.0f);
    }
}

// =============================================================================
// Matrix I/O Tests
// =============================================================================

TEST_CASE("Matrix stream output") {
    SUBCASE("2x2 float matrix") {
        dp::mat::Matrix<float, 2, 2> storage;
        simd::Matrix<float, 2, 2> m(storage);
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
        dp::mat::Matrix<int, 3, 3> storage;
        simd::Matrix<int, 3, 3> m(storage);
        m.iota();

        std::ostringstream oss;
        oss << m;
        // Column-major: storage [0,1,2,3,4,5,6,7,8] = col0[0,1,2], col1[3,4,5], col2[6,7,8]
        CHECK(oss.str() == "[[0, 1, 2]\n [3, 4, 5]\n [6, 7, 8]]");
    }

    SUBCASE("single row matrix") {
        dp::mat::Matrix<int, 1, 4> storage;
        simd::Matrix<int, 1, 4> m(storage);
        m.iota();

        std::ostringstream oss;
        oss << m;
        // 1 row, 4 columns: each column has 1 element
        CHECK(oss.str() == "[[0]\n [1]\n [2]\n [3]]");
    }

    SUBCASE("single column matrix") {
        dp::mat::Matrix<int, 3, 1> storage;
        simd::Matrix<int, 3, 1> m(storage);
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

TEST_CASE("Matrix type conversion using cast_to") {
    SUBCASE("int to float") {
        dp::mat::Matrix<int, 2, 3> storage;
        simd::Matrix<int, 2, 3> mi(storage);
        mi.iota();

        dp::mat::Matrix<float, 2, 3> result_storage;
        simd::cast_to(result_storage.data(), mi);

        // Linear indexing (column-major storage)
        CHECK(result_storage[0] == 0.0f);
        CHECK(result_storage[1] == 1.0f);
        CHECK(result_storage[2] == 2.0f);
        CHECK(result_storage[3] == 3.0f);
        CHECK(result_storage[4] == 4.0f);
        CHECK(result_storage[5] == 5.0f);
    }

    SUBCASE("float to double") {
        dp::mat::Matrix<float, 2, 2> storage;
        simd::Matrix<float, 2, 2> mf(storage);
        mf.fill(3.14f);

        dp::mat::Matrix<double, 2, 2> result_storage;
        simd::cast_to(result_storage.data(), mf);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(result_storage[i] == doctest::Approx(3.14));
        }
    }

    SUBCASE("preserves matrix structure") {
        dp::mat::Matrix<int, 3, 2> storage;
        simd::Matrix<int, 3, 2> mi(storage);
        mi.iota();

        dp::mat::Matrix<float, 3, 2> result_storage;
        simd::cast_to(result_storage.data(), mi);

        CHECK(result_storage.rows() == 3);
        CHECK(result_storage.cols() == 2);
        CHECK(result_storage.size() == 6);
    }
}

// =============================================================================
// Matrix Flatten Tests (if flatten method exists)
// =============================================================================

TEST_CASE("Matrix linear access") {
    SUBCASE("2x3 matrix linear iteration") {
        dp::mat::Matrix<int, 2, 3> storage;
        simd::Matrix<int, 2, 3> m(storage);
        m.iota();

        // Access linearly
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(m[i] == static_cast<int>(i));
        }
    }

    SUBCASE("copy to vector") {
        dp::mat::Matrix<float, 3, 3> storage;
        simd::Matrix<float, 3, 3> m(storage);
        m.iota(1.0f);

        // Copy to vector
        dp::mat::Vector<float, 9> v_storage;
        simd::flatten_to(v_storage.data(), m);

        CHECK(v_storage[0] == 1.0f);
        CHECK(v_storage[8] == 9.0f);
    }
}

// =============================================================================
// Tensor I/O Tests
// =============================================================================

TEST_CASE("Tensor stream output") {
    namespace dp = datapod;

    SUBCASE("2x2x2 tensor") {
        dp::mat::Tensor<int, 2, 2, 2> storage;
        simd::Tensor<int, 2, 2, 2> t(storage);
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
        dp::mat::Tensor<float, 3, 3, 3> storage;
        simd::Tensor<float, 3, 3, 3> t(storage);
        t.fill(1.5f);

        std::ostringstream oss;
        oss << t;
        std::string result = oss.str();

        CHECK(result.find("Tensor<3x3x3>") != std::string::npos);
    }

    SUBCASE("4D tensor") {
        dp::mat::Tensor<int, 2, 2, 2, 2> storage;
        simd::Tensor<int, 2, 2, 2, 2> t(storage);
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

TEST_CASE("Tensor type conversion using cast_to") {
    namespace dp = datapod;

    SUBCASE("int to float") {
        dp::mat::Tensor<int, 2, 2, 2> ti_storage;
        simd::Tensor<int, 2, 2, 2> ti(ti_storage);
        for (std::size_t i = 0; i < 8; ++i) {
            ti[i] = static_cast<int>(i);
        }

        dp::mat::Tensor<float, 2, 2, 2> tf_storage;
        simd::Tensor<float, 2, 2, 2> tf(tf_storage);
        simd::cast_to(tf.data(), ti);
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(tf[i] == static_cast<float>(i));
        }
    }

    SUBCASE("float to double") {
        dp::mat::Tensor<float, 2, 2, 2> tf_storage;
        simd::Tensor<float, 2, 2, 2> tf(tf_storage);
        tf.fill(2.5f);

        dp::mat::Tensor<double, 2, 2, 2> td_storage;
        simd::Tensor<double, 2, 2, 2> td(td_storage);
        simd::cast_to(td.data(), tf);
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(td[i] == doctest::Approx(2.5));
        }
    }

    SUBCASE("preserves tensor shape") {
        dp::mat::Tensor<int, 3, 4, 5> ti_storage;
        simd::Tensor<int, 3, 4, 5> ti(ti_storage);
        ti.fill(1);

        dp::mat::Tensor<float, 3, 4, 5> tf_storage;
        simd::Tensor<float, 3, 4, 5> tf(tf_storage);
        simd::cast_to(tf.data(), ti);
        CHECK(tf.size() == 60);
        CHECK(tf.rank == 3);
        auto shape = tf.shape();
        CHECK(shape[0] == 3);
        CHECK(shape[1] == 4);
        CHECK(shape[2] == 5);
    }

    SUBCASE("4D tensor conversion") {
        dp::mat::Tensor<double, 2, 3, 2, 2> td_storage;
        simd::Tensor<double, 2, 3, 2, 2> td(td_storage);
        for (std::size_t i = 0; i < 24; ++i) {
            td[i] = static_cast<double>(i) * 0.5;
        }

        dp::mat::Tensor<int, 2, 3, 2, 2> ti_storage;
        simd::Tensor<int, 2, 3, 2, 2> ti(ti_storage);
        simd::cast_to(ti.data(), td);
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
        dp::mat::Vector<int, 3> storage;
        simd::Vector<int, 3> vi(storage);
        vi.iota();

        dp::mat::Vector<float, 3> result_storage;
        simd::cast_to(result_storage.data(), vi);
        simd::Vector<float, 3> vf(result_storage);

        std::ostringstream oss;
        oss << vf;
        CHECK(oss.str() == "[0, 1, 2]");
    }

    SUBCASE("cast then print matrix") {
        // Create owned matrix first
        dp::mat::Matrix<int, 2, 2> mi_owned;
        simd::Matrix<int, 2, 2> mi(mi_owned);
        mi.iota();

        dp::mat::Matrix<double, 2, 2> md_owned;
        simd::cast_to(md_owned.data(), mi);
        simd::Matrix<double, 2, 2> md(md_owned);

        std::ostringstream oss;
        oss << md;
        std::string result = oss.str();
        CHECK(result.find("[[") != std::string::npos);
    }
}
