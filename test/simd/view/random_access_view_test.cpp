#include <doctest/doctest.h>
#include <optinum/simd/view/random_access_view.hpp>

using optinum::simd::random_access_view;

TEST_CASE("Random access view construction and element access") {
    double data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::size_t indices[5] = {9, 3, 7, 1, 5}; // Non-contiguous access

    random_access_view<double, 4, 5> view(data, indices);

    // Element access
    CHECK(view[0] == doctest::Approx(9.0)); // data[9]
    CHECK(view[1] == doctest::Approx(3.0)); // data[3]
    CHECK(view[2] == doctest::Approx(7.0)); // data[7]
    CHECK(view[3] == doctest::Approx(1.0)); // data[1]
    CHECK(view[4] == doctest::Approx(5.0)); // data[5]
}

TEST_CASE("Random access view fill") {
    float data[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t indices[4] = {2, 5, 1, 7};

    random_access_view<float, 4, 4> view(data, indices);
    view.fill(3.14f);

    // Check scattered positions
    CHECK(data[2] == doctest::Approx(3.14f));
    CHECK(data[5] == doctest::Approx(3.14f));
    CHECK(data[1] == doctest::Approx(3.14f));
    CHECK(data[7] == doctest::Approx(3.14f));

    // Check untouched positions
    CHECK(data[0] == doctest::Approx(0.0f));
    CHECK(data[3] == doctest::Approx(0.0f));
}

TEST_CASE("Random access view copy operations") {
    double data[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double src[4] = {10.0, 20.0, 30.0, 40.0};
    std::size_t indices[4] = {3, 7, 1, 9};

    random_access_view<double, 4, 4> view(data, indices);

    // Copy from contiguous source
    view.copy_from(src);
    CHECK(data[3] == doctest::Approx(10.0));
    CHECK(data[7] == doctest::Approx(20.0));
    CHECK(data[1] == doctest::Approx(30.0));
    CHECK(data[9] == doctest::Approx(40.0));

    // Copy to contiguous destination
    double dst[4];
    view.copy_to(dst);
    CHECK(dst[0] == doctest::Approx(10.0));
    CHECK(dst[1] == doctest::Approx(20.0));
    CHECK(dst[2] == doctest::Approx(30.0));
    CHECK(dst[3] == doctest::Approx(40.0));
}

TEST_CASE("Random access view elementwise operations") {
    float data1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float data2[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    std::size_t indices1[3] = {1, 3, 5};
    std::size_t indices2[3] = {2, 4, 6};

    random_access_view<float, 4, 3> view1(data1, indices1);
    random_access_view<float, 4, 3> view2(data2, indices2);

    // Scalar multiplication
    view1 *= 2.0f;
    CHECK(data1[1] == doctest::Approx(4.0f));  // 2 * 2
    CHECK(data1[3] == doctest::Approx(8.0f));  // 4 * 2
    CHECK(data1[5] == doctest::Approx(12.0f)); // 6 * 2
}

TEST_CASE("Random access view gather/scatter") {
    double data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::size_t indices[8] = {9, 3, 7, 1, 5, 2, 8, 4};

    random_access_view<double, 4, 8> view(data, indices);

    // Load pack (gather)
    auto p = view.load_pack(0);
    CHECK(p[0] == doctest::Approx(9.0)); // data[9]
    CHECK(p[1] == doctest::Approx(3.0)); // data[3]
    CHECK(p[2] == doctest::Approx(7.0)); // data[7]
    CHECK(p[3] == doctest::Approx(1.0)); // data[1]
}

TEST_CASE("Random access view permutation") {
    int data[6] = {10, 20, 30, 40, 50, 60};
    std::size_t perm[6] = {5, 2, 4, 1, 3, 0}; // Reverse-ish permutation

    random_access_view<int, 4, 6> view(data, perm);

    // View sees permuted order
    CHECK(view[0] == 60); // data[5]
    CHECK(view[1] == 30); // data[2]
    CHECK(view[2] == 50); // data[4]
    CHECK(view[3] == 20); // data[1]
    CHECK(view[4] == 40); // data[3]
    CHECK(view[5] == 10); // data[0]
}
