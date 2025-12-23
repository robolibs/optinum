#include <doctest/doctest.h>
#include <optinum/simd/backend/reduce.hpp>

TEST_CASE("backend reduce_sum/min/max") {
    int a[7] = {5, 1, 9, -2, 3, 3, 0};

    CHECK(optinum::simd::backend::reduce_sum<int, 7>(a) == 19);
    CHECK(optinum::simd::backend::reduce_min<int, 7>(a) == -2);
    CHECK(optinum::simd::backend::reduce_max<int, 7>(a) == 9);
}

