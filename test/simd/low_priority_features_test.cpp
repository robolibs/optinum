#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace simd = optinum::simd;

// =============================================================================
// Test 1: noalias() - Optimization Hint
// =============================================================================

TEST_CASE("noalias() - Vector assignment") {
    dp::mat::Vector<float, 4> a_storage, b_storage, c_storage;
    simd::Vector<float, 4> a(a_storage), b(b_storage), c(c_storage);

    // Initialize using element access
    for (size_t i = 0; i < 4; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(i + 5);
    }

    // Use SIMD backend for addition
    simd::backend::add<float, 4>(c.data(), a.data(), b.data());

    CHECK(c[0] == doctest::Approx(6.0f));
    CHECK(c[1] == doctest::Approx(8.0f));
    CHECK(c[2] == doctest::Approx(10.0f));
    CHECK(c[3] == doctest::Approx(12.0f));
}

TEST_CASE("MatrixNoAlias") {
    dp::mat::Matrix<double, 2, 2> a_storage, b_storage, c_storage;
    simd::Matrix<double, 2, 2> a(a_storage), b(b_storage), c(c_storage);

    // Initialize matrices
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    b(0, 0) = 5.0;
    b(0, 1) = 6.0;
    b(1, 0) = 7.0;
    b(1, 1) = 8.0;

    // Use SIMD backend for addition
    simd::backend::add<double, 4>(c.data(), a.data(), b.data());

    CHECK(c(0, 0) == doctest::Approx(6.0));
    CHECK(c(1, 0) == doctest::Approx(10.0));
    CHECK(c(0, 1) == doctest::Approx(8.0));
    CHECK(c(1, 1) == doctest::Approx(12.0));
}

TEST_CASE("TensorNoAlias") {
    // Create storage for tensors (Tensor is now a non-owning view)
    dp::mat::Tensor<float, 2, 2, 2> a_storage, b_storage, c_storage;
    simd::Tensor<float, 2, 2, 2> a(a_storage), b(b_storage), c(c_storage);

    // Initialize tensors
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                a(i, j, k) = static_cast<float>(i + j + k);
                b(i, j, k) = static_cast<float>(i * j * k + 1);
            }
        }
    }

    // With non-owning views, use in-place operations
    // Copy a to c, then add b in-place
    for (size_t i = 0; i < 8; ++i) {
        c[i] = a[i];
    }
    c += b;

    // Verify results
    CHECK(c(0, 0, 0) == doctest::Approx(0.0f + 1.0f));
    CHECK(c(1, 1, 1) == doctest::Approx(3.0f + 2.0f));
}

TEST_CASE("WrapperDataAccess") {
    dp::mat::Vector<float, 3> storage;
    simd::Vector<float, 3> v(storage);
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;

    auto wrapped = simd::noalias(v);

    // Verify we can access the underlying reference
    CHECK(&wrapped.get() == &v);
    CHECK(wrapped.get()[0] == doctest::Approx(1.0f));
    CHECK(wrapped.get()[1] == doctest::Approx(2.0f));
    CHECK(wrapped.get()[2] == doctest::Approx(3.0f));
}

// =============================================================================
// Test 2: Layout Conversion
// =============================================================================

TEST_CASE("ToRowMajorMatrix") {
    // Create column-major matrix (optinum default)
    dp::mat::Matrix<double, 2, 3> storage;
    simd::Matrix<double, 2, 3> col_major(storage);
    col_major(0, 0) = 1.0;
    col_major(0, 1) = 2.0;
    col_major(0, 2) = 3.0;
    col_major(1, 0) = 4.0;
    col_major(1, 1) = 5.0;
    col_major(1, 2) = 6.0;

    // Convert to row-major (returns transpose)
    auto row_major = simd::torowmajor(col_major);

    // torowmajor transposes the matrix
    CHECK(row_major(0, 0) == doctest::Approx(1.0));
    CHECK(row_major(0, 1) == doctest::Approx(4.0));
    CHECK(row_major(1, 0) == doctest::Approx(2.0));
    CHECK(row_major(1, 1) == doctest::Approx(5.0));
    CHECK(row_major(2, 0) == doctest::Approx(3.0));
    CHECK(row_major(2, 1) == doctest::Approx(6.0));
}

TEST_CASE("ToColumnMajorMatrix") {
    dp::mat::Matrix<float, 3, 2> storage;
    simd::Matrix<float, 3, 2> mat(storage);
    mat(0, 0) = 1.0f;
    mat(0, 1) = 2.0f;
    mat(1, 0) = 3.0f;
    mat(1, 1) = 4.0f;
    mat(2, 0) = 5.0f;
    mat(2, 1) = 6.0f;

    auto col_major = simd::tocolumnmajor(mat);

    // Since optinum is already column-major, this transposes
    CHECK(col_major(0, 0) == doctest::Approx(1.0f));
    CHECK(col_major(0, 1) == doctest::Approx(3.0f));
    CHECK(col_major(0, 2) == doctest::Approx(5.0f));
    CHECK(col_major(1, 0) == doctest::Approx(2.0f));
    CHECK(col_major(1, 1) == doctest::Approx(4.0f));
    CHECK(col_major(1, 2) == doctest::Approx(6.0f));
}

TEST_CASE("CopyToRowMajorArray") {
    dp::mat::Matrix<double, 2, 3> storage;
    simd::Matrix<double, 2, 3> mat(storage);
    mat(0, 0) = 1.0;
    mat(0, 1) = 2.0;
    mat(0, 2) = 3.0;
    mat(1, 0) = 4.0;
    mat(1, 1) = 5.0;
    mat(1, 2) = 6.0;

    double row_array[6];
    simd::copy_to_rowmajor(mat, row_array);

    // Row-major layout: [row0, row1] = [1,2,3,4,5,6]
    CHECK(row_array[0] == doctest::Approx(1.0));
    CHECK(row_array[1] == doctest::Approx(2.0));
    CHECK(row_array[2] == doctest::Approx(3.0));
    CHECK(row_array[3] == doctest::Approx(4.0));
    CHECK(row_array[4] == doctest::Approx(5.0));
    CHECK(row_array[5] == doctest::Approx(6.0));
}

TEST_CASE("CopyFromRowMajorArray") {
    double row_array[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    dp::mat::Matrix<double, 2, 3> storage;
    simd::Matrix<double, 2, 3> mat(storage);
    simd::copy_from_rowmajor(mat, row_array);

    // Verify column-major storage
    CHECK(mat(0, 0) == doctest::Approx(1.0));
    CHECK(mat(0, 1) == doctest::Approx(2.0));
    CHECK(mat(0, 2) == doctest::Approx(3.0));
    CHECK(mat(1, 0) == doctest::Approx(4.0));
    CHECK(mat(1, 1) == doctest::Approx(5.0));
    CHECK(mat(1, 2) == doctest::Approx(6.0));
}

TEST_CASE("TensorColumnMajorConversion") {
    // Create storage for tensor (Tensor is now a non-owning view)
    dp::mat::Tensor<float, 2, 2, 2> tensor_storage;
    simd::Tensor<float, 2, 2, 2> tensor(tensor_storage);

    // Initialize with known pattern
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                tensor(i, j, k) = static_cast<float>(i * 4 + j * 2 + k);
            }
        }
    }

    float col_array[8];
    simd::copy_to_columnmajor(tensor, col_array);

    // Verify first and last elements
    CHECK(col_array[0] == doctest::Approx(0.0f));
    CHECK(col_array[7] == doctest::Approx(7.0f));

    // Copy back and verify
    dp::mat::Tensor<float, 2, 2, 2> recovered_storage;
    simd::Tensor<float, 2, 2, 2> recovered(recovered_storage);
    simd::copy_from_columnmajor(recovered, col_array);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                CHECK(recovered(i, j, k) == doctest::Approx(tensor(i, j, k)));
            }
        }
    }
}

// =============================================================================
// Test 3: Voigt Notation
// =============================================================================

TEST_CASE("ToVoigtStressTensor") {
    // Create symmetric 3x3 stress tensor
    dp::mat::Matrix<double, 3, 3> storage;
    simd::Matrix<double, 3, 3> stress(storage);
    stress(0, 0) = 1.0;
    stress(0, 1) = 4.0;
    stress(0, 2) = 5.0;
    stress(1, 0) = 4.0;
    stress(1, 1) = 2.0;
    stress(1, 2) = 6.0;
    stress(2, 0) = 5.0;
    stress(2, 1) = 6.0;
    stress(2, 2) = 3.0;

    auto voigt = simd::to_voigt(stress);

    // Voigt notation: [σ11, σ22, σ33, σ23, σ13, σ12]
    CHECK(voigt[0] == doctest::Approx(1.0)); // σ11
    CHECK(voigt[1] == doctest::Approx(2.0)); // σ22
    CHECK(voigt[2] == doctest::Approx(3.0)); // σ33
    CHECK(voigt[3] == doctest::Approx(6.0)); // σ23
    CHECK(voigt[4] == doctest::Approx(5.0)); // σ13
    CHECK(voigt[5] == doctest::Approx(4.0)); // σ12
}

TEST_CASE("FromVoigtToTensor") {
    dp::mat::Vector<double, 6> storage;
    simd::Vector<double, 6> voigt(storage);
    voigt[0] = 1.0;
    voigt[1] = 2.0;
    voigt[2] = 3.0;
    voigt[3] = 4.0;
    voigt[4] = 5.0;
    voigt[5] = 6.0;

    auto tensor = simd::from_voigt(voigt);

    // Verify diagonal
    CHECK(tensor(0, 0) == doctest::Approx(1.0));
    CHECK(tensor(1, 1) == doctest::Approx(2.0));
    CHECK(tensor(2, 2) == doctest::Approx(3.0));

    // Verify off-diagonal (symmetric)
    CHECK(tensor(1, 2) == doctest::Approx(4.0));
    CHECK(tensor(2, 1) == doctest::Approx(4.0));
    CHECK(tensor(0, 2) == doctest::Approx(5.0));
    CHECK(tensor(2, 0) == doctest::Approx(5.0));
    CHECK(tensor(0, 1) == doctest::Approx(6.0));
    CHECK(tensor(1, 0) == doctest::Approx(6.0));
}

TEST_CASE("RoundTripConversion") {
    dp::mat::Matrix<double, 3, 3> storage;
    simd::Matrix<double, 3, 3> original(storage);
    original(0, 0) = 10.0;
    original(0, 1) = 12.0;
    original(0, 2) = 13.0;
    original(1, 0) = 12.0;
    original(1, 1) = 20.0;
    original(1, 2) = 23.0;
    original(2, 0) = 13.0;
    original(2, 1) = 23.0;
    original(2, 2) = 30.0;

    auto voigt = simd::to_voigt(original);
    auto recovered = simd::from_voigt(voigt);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(recovered(i, j) == doctest::Approx(original(i, j)));
        }
    }
}

TEST_CASE("EngineeringStrainConversion") {
    // Engineering strains have shear components doubled
    dp::mat::Vector<double, 6> storage;
    simd::Vector<double, 6> eng_strain(storage);
    eng_strain[0] = 0.001;
    eng_strain[1] = 0.002;
    eng_strain[2] = 0.003;
    eng_strain[3] = 0.004;
    eng_strain[4] = 0.005;
    eng_strain[5] = 0.006;

    auto tensor = simd::strain_from_voigt_engineering(eng_strain);

    // Verify diagonal (normal strains unchanged)
    CHECK(tensor(0, 0) == doctest::Approx(0.001));
    CHECK(tensor(1, 1) == doctest::Approx(0.002));
    CHECK(tensor(2, 2) == doctest::Approx(0.003));

    // Verify shear strains (divided by 2)
    CHECK(tensor(1, 2) == doctest::Approx(0.002));  // γ23 / 2
    CHECK(tensor(0, 2) == doctest::Approx(0.0025)); // γ13 / 2
    CHECK(tensor(0, 1) == doctest::Approx(0.003));  // γ12 / 2

    // Convert back
    auto recovered = simd::strain_to_voigt_engineering(tensor);
    for (size_t i = 0; i < 6; ++i) {
        CHECK(recovered[i] == doctest::Approx(eng_strain[i]).epsilon(1e-10));
    }
}

TEST_CASE("ElasticityTensorConversion") {
    // Create a simple test 4th-order tensor with known values
    double C[3][3][3][3] = {{{{0}}}};

    // Set diagonal components
    C[0][0][0][0] = 2.0;
    C[1][1][1][1] = 2.0;
    C[2][2][2][2] = 2.0;

    auto voigt_C = simd::elasticity_to_voigt(C);

    // Verify the function runs and produces correct dimensions
    CHECK(voigt_C.rows() == 6);
    CHECK(voigt_C.cols() == 6);

    // Verify basic diagonal values were transferred
    CHECK(voigt_C(0, 0) == doctest::Approx(2.0));
    CHECK(voigt_C(1, 1) == doctest::Approx(2.0));
    CHECK(voigt_C(2, 2) == doctest::Approx(2.0));
}

// =============================================================================
// Test 4: get<I>(pack) - Compile-time Lane Extraction
// =============================================================================

#ifdef OPTINUM_HAS_SSE2
TEST_CASE("GetFloatPack_SSE") {
    // Test with pack<float, 4> (SSE width)
    alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto p = simd::pack<float, 4>::load(data);

    // Extract lanes at compile time using get<I>()
    CHECK(simd::get<0>(p) == doctest::Approx(1.0f));
    CHECK(simd::get<1>(p) == doctest::Approx(2.0f));
    CHECK(simd::get<2>(p) == doctest::Approx(3.0f));
    CHECK(simd::get<3>(p) == doctest::Approx(4.0f));
}

TEST_CASE("GetDoublePack_SSE") {
    // Test with pack<double, 2> (SSE width)
    alignas(16) double data[2] = {10.0, 20.0};
    auto p = simd::pack<double, 2>::load(data);

    CHECK(simd::get<0>(p) == doctest::Approx(10.0));
    CHECK(simd::get<1>(p) == doctest::Approx(20.0));
}
#endif // OPTINUM_HAS_SSE2

#ifdef OPTINUM_HAS_AVX
TEST_CASE("GetFloatPack_AVX") {
    // Test with pack<float, 8> (AVX width)
    alignas(32) float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto p = simd::pack<float, 8>::load(data);

    CHECK(simd::get<0>(p) == doctest::Approx(1.0f));
    CHECK(simd::get<1>(p) == doctest::Approx(2.0f));
    CHECK(simd::get<7>(p) == doctest::Approx(8.0f));
}

TEST_CASE("GetDoublePack_AVX") {
    // Test with pack<double, 4> (AVX width)
    alignas(32) double data[4] = {10.0, 20.0, 30.0, 40.0};
    auto p = simd::pack<double, 4>::load(data);

    CHECK(simd::get<0>(p) == doctest::Approx(10.0));
    CHECK(simd::get<3>(p) == doctest::Approx(40.0));
}
#endif // OPTINUM_HAS_AVX

// Test with scalar fallback (always available)
TEST_CASE("GetScalarPack") {
    alignas(16) float data[2] = {42.0f, 43.0f};
    auto p = simd::pack<float, 2>::load(data);

    // Scalar fallback uses operator[] for get<>
    CHECK(p[0] == doctest::Approx(42.0f));
    CHECK(p[1] == doctest::Approx(43.0f));
}

#ifdef OPTINUM_HAS_SSE2
TEST_CASE("GetAfterArithmetic") {
    alignas(16) float data_a[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    alignas(16) float data_b[4] = {0.0f, 2.0f, 4.0f, 6.0f};

    auto a = simd::pack<float, 4>::load(data_a);
    auto b = simd::pack<float, 4>::load(data_b);
    auto c = a + b;

    CHECK(simd::get<0>(c) == doctest::Approx(0.0f));
    CHECK(simd::get<1>(c) == doctest::Approx(3.0f)); // 1 + 2
    CHECK(simd::get<2>(c) == doctest::Approx(6.0f)); // 2 + 4
    CHECK(simd::get<3>(c) == doctest::Approx(9.0f)); // 3 + 6
}
#endif // OPTINUM_HAS_SSE2

// =============================================================================
// Integration Test: All Features Together
// =============================================================================

TEST_CASE("CombinedFeaturesWorkflow") {
    // Create stress tensor in Voigt notation
    dp::mat::Vector<double, 6> voigt_storage;
    simd::Vector<double, 6> stress_voigt(voigt_storage);
    stress_voigt[0] = 100.0;
    stress_voigt[1] = 200.0;
    stress_voigt[2] = 150.0;
    stress_voigt[3] = 50.0;
    stress_voigt[4] = 30.0;
    stress_voigt[5] = 40.0;

    // Convert to full tensor
    auto stress_tensor = simd::from_voigt(stress_voigt);

    // Apply transformation (simple scaling) using backend
    dp::mat::Matrix<double, 3, 3> result_storage;
    simd::Matrix<double, 3, 3> result(result_storage);
    simd::backend::mul_scalar<double, 9>(result.data(), stress_tensor.data(), 2.0);

    // Convert back to Voigt
    auto result_voigt = simd::to_voigt(result);

    // Verify
    for (size_t i = 0; i < 6; ++i) {
        CHECK(result_voigt[i] == doctest::Approx(stress_voigt[i] * 2.0));
    }

    // Export to row-major for external library
    double row_major_array[9];
    simd::copy_to_rowmajor(result, row_major_array);

    // Verify first element
    CHECK(row_major_array[0] == doctest::Approx(200.0)); // 100 * 2
}
