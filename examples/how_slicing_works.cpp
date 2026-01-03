// =============================================================================
// examples/how_slicing_works.cpp
// Explains how slicing works - NO data copying, just pointer arithmetic!
// =============================================================================

#include <datapod/datapod.hpp>
#include <iostream>
#include <optinum/optinum.hpp>
#include <vector>

using namespace optinum;
namespace dp = datapod;

void print_separator() { std::cout << "\n" << std::string(70, '=') << "\n\n"; }

int main() {
    std::cout << "=== HOW SLICING WORKS IN OPTINUM ===\n";
    print_separator();

    // =========================================================================
    // PART 1: The Data Ownership Model
    // =========================================================================
    std::cout << "PART 1: Data Ownership - Views Don't Own Data!\n\n";

    // Data is owned by datapod types (dp::mat::vector, matrix, tensor)
    dp::mat::Vector<float, 10> owned_data;
    for (int i = 0; i < 10; i++) {
        owned_data[i] = static_cast<float>(i * 10);
    }

    std::cout << "Original data (owned by dp::mat::Vector<float,10>):\n   ";
    for (int i = 0; i < 10; i++) {
        std::cout << owned_data[i] << " ";
    }
    std::cout << "\n";
    std::cout << "   Memory address: " << (void *)owned_data.data() << "\n\n";

    // Create a view - it's just a lightweight wrapper around the pointer
    vector_view<float, 4> view(owned_data.data(), 10);

    std::cout << "Created vector_view<float, 4>:\n";
    std::cout << "   Points to same address: " << (void *)view.data() << "\n";
    std::cout << "   Size: " << view.size() << " elements\n";
    std::cout << "   NO DATA WAS COPIED - view just holds a pointer!\n";

    print_separator();

    // =========================================================================
    // PART 2: What Slicing Actually Does
    // =========================================================================
    std::cout << "PART 2: Slicing = Pointer Arithmetic (Still No Copying!)\n\n";

    std::cout << "Creating slice: view.slice(seq(3, 7))\n";
    auto sliced = view.slice(seq(3, 7));

    std::cout << "\nWhat happened internally:\n";
    std::cout << "   Original pointer: " << (void *)view.data() << "\n";
    std::cout << "   Slice start index: 3\n";
    std::cout << "   New pointer: " << (void *)view.data() << " + 3*sizeof(float)\n";
    std::cout << "                = " << (void *)sliced.data() << "\n";
    std::cout << "   Slice size: 4 elements (indices 3, 4, 5, 6)\n\n";

    std::cout << "Sliced view data:\n   ";
    for (std::size_t i = 0; i < sliced.size(); i++) {
        std::cout << sliced[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "KEY INSIGHT: The slice is just a NEW VIEW with:\n";
    std::cout << "   - Adjusted pointer (points into original data)\n";
    std::cout << "   - New size (4 elements instead of 10)\n";
    std::cout << "   - NO DATA COPYING AT ALL!\n";

    print_separator();

    // =========================================================================
    // PART 3: Modifying Through Slices
    // =========================================================================
    std::cout << "PART 3: Modifying Through Slices\n\n";

    std::cout << "Original data before modification:\n   ";
    for (int i = 0; i < 10; i++) {
        std::cout << owned_data[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Modifying through slice: sliced[1] = 999\n";
    sliced[1] = 999.0f;

    std::cout << "\nOriginal data after modification:\n   ";
    for (int i = 0; i < 10; i++) {
        std::cout << owned_data[i] << " ";
    }
    std::cout << "\n";
    std::cout << "   ↑ Notice element 4 changed to 999!\n\n";

    std::cout << "CONCLUSION: Slices share the same underlying memory.\n";
    std::cout << "            Changes through a slice affect the original data.\n";

    print_separator();

    // =========================================================================
    // PART 4: Strided Slicing (Step > 1)
    // =========================================================================
    std::cout << "PART 4: Strided Slicing\n\n";

    // Reset data
    for (int i = 0; i < 10; i++) {
        owned_data[i] = static_cast<float>(i * 10);
    }

    std::cout << "Original: ";
    for (int i = 0; i < 10; i++) {
        std::cout << owned_data[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Creating strided slice: view.slice(seq(0, 10, 3))\n";
    std::cout << "   This picks every 3rd element: indices 0, 3, 6, 9\n\n";

    auto strided = view.slice(seq(0, 10, 3));

    std::cout << "Strided view internal state:\n";
    std::cout << "   Pointer: " << (void *)strided.data() << " (same as original)\n";
    std::cout << "   Size: " << strided.size() << " elements\n";
    std::cout << "   Stride: 3 (jumps 3 elements each time)\n\n";

    std::cout << "Strided view data:\n   ";
    for (std::size_t i = 0; i < strided.size(); i++) {
        std::cout << strided[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "How stride works:\n";
    std::cout << "   strided[0] → data[0 + 0*3] = data[0] = " << strided[0] << "\n";
    std::cout << "   strided[1] → data[0 + 1*3] = data[3] = " << strided[1] << "\n";
    std::cout << "   strided[2] → data[0 + 2*3] = data[6] = " << strided[2] << "\n";
    std::cout << "   strided[3] → data[0 + 3*3] = data[9] = " << strided[3] << "\n";

    print_separator();

    // =========================================================================
    // PART 5: Matrix Slicing (2D)
    // =========================================================================
    std::cout << "PART 5: Matrix Slicing (2D)\n\n";

    std::vector<float> mat_data(20);
    for (int i = 0; i < 20; i++) {
        mat_data[i] = static_cast<float>(i);
    }

    matrix_view<float, 4> mat(mat_data.data(), 4, 5);

    std::cout << "Original 4x5 matrix (column-major):\n";
    for (std::size_t r = 0; r < 4; r++) {
        std::cout << "   ";
        for (std::size_t c = 0; c < 5; c++) {
            std::cout << mat(r, c) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "Memory layout (column-major):\n";
    std::cout << "   Linear: ";
    for (int i = 0; i < 20; i++) {
        std::cout << mat_data[i] << " ";
    }
    std::cout << "\n";
    std::cout << "   element(r,c) is at: data[r + c*rows]\n\n";

    std::cout << "Slicing: mat.slice(seq(1,3), seq(2,5))\n";
    std::cout << "   Rows 1..2, Columns 2..4\n\n";

    auto mat_sliced = mat.slice(seq(1, 3), seq(2, 5));

    std::cout << "Slice calculation:\n";
    std::cout << "   row_start = 1, col_start = 2\n";
    std::cout << "   offset = row_start + col_start * rows = 1 + 2*4 = 9\n";
    std::cout << "   new_pointer = mat.data() + 9\n";
    std::cout << "   new_dimensions = (2 rows, 3 cols)\n\n";

    std::cout << "Sliced " << mat_sliced.rows() << "x" << mat_sliced.cols() << " matrix:\n";
    for (std::size_t r = 0; r < mat_sliced.rows(); r++) {
        std::cout << "   ";
        for (std::size_t c = 0; c < mat_sliced.cols(); c++) {
            std::cout << mat_sliced(r, c) << "\t";
        }
        std::cout << "\n";
    }

    print_separator();

    // =========================================================================
    // PART 6: Dimensionality Reduction (NEW!)
    // =========================================================================
    std::cout << "PART 6: Dimensionality Reduction with fix<N>\n\n";

    std::cout << "When you use fix<N>, you're selecting a SINGLE index,\n";
    std::cout << "which reduces the rank by 1 for each fix<>.\n\n";

    std::cout << "Example 1: Extract a single row as vector\n";
    std::cout << "   mat.slice(fix<2>(), all) → extract row 2\n\n";

    auto row_vec = mat.slice(fix<2>(), all);
    std::cout << "   Result type: vector_view (2D → 1D)\n";
    std::cout << "   Size: " << row_vec.size() << " elements\n";
    std::cout << "   Data: ";
    for (std::size_t i = 0; i < row_vec.size(); i++) {
        std::cout << row_vec[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Example 2: 3D Tensor → 2D Matrix\n";
    std::vector<float> tensor_data(24);
    for (int i = 0; i < 24; i++) {
        tensor_data[i] = static_cast<float>(i);
    }
    tensor_view<float, 4, 3> tensor(tensor_data.data(), {2, 3, 4}, {12, 4, 1});

    std::cout << "   tensor.slice(fix<0>(), all, all) → fix first dimension\n";
    auto tensor_to_mat = tensor.slice(fix<0>(), all, all);
    std::cout << "   Result type: matrix_view (3D → 2D)\n";
    std::cout << "   Dimensions: " << tensor_to_mat.rows() << "x" << tensor_to_mat.cols() << "\n\n";

    std::cout << "Example 3: 3D Tensor → 1D Vector\n";
    std::cout << "   tensor.slice(fix<0>(), fix<1>(), all) → fix two dimensions\n";
    auto tensor_to_vec = tensor.slice(fix<0>(), fix<1>(), all);
    std::cout << "   Result type: vector_view (3D → 1D)\n";
    std::cout << "   Size: " << tensor_to_vec.size() << " elements\n";

    print_separator();

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "SUMMARY: How Slicing Works\n\n";

    std::cout << "1. Views are NON-OWNING lightweight wrappers:\n";
    std::cout << "   - Hold a pointer to data\n";
    std::cout << "   - Store dimensions and strides\n";
    std::cout << "   - Data is owned by dp::mat::{vector,matrix,tensor}\n\n";

    std::cout << "2. Slicing creates a NEW VIEW (not a copy):\n";
    std::cout << "   - Calculates new pointer: original_ptr + offset\n";
    std::cout << "   - Stores new dimensions (sliced size)\n";
    std::cout << "   - Stores new strides (for strided slicing)\n\n";

    std::cout << "3. NO DATA IS EVER COPIED:\n";
    std::cout << "   - All views point into the same memory\n";
    std::cout << "   - Modifications affect the original data\n";
    std::cout << "   - This is ultra-fast and memory-efficient!\n\n";

    std::cout << "4. Dimensionality Reduction:\n";
    std::cout << "   - fix<N> selects a single index → reduces rank by 1\n";
    std::cout << "   - tensor[fix, fix, all] → vector\n";
    std::cout << "   - tensor[fix, all, all] → matrix\n";
    std::cout << "   - matrix[fix, all] → vector\n";

    print_separator();

    return 0;
}
