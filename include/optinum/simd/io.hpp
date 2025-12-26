#pragma once

// =============================================================================
// optinum/simd/io.hpp
// I/O utilities for Vector, Matrix, Tensor, Complex
// =============================================================================

#include <optinum/simd/complex.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/scalar.hpp>
#include <optinum/simd/tensor.hpp>
#include <optinum/simd/vector.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace optinum::simd {

    // ===== Print functions with formatting =====

    /**
     * @brief Print scalar with formatting
     * @param s Scalar to print
     * @param precision Number of decimal places (default: 6)
     * @param label Optional label to print before value
     */
    template <typename T> inline void print(const Scalar<T> &s, int precision = 6, const std::string &label = "") {
        if (!label.empty()) {
            std::cout << label << ": ";
        }
        std::cout << std::fixed << std::setprecision(precision) << static_cast<T>(s) << "\n";
    }
    /**
     * @brief Print vector with formatting
     * @param v Vector to print
     * @param precision Number of decimal places (default: 6)
     * @param label Optional label to print before vector
     */
    template <typename T, std::size_t N>
    inline void print(const Vector<T, N> &v, int precision = 6, const std::string &label = "") {
        if (!label.empty()) {
            std::cout << label << ":\n";
        }
        std::cout << std::fixed << std::setprecision(precision);
        std::cout << "[";
        for (std::size_t i = 0; i < N; ++i) {
            std::cout << v[i];
            if (i + 1 < N)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }

    /**
     * @brief Print matrix with formatting (column-major layout visualization)
     * @param m Matrix to print
     * @param precision Number of decimal places (default: 6)
     * @param label Optional label to print before matrix
     */
    template <typename T, std::size_t R, std::size_t C>
    inline void print(const Matrix<T, R, C> &m, int precision = 6, const std::string &label = "") {
        if (!label.empty()) {
            std::cout << label << ":\n";
        }
        std::cout << std::fixed << std::setprecision(precision);

        // Find max width for alignment
        int max_width = precision + 4; // For sign, decimal point, and some digits

        std::cout << "[\n";
        for (std::size_t i = 0; i < R; ++i) {
            std::cout << "  [";
            for (std::size_t j = 0; j < C; ++j) {
                std::cout << std::setw(max_width) << m(i, j);
                if (j + 1 < C)
                    std::cout << ", ";
            }
            std::cout << "]";
            if (i + 1 < R)
                std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "]\n";
    }

    /**
     * @brief Print tensor with formatting
     * @param t Tensor to print
     * @param precision Number of decimal places (default: 6)
     * @param label Optional label to print before tensor
     */
    template <typename T, std::size_t... Dims>
    inline void print(const Tensor<T, Dims...> &t, int precision = 6, const std::string &label = "") {
        if (!label.empty()) {
            std::cout << label << ":\n";
        }
        std::cout << std::fixed << std::setprecision(precision);
        std::cout << "Tensor<" << (Dims, ...) << ">:\n";
        std::cout << t; // Use existing operator<<
    }

    /**
     * @brief Print complex array with formatting
     * @param c Complex array to print
     * @param precision Number of decimal places (default: 6)
     * @param label Optional label to print before complex array
     */
    template <typename T, std::size_t N>
    inline void print(const Complex<T, N> &c, int precision = 6, const std::string &label = "") {
        if (!label.empty()) {
            std::cout << label << ":\n";
        }
        std::cout << std::fixed << std::setprecision(precision);
        std::cout << "[";
        for (std::size_t i = 0; i < N; ++i) {
            std::cout << "(" << c[i].real;
            if (c[i].imag >= 0)
                std::cout << "+";
            std::cout << c[i].imag << "i)";
            if (i + 1 < N)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // ===== Write to file functions =====

    /**
     * @brief Write scalar to file
     * @param s Scalar to write
     * @param filename Output filename
     * @param precision Number of decimal places (default: 16 for full precision)
     * @return true if successful, false otherwise
     */
    template <typename T> inline bool write(const Scalar<T> &s, const std::string &filename, int precision = 16) {
        std::ofstream file(filename);
        if (!file.is_open())
            return false;

        file << std::fixed << std::setprecision(precision) << static_cast<T>(s) << "\n";
        file.close();
        return true;
    }
    /**
     * @brief Write vector to file (one value per line)
     * @param v Vector to write
     * @param filename Output filename
     * @param precision Number of decimal places (default: 16)
     * @return true if successful, false otherwise
     */
    template <typename T, std::size_t N>
    inline bool write(const Vector<T, N> &v, const std::string &filename, int precision = 16) {
        std::ofstream file(filename);
        if (!file.is_open())
            return false;

        file << std::fixed << std::setprecision(precision);
        for (std::size_t i = 0; i < N; ++i) {
            file << v[i] << "\n";
        }
        file.close();
        return true;
    }

    /**
     * @brief Write matrix to file (CSV format: rows, comma-separated columns)
     * @param m Matrix to write
     * @param filename Output filename
     * @param precision Number of decimal places (default: 16)
     * @return true if successful, false otherwise
     */
    template <typename T, std::size_t R, std::size_t C>
    inline bool write(const Matrix<T, R, C> &m, const std::string &filename, int precision = 16) {
        std::ofstream file(filename);
        if (!file.is_open())
            return false;

        file << std::fixed << std::setprecision(precision);
        for (std::size_t i = 0; i < R; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                file << m(i, j);
                if (j + 1 < C)
                    file << ",";
            }
            file << "\n";
        }
        file.close();
        return true;
    }

    /**
     * @brief Write complex array to file (one complex number per line: real,imag)
     * @param c Complex array to write
     * @param filename Output filename
     * @param precision Number of decimal places (default: 16)
     * @return true if successful, false otherwise
     */
    template <typename T, std::size_t N>
    inline bool write(const Complex<T, N> &c, const std::string &filename, int precision = 16) {
        std::ofstream file(filename);
        if (!file.is_open())
            return false;

        file << std::fixed << std::setprecision(precision);
        for (std::size_t i = 0; i < N; ++i) {
            file << c[i].real << "," << c[i].imag << "\n";
        }
        file.close();
        return true;
    }

    // ===== Read from file functions =====

    /**
     * @brief Read vector from file (one value per line)
     * @param v Vector to read into
     * @param filename Input filename
     * @return true if successful, false otherwise
     */
    template <typename T, std::size_t N> inline bool read(Vector<T, N> &v, const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        for (std::size_t i = 0; i < N; ++i) {
            if (!(file >> v[i])) {
                file.close();
                return false;
            }
        }
        file.close();
        return true;
    }

    /**
     * @brief Read matrix from file (CSV format: rows, comma-separated columns)
     * @param m Matrix to read into
     * @param filename Input filename
     * @return true if successful, false otherwise
     */
    template <typename T, std::size_t R, std::size_t C>
    inline bool read(Matrix<T, R, C> &m, const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        std::string line;
        for (std::size_t i = 0; i < R; ++i) {
            if (!std::getline(file, line)) {
                file.close();
                return false;
            }

            std::stringstream ss(line);
            std::string value;
            for (std::size_t j = 0; j < C; ++j) {
                if (!std::getline(ss, value, ',')) {
                    file.close();
                    return false;
                }
                m(i, j) = std::stod(value);
            }
        }
        file.close();
        return true;
    }

} // namespace optinum::simd
