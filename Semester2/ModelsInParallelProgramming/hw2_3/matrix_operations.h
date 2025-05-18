#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// Define matrix as a 2D vector of doubles.
typedef std::vector<std::vector<double>> matrix;

// ===================
// Sequential Functions
// ===================

// Reads an n x n matrix from a binary file.
matrix readMatrixFromBinary(const std::string &filename, uint32_t n) {
    matrix mat(n, std::vector<double>(n));
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in) {
        std::cerr << "Error opening file " << filename << std::endl;
        std::exit(1);
    }
    for (uint32_t i = 0; i < n; ++i) {
        in.read(reinterpret_cast<char*>(mat[i].data()), n * sizeof(double));
    }
    in.close();
    return mat;
}

// Multiplies two matrices sequentially.
matrix multiplyMatrices(const matrix &A, const matrix &B) {
    uint32_t n = A.size();
    matrix C(n, std::vector<double>(n, 0.0));
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j)
            for (uint32_t k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Writes an n x n matrix to a binary file.
void writeMatrixToBinary(const std::string &filename, const matrix &mat) {
    std::ofstream out(filename.c_str(), std::ios::binary);
    if (!out) {
        std::cerr << "Error opening file " << filename << std::endl;
        std::exit(1);
    }
    uint32_t n = mat.size();
    for (uint32_t i = 0; i < n; ++i) {
        out.write(reinterpret_cast<const char*>(mat[i].data()), n * sizeof(double));
    }
    out.close();
}

// =============================================
// Parallel Functions Using Explicit std::thread
// =============================================

// Helper lambda function is defined within the function (C++11 style).
// This function reads rows [start, end) from a binary file.
matrix readMatrixFromBinaryParallel(const std::string &filename, uint32_t n, int num_threads) {
    matrix mat(n, std::vector<double>(n));
    // Lambda to read a block of rows.
    auto readRows = [&filename, n, &mat](uint32_t start, uint32_t end) {
        std::ifstream in(filename.c_str(), std::ios::binary);
        if (!in) {
            std::cerr << "Error opening file " << filename << std::endl;
            std::exit(1);
        }
        // Seek to the start row.
        in.seekg(start * n * sizeof(double), std::ios::beg);
        for (uint32_t i = start; i < end; ++i) {
            in.read(reinterpret_cast<char*>(mat[i].data()), n * sizeof(double));
        }
        in.close();
    };

    std::vector<std::thread> threads;
    uint32_t rows_per_thread = n / num_threads;
    uint32_t remainder = n % num_threads;
    uint32_t start = 0;
    for (int i = 0; i < num_threads; i++) {
        uint32_t end = start + rows_per_thread + (i < static_cast<int>(remainder) ? 1 : 0);
        threads.push_back(std::thread(readRows, start, end));
        start = end;
    }
    for (std::size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
    return mat;
}

// Helper function for parallel multiplication: computes rows [start, end) of the result matrix.
void multiplyPartial(const matrix &A, const matrix &B, matrix &C, uint32_t start, uint32_t end) {
    uint32_t n = A.size();
    for (uint32_t i = start; i < end; ++i)
        for (uint32_t j = 0; j < n; ++j)
            for (uint32_t k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

// Multiplies two matrices in parallel using explicit threads.
matrix multiplyMatricesParallelExplicit(const matrix &A, const matrix &B, int num_threads) {
    uint32_t n = A.size();
    matrix C(n, std::vector<double>(n, 0.0));
    std::vector<std::thread> threads;
    uint32_t rows_per_thread = n / num_threads;
    uint32_t remainder = n % num_threads;
    uint32_t start = 0;
    for (int i = 0; i < num_threads; i++) {
        uint32_t end = start + rows_per_thread + (i < static_cast<int>(remainder) ? 1 : 0);
        threads.push_back(std::thread(multiplyPartial, std::cref(A), std::cref(B), std::ref(C), start, end));
        start = end;
    }
    for (std::size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
    return C;
}

// =============================================
// Parallel Functions Using OpenMP
// =============================================

// Reads an n x n matrix from a binary file in parallel using OpenMP.
matrix readMatrixFromBinaryOpenMP(const std::string &filename, uint32_t n, int num_threads) {
    matrix mat(n, std::vector<double>(n));
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (uint32_t i = 0; i < n; ++i) {
        std::ifstream in(filename.c_str(), std::ios::binary);
        if (!in) {
            std::cerr << "Error opening file " << filename << std::endl;
            std::exit(1);
        }
        in.seekg(i * n * sizeof(double), std::ios::beg);
        in.read(reinterpret_cast<char*>(mat[i].data()), n * sizeof(double));
        in.close();
    }
#else
    // If OpenMP is not enabled, fallback to sequential reading.
    mat = readMatrixFromBinary(filename, n);
#endif
    return mat;
}

// Multiplies two matrices in parallel using OpenMP.
matrix multiplyMatricesParallelOpenMP(const matrix &A, const matrix &B, int num_threads) {
    uint32_t n = A.size();
    matrix C(n, std::vector<double>(n, 0.0));
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            for (uint32_t k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
#else
    // If OpenMP is not enabled, fallback to sequential multiplication.
    C = multiplyMatrices(A, B);
#endif
    return C;
}

#endif // MATRIX_OPERATIONS_H
