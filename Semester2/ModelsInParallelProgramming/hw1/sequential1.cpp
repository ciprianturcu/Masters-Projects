#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using matrix = std::vector<std::vector<double>>;

matrix readMatrixFromBinary(const std::string &filename, uint32_t n) {
    matrix mat(n, std::vector<double>(n));
    std::ifstream in(filename, std::ios::binary);
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

matrix multiplyMatrices(const matrix &A, const matrix &B) {
    uint32_t n = A.size();
    matrix C(n, std::vector(n, 0.0));
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            for (uint32_t k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Write an n x n matrix to a binary file (row by row).
void writeMatrixToBinary(const std::string &filename, const matrix &mat) {
    std::ofstream out(filename, std::ios::binary);
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

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <filenameMatrix1> <filenameMatrix2> <filenameMatrix3>\n";
        return 1;
    }

    // Parse command-line arguments
    uint32_t n = std::stoi(argv[1]);
    std::string filenameMatrix1 = argv[2];
    std::string filenameMatrix2 = argv[3];
    std::string filenameMatrix3 = argv[4];

    // Start total timer
    auto t_total_start = std::chrono::high_resolution_clock::now();

    // Read matrices
    auto t_reading_start = std::chrono::high_resolution_clock::now();
    matrix A = readMatrixFromBinary(filenameMatrix1, n);
    matrix B = readMatrixFromBinary(filenameMatrix2, n);
    auto t_reading_end = std::chrono::high_resolution_clock::now();
    auto t_reading = std::chrono::duration_cast<std::chrono::milliseconds>(
                         t_reading_end - t_reading_start)
                         .count();

    // Multiply
    auto t_multiplication_start = std::chrono::high_resolution_clock::now();
    matrix C = multiplyMatrices(A, B);
    auto t_multiplication_end = std::chrono::high_resolution_clock::now();
    auto t_multiplication = std::chrono::duration_cast<std::chrono::milliseconds>(
                                t_multiplication_end - t_multiplication_start)
                                .count();

    // Write result
    auto t_writing_start = std::chrono::high_resolution_clock::now();
    writeMatrixToBinary(filenameMatrix3, C);
    auto t_writing_end = std::chrono::high_resolution_clock::now();
    auto t_writing = std::chrono::duration_cast<std::chrono::milliseconds>(
                         t_writing_end - t_writing_start)
                         .count();

    // End total timer
    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto t_total = std::chrono::duration_cast<std::chrono::milliseconds>(
                       t_total_end - t_total_start)
                       .count();

    // Print a single CSV line: n,t_reading,t_multiplication,t_writing,t_total
    std::cout << n << ","
              << t_reading << ","
              << t_multiplication << ","
              << t_writing << ","
              << t_total
              << std::endl;

    return 0;
}
