#include "timer.h"
#include "matrix_operations.h"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <filenameMatrix1> <filenameMatrix2> <filenameMatrix3> [read_method] [mult_method] [num_threads]\n";
        std::cerr << "read_method: seq, par_thread, par_omp\n";
        std::cerr << "mult_method: seq, par_thread, par_omp\n";
        return 1;
    }

    uint32_t n = std::stoi(argv[1]);
    std::string filenameMatrix1 = argv[2];
    std::string filenameMatrix2 = argv[3];
    std::string filenameMatrix3 = argv[4];

    // Default to sequential if methods not provided.
    std::string read_method = "seq";
    std::string mult_method = "seq";
    int num_threads = 1;
    if (argc >= 6) {
        read_method = argv[5];
    }
    if (argc >= 7) {
        mult_method = argv[6];
    }
    if (argc >= 8) {
        num_threads = std::stoi(argv[7]);
    }

    // Start total timer.
    Timer total_timer;

    // ---------------------
    // Matrix Reading Stage
    // ---------------------
    Timer read_timer;
    matrix A, B;
    if (read_method == "seq") {
        A = readMatrixFromBinary(filenameMatrix1, n);
        B = readMatrixFromBinary(filenameMatrix2, n);
    } else if (read_method == "par_thread") {
        A = readMatrixFromBinaryParallel(filenameMatrix1, n, num_threads);
        B = readMatrixFromBinaryParallel(filenameMatrix2, n, num_threads);
    } else if (read_method == "par_omp") {
        A = readMatrixFromBinaryOpenMP(filenameMatrix1, n, num_threads);
        B = readMatrixFromBinaryOpenMP(filenameMatrix2, n, num_threads);
    } else {
        std::cerr << "Invalid read_method. Use seq, par_thread, or par_omp.\n";
        return 1;
    }
    double t_reading = read_timer.elapsedMilliseconds();

    // --------------------------
    // Matrix Multiplication Stage
    // --------------------------
    Timer mult_timer;
    matrix C;
    if (mult_method == "seq") {
        C = multiplyMatrices(A, B);
    } else if (mult_method == "par_thread") {
        C = multiplyMatricesParallelExplicit(A, B, num_threads);
    } else if (mult_method == "par_omp") {
        C = multiplyMatricesParallelOpenMP(A, B, num_threads);
    } else {
        std::cerr << "Invalid mult_method. Use seq, par_thread, or par_omp.\n";
        return 1;
    }
    double t_multiplication = mult_timer.elapsedMilliseconds();

    // ---------------------
    // Matrix Writing Stage
    // ---------------------
    Timer write_timer;
    writeMatrixToBinary(filenameMatrix3, C);
    double t_writing = write_timer.elapsedMilliseconds();

    double t_total = total_timer.elapsedMilliseconds();

    // Output a CSV line: n,t_reading,t_multiplication,t_writing,t_total
    std::cout << n << ","
              << t_reading << ","
              << t_multiplication << ","
              << t_writing << ","
              << t_total
              << std::endl;

    return 0;
}
