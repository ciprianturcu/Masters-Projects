#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "timer.h"
#include "matrix_operations.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -- Argument parsing & usage --
    if (argc < 7 || (std::string(argv[6]) == "hybrid" && argc < 8)) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <n> <A.bin> <B.bin> <C.bin>"
                      << " <read_method:seq,par>"
                      << " <mult_method:seq,mpi,hybrid>"
                      << " [num_threads]\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n = std::stoi(argv[1]);
    std::string fileA = argv[2], fileB = argv[3], fileC = argv[4];
    std::string read_m = argv[5], mult_m = argv[6];
    int num_threads = (mult_m == "hybrid") ? std::stoi(argv[7]) : 1;

    // -- Simple 1D MPI matrix multiply --
    if (mult_m == "mpi") {
        Timer totalTimer, timer;
        double t_read = 0, t_mult = 0, t_write = 0;

        int rows_per_rank = n / size;
        std::vector<double> A_block(rows_per_rank * n);
        std::vector<double> B_flat(n * n);

        // 1) Read & scatter A, read & flatten B on root
        timer.reset();
        if (rank == 0) {
            Matrix A = readFullMatrixFromBinary(fileA, n);
            Matrix B = readFullMatrixFromBinary(fileB, n);

            // Flatten A into row-major
            std::vector<double> A_flat(n * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    A_flat[i * n + j] = A[i][j];
            // Flatten B
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    B_flat[i * n + j] = B[i][j];

            MPI_Scatter(
                A_flat.data(), rows_per_rank * n, MPI_DOUBLE,
                A_block.data(), rows_per_rank * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );
        } else {
            MPI_Scatter(
                nullptr, rows_per_rank * n, MPI_DOUBLE,
                A_block.data(), rows_per_rank * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );
        }
        t_read = timer.elapsedMilliseconds();

        // 2) Broadcast B to all ranks
        timer.reset();
        MPI_Bcast(B_flat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        t_read += timer.elapsedMilliseconds();

        // 3) Local multiply
        timer.reset();
        // Reshape A_block -> localA, B_flat -> Bmat
        Matrix localA(rows_per_rank, std::vector<double>(n));
        Matrix Bmat(n, std::vector<double>(n));
        for (int i = 0; i < rows_per_rank; ++i)
            for (int j = 0; j < n; ++j)
                localA[i][j] = A_block[i * n + j];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Bmat[i][j] = B_flat[i * n + j];

        Matrix Btrans = transposeMatrix(Bmat);
        Matrix C_local = multiplyLocal(localA, Btrans);
        t_mult = timer.elapsedMilliseconds();

        // 4) Gather C blocks
        timer.reset();
        std::vector<double> C_block(rows_per_rank * n);
        for (int i = 0; i < rows_per_rank; ++i)
            for (int j = 0; j < n; ++j)
                C_block[i * n + j] = C_local[i][j];

        std::vector<double> C_flat;
        if (rank == 0) C_flat.resize(n * n);
        MPI_Gather(
            C_block.data(), rows_per_rank * n, MPI_DOUBLE,
            rank == 0 ? C_flat.data() : nullptr,
            rows_per_rank * n, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
        t_write = timer.elapsedMilliseconds();

        // 5) Write result on root
        if (rank == 0) {
            Matrix C(n, std::vector<double>(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    C[i][j] = C_flat[i * n + j];
            std::ofstream out(fileC, std::ios::binary);
            for (auto &row : C)
                out.write(reinterpret_cast<const char*>(row.data()),
                          row.size() * sizeof(double));
        }

        double t_total = totalTimer.elapsedMilliseconds();
        if (rank == 0)
            std::cout << n << "," << t_read << "," << t_mult
                      << "," << t_write << "," << t_total << "\n";

        MPI_Finalize();
        return 0;
    }

    // -- Original 2D-block code for seq/hybrid methods --
    // Create 2D Cartesian communicator
    int dims[2] = {0, 0}, periods[2] = {1, 1};
    MPI_Dims_create(size, 2, dims);
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cartComm);

    Timer totalTimer, timer;
    double t_read = 0, t_mult = 0, t_write = 0;
    Matrix localA, localB, localC;

    // --- Reading stage ---
    totalTimer.reset();
    timer.reset();
    if (read_m == "seq") {
        Matrix A, B;
        if (rank == 0) {
            A = readFullMatrixFromBinary(fileA, n);
            B = readFullMatrixFromBinary(fileB, n);
        }
        scatterMatrix(A, localA, n / dims[0], cartComm);
        scatterMatrix(B, localB, n / dims[0], cartComm);
    } else {
        readMatrixBlocksMPI(fileA, n, localA, cartComm);
        readMatrixBlocksMPI(fileB, n, localB, cartComm);
    }
    t_read = timer.elapsedMilliseconds();

    // --- Multiplication stage ---
    timer.reset();
    if (mult_m == "seq") {
        localC = multiplyLocal(localA, transposeMatrix(localB));
    }
    else if (mult_m == "hybrid") {
        localC = multiplyMatricesHybrid(localA, localB, num_threads);
    }
    else {
        if (rank == 0) std::cerr << "Invalid mult_method\n";
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    t_mult = timer.elapsedMilliseconds();

    // --- Gathering & Writing stage ---
    timer.reset();
    Matrix C;
    gatherMatrix(localC, C, n / dims[0], cartComm);
    if (rank == 0) {
        std::ofstream out(fileC, std::ios::binary);
        for (auto &row : C)
            out.write(reinterpret_cast<const char*>(row.data()),
                      row.size() * sizeof(double));
    }
    t_write = timer.elapsedMilliseconds();

    double t_total = totalTimer.elapsedMilliseconds();
    if (rank == 0)
        std::cout << n << "," << t_read << "," << t_mult
                  << "," << t_write << "," << t_total << "\n";

    MPI_Comm_free(&cartComm);
    MPI_Finalize();
    return 0;
}
