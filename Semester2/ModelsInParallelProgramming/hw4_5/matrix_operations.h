#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <mpi.h>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <algorithm>

using Matrix = std::vector<std::vector<double>>;

// allocate a rows×cols matrix
static Matrix allocateMatrix(int rows, int cols) {
    return Matrix(rows, std::vector<double>(cols));
}

// sequential binary read of an n×n matrix
static Matrix readFullMatrixFromBinary(const std::string &filename, int n) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file " + filename);
    Matrix M = allocateMatrix(n, n);
    for (int i = 0; i < n; ++i)
        in.read(reinterpret_cast<char*>(M[i].data()), n * sizeof(double));
    return M;
}

// scatter an n×n matrix M into local blocks of size blockSize×blockSize on a q×q Cartesian grid
static void scatterMatrix(const Matrix &M, Matrix &localM, int blockSize, MPI_Comm comm) {
    int rank, size, dims[2], periods[2], coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_get(comm, 2, dims, periods, coords);

    std::vector<double> sendbuf, recvbuf(blockSize * blockSize);
    if (rank == 0) {
        sendbuf.resize(size * blockSize * blockSize);
        for (int r = 0; r < size; ++r) {
            MPI_Cart_coords(comm, r, 2, coords);
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j)
                    sendbuf[r*blockSize*blockSize + i*blockSize + j] =
                        M[coords[0]*blockSize + i][coords[1]*blockSize + j];
        }
    }
    MPI_Scatter(
        rank==0 ? sendbuf.data() : nullptr,
        blockSize*blockSize, MPI_DOUBLE,
        recvbuf.data(),
        blockSize*blockSize, MPI_DOUBLE,
        0, comm
    );
    localM = allocateMatrix(blockSize, blockSize);
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            localM[i][j] = recvbuf[i*blockSize + j];
}

// gather local blocks back into a full matrix of size (blockSize*q)×(blockSize*q)
static void gatherMatrix(const Matrix &localM, Matrix &M, int blockSize, MPI_Comm comm) {
    int rank, size, dims[2], periods[2], coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_get(comm, 2, dims, periods, coords);

    std::vector<double> sendbuf(blockSize*blockSize), recvbuf;
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            sendbuf[i*blockSize + j] = localM[i][j];
    if (rank == 0) recvbuf.resize(size * blockSize * blockSize);

    MPI_Gather(
        sendbuf.data(), blockSize*blockSize, MPI_DOUBLE,
        rank==0 ? recvbuf.data() : nullptr,
        blockSize*blockSize, MPI_DOUBLE,
        0, comm
    );
    if (rank == 0) {
        int q = dims[0];
        M = allocateMatrix(blockSize*q, blockSize*q);
        for (int r = 0; r < size; ++r) {
            MPI_Cart_coords(comm, r, 2, coords);
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j)
                    M[coords[0]*blockSize + i][coords[1]*blockSize + j] =
                        recvbuf[r*blockSize*blockSize + i*blockSize + j];
        }
    }
}

// parallel MPI file read of a local block
static void readMatrixBlocksMPI(const std::string &filename, int n, Matrix &localM, MPI_Comm comm) {
    int rank, dims[2], periods[2], coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_get(comm, 2, dims, periods, coords);
    MPI_Cart_coords(comm, rank, 2, coords);
    int q = dims[0], blockSize = n / q;

    MPI_File fh;
    MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Datatype filetype;
    int sizes[2]    = { n, n };
    int subsizes[2] = { blockSize, blockSize };
    int starts[2]   = { coords[0]*blockSize, coords[1]*blockSize };
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    std::vector<double> buf(blockSize*blockSize);
    MPI_File_read_all(fh, buf.data(), blockSize*blockSize, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);

    localM = allocateMatrix(blockSize, blockSize);
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            localM[i][j] = buf[i*blockSize + j];
}

// transpose matrix for cache-friendly multiply
static Matrix transposeMatrix(const Matrix &M) {
    int rows = M.size(), cols = M[0].size();
    Matrix T = allocateMatrix(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            T[j][i] = M[i][j];
    return T;
}

// simple local multiply using transposed B
static Matrix multiplyLocal(const Matrix &A, const Matrix &Btrans) {
    int r = A.size(), c = Btrans.size(), kdim = A[0].size();
    Matrix C = allocateMatrix(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) {
            double sum = 0.0;
            for (int k = 0; k < kdim; ++k)
                sum += A[i][k] * Btrans[j][k];
            C[i][j] = sum;
        }
    return C;
}

// MPI + multithreading hybrid multiply
static Matrix multiplyMatricesHybrid(const Matrix &localA, const Matrix &localB, int num_threads) {
    Matrix Btrans = transposeMatrix(localB);
    int rows = localA.size(), cols = Btrans.size();
    Matrix C = allocateMatrix(rows, cols);

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i)
            for (int j = 0; j < cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < localA[0].size(); ++k)
                    sum += localA[i][k] * Btrans[j][k];
                C[i][j] = sum;
            }
    };

    int chunk = (rows + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end   = std::min(start + chunk, rows);
        if (start < end)
            threads.emplace_back(worker, start, end);
    }
    for (auto &th : threads) th.join();
    return C;
}

#endif
