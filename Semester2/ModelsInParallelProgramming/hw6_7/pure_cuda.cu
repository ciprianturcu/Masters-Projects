// pure_cuda.cu
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

// Timer for measuring stages
class Timer {
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsedMilliseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start;
};

using Matrix = std::vector<std::vector<double>>;

// Helpers to flatten/unflatten
static std::vector<double> flatten(const Matrix& M) {
    int r = M.size(), c = r ? M[0].size() : 0;
    std::vector<double> F(r * c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            F[i*c + j] = M[i][j];
    return F;
}

static Matrix unflatten(const std::vector<double>& F, int r, int c) {
    Matrix M(r, std::vector<double>(c));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            M[i][j] = F[i*c + j];
    return M;
}

// Allocate zeroed matrix
static Matrix allocateMatrix(int rows, int cols) {
    return Matrix(rows, std::vector<double>(cols, 0.0));
}

// Sequential binary read
static Matrix readMatrixSequential(const std::string& fn, int rows, int cols) {
    Matrix M = allocateMatrix(rows, cols);
    std::ifstream in(fn, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << fn << "\n"; std::exit(1); }
    for (int i = 0; i < rows; ++i)
        in.read(reinterpret_cast<char*>(M[i].data()), cols * sizeof(double));
    return M;
}

// Sequential binary write
static void writeMatrixSequential(const Matrix& mat, const std::string& filename) {
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

// CUDA kernel: C = A×B
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
            sum += A[row*n + k] * B[k*n + col];
        C[row*n + col] = sum;
    }
}

// GPU multiply
static Matrix multiplyCUDA(const Matrix& A, const Matrix& B, int n, int bs) {
    // Separate declarations to avoid 'auto' ambiguity
    std::vector<double> hA = flatten(A);
    std::vector<double> hB = flatten(B);
    std::vector<double> hC(n * n);

    double *dA, *dB, *dC;
    size_t bytes = size_t(n) * n * sizeof(double);

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

    dim3 th(bs, bs), gr((n + bs - 1) / bs, (n + bs - 1) / bs);
    matrixMulKernel<<<gr, th>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return unflatten(hC, n, n);
}

int main(int argc, char** argv) {
    std::cout<<"\n"<<argv<<"\n";
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <A.bin> <B.bin> <C.bin> <blockSize>\n";
        return 1;
    }
    int         n         = std::stoi(argv[1]);
    std::string fileA     = argv[2];
    std::string fileB     = argv[3];
    std::string fileC     = argv[4];
    int         blockSize = std::stoi(argv[5]);

    Timer totalTimer, stage;
    double t_read, t_mult, t_write;

    // Read
    stage.reset();
    Matrix A = readMatrixSequential(fileA, n, n);
    Matrix B = readMatrixSequential(fileB, n, n);
    t_read = stage.elapsedMilliseconds();

    // Multiply
    stage.reset();
    Matrix C = multiplyCUDA(A, B, n, blockSize);
    t_mult = stage.elapsedMilliseconds();

    // Write
    stage.reset();
    writeMatrixSequential(C, fileC);
    t_write = stage.elapsedMilliseconds();

    double t_total = totalTimer.elapsedMilliseconds();
    std::cout << n << "," << t_read << "," << t_mult
              << "," << t_write << "," << t_total << "\n";

    return 0;
}
