// mpi_cuda.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>

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

// Helpers
static std::vector<double> flatten(const Matrix& M){
    int r=M.size(), c=r?M[0].size():0;
    std::vector<double> F(r*c);
    for(int i=0;i<r;i++) for(int j=0;j<c;j++) F[i*c+j]=M[i][j];
    return F;
}
static Matrix unflatten(const std::vector<double>& F,int r,int c){
    Matrix M(r, std::vector<double>(c));
    for(int i=0;i<r;i++) for(int j=0;j<c;j++) M[i][j]=F[i*c+j];
    return M;
}
static Matrix allocateMatrix(int rows,int cols){
    return Matrix(rows, std::vector<double>(cols,0.0));
}
static Matrix readMatrixSequential(const std::string& fn,int rows,int cols){
    Matrix M=allocateMatrix(rows,cols);
    std::ifstream in(fn,std::ios::binary);
    if(!in){ if(MPI::COMM_WORLD.Get_rank()==0) std::cerr<<"Cannot open "<<fn<<"\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    for(int i=0;i<rows;i++) in.read(reinterpret_cast<char*>(M[i].data()), cols*sizeof(double));
    return M;
}
static Matrix readMatrixParallel(const std::string& fn,int rows,int cols,MPI_Comm comm){
    std::vector<double> buf(rows*cols);
    MPI_File fh;
    MPI_File_open(comm,fn.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);
    MPI_File_set_view(fh,0,MPI_DOUBLE,MPI_DOUBLE,"native",MPI_INFO_NULL);
    MPI_File_read_all(fh,buf.data(),rows*cols,MPI_DOUBLE,MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    return unflatten(buf,rows,cols);
}
static void writeMatrixSequential(const Matrix& M,const std::string& fn,int rows,int cols){
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0){
        std::ofstream out(fn,std::ios::binary);
        for(int i=0;i<rows;i++)
            out.write(reinterpret_cast<const char*>(M[i].data()), cols*sizeof(double));
    }
}

// CUDA kernel
__global__ void matrixMulKernel(const double* A,const double* B,double* C,int n){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<n && col<n){
        double sum=0;
        for(int k=0;k<n;k++) sum+=A[row*n+k]*B[k*n+col];
        C[row*n+col]=sum;
    }
}

// GPU multiply helper for part 7
static Matrix multiplyMPI_CUDA(const Matrix& A,const Matrix& B,int n,int bs,MPI_Comm comm){
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    std::vector<double> flatA, flatB;
    if(rank==0){ flatA=flatten(A); flatB=flatten(B); }
    else       { flatB.resize(size_t(n)*n); }

    MPI_Bcast(flatB.data(), n*n, MPI_DOUBLE, 0, comm);

    int base=n/size, rem=n%size;
    std::vector<int> cnt(size), disp(size);
    for(int i=0;i<size;i++){
        int rows_i = base + (i<rem?1:0);
        cnt[i]=rows_i*n;
        disp[i]=( (base*i + std::min(i,rem)) * n );
    }
    int lr = base + (rank<rem?1:0);

    std::vector<double> localA(lr*n), localC(lr*n);
    MPI_Scatterv(flatA.data(),cnt.data(),disp.data(),MPI_DOUBLE,
                 localA.data(),cnt[rank],MPI_DOUBLE,0,comm);

    double *dA,*dB,*dC;
    size_t bA=lr*n*sizeof(double), bB=n*n*sizeof(double), bC=lr*n*sizeof(double);
    cudaMalloc(&dA,bA); cudaMalloc(&dB,bB); cudaMalloc(&dC,bC);
    cudaMemcpy(dA, localA.data(), bA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, flatB.data(), bB, cudaMemcpyHostToDevice);

    dim3 th(bs,bs), gr((n+bs-1)/bs,(lr+bs-1)/bs);
    matrixMulKernel<<<gr,th>>>(dA,dB,dC,n);
    cudaDeviceSynchronize();

    cudaMemcpy(localC.data(), dC, bC, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    std::vector<double> flatC;
    if(rank==0) flatC.resize(n*n);
    MPI_Gatherv(localC.data(), lr*n, MPI_DOUBLE,
                flatC.data(), cnt.data(), disp.data(), MPI_DOUBLE,
                0, comm);

    return (rank==0?unflatten(flatC,n,n):Matrix());
}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(argc<7){ 
        if(rank==0) std::cerr<<"Usage: "<<argv[0]
            <<" <n> <A.bin> <B.bin> <C.bin> <blockSize> <seq_read|par_read>\n";
        MPI_Finalize(); return 1;
    }
    int         n         = std::stoi(argv[1]);
    std::string fileA     = argv[2];
    std::string fileB     = argv[3];
    std::string fileC     = argv[4];
    int         bs        = std::stoi(argv[5]);
    std::string mode      = argv[6];

    Timer total, stage;
    double t_read,t_mult,t_write;

    // Read stage
    stage.reset();
    Matrix A = (mode=="par_read")
               ? readMatrixParallel(fileA,n,n,MPI_COMM_WORLD)
               : readMatrixSequential(fileA,n,n);
    Matrix B = (mode=="par_read")
               ? readMatrixParallel(fileB,n,n,MPI_COMM_WORLD)
               : readMatrixSequential(fileB,n,n);
    t_read = stage.elapsedMilliseconds();

    // Multiply stage
    stage.reset();
    Matrix C = multiplyMPI_CUDA(A,B,n,bs,MPI_COMM_WORLD);
    t_mult = stage.elapsedMilliseconds();

    // Write stage
    stage.reset();
    writeMatrixSequential(C,fileC,n,n);
    t_write = stage.elapsedMilliseconds();

    if(rank==0){
        double t_total = total.elapsedMilliseconds();
        std::cout<<n<<","<<t_read<<","<<t_mult<<","<<t_write<<","<<t_total<<"\n";
    }
    MPI_Finalize();
    return 0;
}
