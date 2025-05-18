#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

void getUserInput(uint32_t &n, string &baseFileName) {
    cout<<"Enter matrix size (n): ";
    cin>>n;
    cout<<"Enter base filename (without extension): ";
    cin>>baseFileName;
}

vector<vector<double>> generateMatrix(const uint32_t n, mt19937 &rng, uniform_real_distribution<> &dist) {
    vector matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = dist(rng);
        }
    }
    return matrix;
}

void writeMatrixToBinaryFile(vector<vector<double>> &matrix,const string& filename) {
    ofstream out(filename.c_str(), ios::binary);
    if (!out) {
        cerr << "Unable to open file " << filename << endl;
        exit(1);
    }

    for (const auto &row : matrix) {
        out.write(reinterpret_cast<const char *>(row.data()), row.size()*sizeof(double));
    }
    out.close();
}

int main() {
    uint32_t n;
    string filename;

    getUserInput(n, filename);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution dis(0.0, 100.0);

    auto matrix1 = generateMatrix(n, gen, dis);

    filename = filename + ".bin";

    writeMatrixToBinaryFile(matrix1, filename);

    cout << "Matrix has been written to " << filename << endl;
    return 0;
}