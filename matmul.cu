#include <torch/extension.h>

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; ++i)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

void matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16);
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
}