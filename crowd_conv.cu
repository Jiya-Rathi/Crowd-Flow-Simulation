// crowd_conv.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv2d_cuda_kernel(const float* input, const float* kernel, float* output,
                                   int B, int C, int H, int W, int KH, int KW) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;

    if (i >= H || j >= W) return;

    float sum = 0.0f;
    for (int ki = 0; ki < KH; ++ki) {
        for (int kj = 0; kj < KW; ++kj) {
            int ni = i + ki - KH / 2;
            int nj = j + kj - KW / 2;
            if (ni >= 0 && nj >= 0 && ni < H && nj < W) {
                int idx = ((b * C + c) * H + ni) * W + nj;
                int kidx = (c * KH + ki) * KW + kj;
                sum += input[idx] * kernel[kidx];
            }
        }
    }
    output[((b * C + c) * H + i) * W + j] = sum;
}

void conv2d_cuda(torch::Tensor input, torch::Tensor kernel, torch::Tensor output) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int KH = kernel.size(1);
    int KW = kernel.size(2);

    dim3 threads(16, 16);
    dim3 blocks((H + 15) / 16, C, B);

    conv2d_cuda_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(),
        B, C, H, W, KH, KW
    );
}
