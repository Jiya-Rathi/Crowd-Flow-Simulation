// crowd_conv.cpp
#include <torch/extension.h>

void conv2d_cuda(torch::Tensor input, torch::Tensor kernel, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_cuda", &conv2d_cuda, "Custom CUDA Conv2D");
}
