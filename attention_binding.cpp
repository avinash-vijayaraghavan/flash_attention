#include <torch/extension.h>

// Function declaration
at::Tensor permute(const at::Tensor &input, bool t0213);
at::Tensor permute_back(const at::Tensor &input, bool t0213);
at::Tensor matmul(const at::Tensor &input, const at::Tensor &weight);
at::Tensor batch_mm_xy(const at::Tensor &x, const at::Tensor &y);
at::Tensor max(at::Tensor &x);
at::Tensor softmax(at::Tensor &x);
at::Tensor attention_forward1(at::Tensor &q, at::Tensor &k, at::Tensor &v, bool causal, bool scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("permute", &permute, "Custom CUDA permute");
    m.def("permute_back", &permute_back, "Custom CUDA permute back");
    m.def("matmul", &matmul, "Cuda GEMM matmul");
    m.def("batch_matmul", &batch_mm_xy);
    m.def("max", &max);
    m.def("softmax", &softmax);
    m.def("attention_forward", &attention_forward1);
}
