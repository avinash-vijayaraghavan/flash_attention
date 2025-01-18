from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_cuda",
    ext_modules=[
        CUDAExtension(
            "cuda_attention",
            [
                "attention.cu",
                "attention_binding.cpp",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
