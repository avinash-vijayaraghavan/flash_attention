Implementation of flash attention 2 in Cuda and Triton. There are still some minor bugs, but it was written to learn to implement LLM kernels

Tested on:

    python:     : 3.12
    Triton      : 3.1.0
    CUDA Version: 12.7
    torch       : 2.5.1
    GPU         : NVIDIA GeForce RTX 4050

The notebook  `flash_attention.ipynb` has details on the usage

