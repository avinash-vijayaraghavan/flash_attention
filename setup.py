from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_cuda",
    ext_modules=[
        # CUDAExtension(
        #     "custom_cuda",
        #     [
        #         "kernels.cu",
        #         "kernels_binding.cpp",
        #     ],
        # ),
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


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query, key, value, dropout_p=0.0, attn_mask=None, is_causal=False, scale=None
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
