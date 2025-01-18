import torch
import math
import triton
import triton.language as tl
import os

BLOCK_Q = 16
BLOCK_K = 16


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    log2_e = 1.44269504
    eps = 1e-9
    bs, num_heads, seq_len, head_dim = q.shape  # bs, nh, seq_len, head_dim

    q = (
        q.contiguous().view(-1, seq_len, head_dim).to(q.device)
    )  # batch*NH, seq_len, head_dim
    v = v.contiguous().view(-1, seq_len, head_dim).to(q.device)
    k = (k.contiguous().view(-1, seq_len, head_dim).transpose(2, 1)).to(
        q.device
    )  # batch*NH, head_dim, seq_len
    out = torch.zeros_like(q)
    for rows in range(0, seq_len, BLOCK_Q):
        # for b, hi in itertools.product(range(bs), range(num_heads)):
        for bh in range(bs * num_heads):
            qrows = (rows + torch.arange(0, BLOCK_Q)).to(q.device)
            qrows = qrows[qrows < seq_len]  # row mask
            qblock = q[bh][qrows]  # BLOCK_Q, HEAD_DIM_Q
            o = out[bh][qrows]
            running_max = (
                torch.zeros((qblock.shape[0], 1), device=q.device) - torch.inf
            )  # running max for each row
            running_den = torch.zeros((qblock.shape[0], 1), device=q.device)  # running den for each row

            for cols in range(0, seq_len, BLOCK_K):
                kcols = (cols + torch.arange(0, BLOCK_K)).to(q.device)
                kcols = kcols[kcols < seq_len]
                kblock = k[bh][:, kcols]  # HEAD_DIM, BLOCK_K
                vblock = v[bh][kcols]  # BLOCK_k , HEAD_DIM
                scores = qblock.matmul(kblock) / math.sqrt(head_dim)  # BLOCK_Q, BLOCK_K

                # causal attention
                scores = torch.where(
                    qrows[:, None] >= kcols[None, :], scores, -torch.inf
                )
                row_max = torch.maximum(
                    running_max, scores.max(dim=1, keepdim=True)[0]
                )  # BLOCK_Q, 1
                scores -= row_max
                exp_scores = torch.exp(scores)
                sf = torch.exp(running_max - row_max)
                den = (
                    torch.sum(exp_scores, dim=1, keepdim=True) + sf * running_den + eps
                )

                block_prob = exp_scores / den  # BLOCK_Q, BLOCK_K

                # uodate the output rows using the new values
                o = torch.matmul(block_prob, vblock) + o * running_den * sf / den
                running_max = row_max
                running_den = den
            out[bh][qrows] = o
    return out.view(bs, num_heads, seq_len, head_dim)


@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    T,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # q, v : (BS*NH, T, HEAD_DIM)
    # k : (BS*NH, HEAD_DIM, T)

    row_id = tl.program_id(0)
    bh = tl.program_id(1)
    inf = 1e10
    log2_e = 1.44269504
    eps = tl.zeros((BLOCK_Q, 1), dtype=tl.float32) + 1e-9
    qrows = row_id * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
    qcols = tl.arange(0, HEAD_DIM)[None, :]
    offset = bh * T * HEAD_DIM + qrows * HEAD_DIM + qcols
    q_offset = q_ptr + offset
    q_mask = qrows < T
    out_offset = out_ptr + offset
    q = tl.load(q_offset, q_mask, 0.0)  # BLOCK_Q, HEAD_DIM
    z = tl.load(out_offset, q_mask, 0.0)  # BLOCK_Q, HEAD_DIM
    running_max = (
        tl.zeros((BLOCK_Q, 1), dtype=tl.float32) - inf
    )  # running max for each row
    running_den = tl.zeros((BLOCK_Q, 1), dtype=tl.float32)  # running den for each row
    head_dim = tl.zeros((1, 1), dtype=tl.float32) + HEAD_DIM
    for col in range(0, T, BLOCK_K):
        kcols = col + tl.arange(0, BLOCK_K)
        krows = tl.arange(0, HEAD_DIM)[:, None]
        koffset = k_ptr + bh * T * HEAD_DIM + krows * T + kcols[None, :]
        voffset = (
            v_ptr
            + bh * T * HEAD_DIM
            + kcols[:, None] * HEAD_DIM
            + tl.arange(0, HEAD_DIM)[None, :]
        )

        k = tl.load(koffset, kcols[None, :] < T, 0.0)  # HEAD_DIM, BLOCK_K
        v = tl.load(voffset, kcols[:, None] < T, 0.0)  # BLOCK_K, HEAD_DIM

        # current col block
        scores = tl.dot(q, k) / tl.math.sqrt(head_dim)  # BLOCK_Q, BLOCK_K

        # causal attention (if row < col or out of bounds set at -inf )
        scores = tl.where(
            (qrows >= kcols[None, :]) & ((qrows < T) & (kcols < T[None, :])),
            scores,
            -2 * inf,
        )
        row_max = tl.maximum(
            running_max, tl.max(scores, axis=1, keep_dims=True)
        )  # max seen upto this point in the row
        scores -= row_max
        # scores += -inf * ((qrows >= T) | (kcols >= T[None, :]))  # BLOCK_Q, BLOCK_K
        exp_scores = tl.math.exp2(scores * log2_e)

        # update the values
        sf = tl.math.exp2((running_max - row_max) * log2_e)
        den = (
            tl.sum(exp_scores, axis=1, keep_dims=True) + sf * running_den + eps
        )  # BLOCK_Q,1
        block_prob = exp_scores / den
        z = tl.dot(block_prob, v) + sf * z * running_den / den
        running_max = row_max
        running_den = den
    tl.store(out_offset, z, q_mask)


def triton_flash_attention(qp, kp, vp):
    """
    qp : (BS, NH, T, HEAD_DIM)
    kp : (BS, NH, HEAD_DIM, T) This is diffirent from the torch version
    vp : (BS, NH, T, HEAD_DIM)
    """
    batch_size, num_heads, seq_len, head_dim = qp.shape
    device = qp.device
    grid = ((seq_len + BLOCK_Q - 1) // BLOCK_Q, batch_size * num_heads, 1)
    print(f"Grid: {grid}")

    out = torch.zeros_like(qp).to(device)  # bs, nh, seq_len, head_dim

    flash_attention_kernel[grid](
        qp,
        kp,
        vp,
        out,
        seq_len,
        head_dim,
        BLOCK_Q,
        BLOCK_K,
        num_warps=2,
        num_stages=1,
    )  
    return out # bs, nh, seq_len, head_dim

def torch_flash_attention(qp, kp, vp, M, device="cuda"):
    bs, nh, seq_len, head_dim = qp.shape
    p = torch.matmul(qp, kp) / math.sqrt(head_dim)
    p[:, :, M == 0] = torch.tensor([[-torch.inf]]).to(device)

    p = torch.softmax(p.float(), dim=-1)
    ref_out = torch.matmul(p, vp)
    return ref_out


if __name__ == "__main__":
    import timeit

    def flash_attention_example(should_time=False):

        # Example usage
        device = "cpu" if os.environ.get("TRITON_INTERPRET") is not None else "cuda"
        print(device)
        batch_size, seq_len, num_heads, head_dim = 1, 35, 1, 32

        global BLOCK_Q, BLOCK_K
        batch_size, seq_len, num_heads, head_dim = (
            10,
            913,
            24,
            512,
        )  # 5, 651, 11, 512  # 2, 5, 1, 5
        print(
            f"batch_size: {batch_size}, seq_len: {seq_len}, num_heads: {num_heads}, head_dim: {head_dim}, BLOCK_Q: {BLOCK_Q}, BLOCK_K: {BLOCK_K}"
        )

        # Create sample inputs (on cpu)
        torch.random.manual_seed(1)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # still on cpu
        qp = q.permute(0, 2, 1, 3).contiguous()  # bs, nh, seq_len, head_dim
        kp = k.permute(0, 2, 1, 3).contiguous()
        vp = v.permute(0, 2, 1, 3).contiguous()

        # Compute attention
        print("-----------------CPU Flash Attention -----------------")
        output = flash_attention(qp, kp, vp).transpose(2, 1)

        atol = 1e-3
        rtol = 1e-4

        qp = qp.to(device)
        kp = kp.to(device).transpose(2, 3).contiguous()
        vp = vp.to(device)
        if os.environ.get("TRITON_INTERPRET") is None:
            assert device == "cuda"
            atol = 1e-2  # default 1e-2
            rtol = 1e-3  # default 1e-3

        M = torch.tril(torch.ones((seq_len, seq_len), device=device))
        ref_out = (
            torch_flash_attention(qp, kp, vp, M, device)
            .transpose(2, 1)
            .contiguous()
            .to("cpu")
        )
        assert torch.allclose(
            output, ref_out, atol=atol, rtol=rtol
        ), f"NOT MATCHING torch_flash_attention AND flash_attention\nflash_attention:\n{output[0,:5,0,:5]}\ntorch_flash_attention:\n{ref_out[0,:5,0,:5]}\nTry with larger atol: {atol} and rtol: {rtol}"
        print(
            f"MATCHING torch_flash_attention AND flash_attention with atol: {atol} and rtol: {rtol}"
        )
        if should_time:
            print(f"Running torch flash attention on {qp.device, kp.device, vp.device}")
            torch_flash_attentiion_timer = timeit.Timer(
                lambda: torch_flash_attention(qp, kp, vp, M, device)
            )
            print(torch_flash_attentiion_timer.timeit(number=50))

        print("-----------------CUDA Flash Attention -----------------")
        out = triton_flash_attention(qp, kp, vp)
        out = out.transpose(1, 2).to("cpu")
        assert torch.allclose(
            out, ref_out, atol=atol, rtol=rtol
        ), f"NOT MATCHING triton_flash_attention AND torch_flash_attention\ntriton_flash_attention:\n{out[0,:5,0,:5]}\ntorch_flash_attention:\n{ref_out[0,:5,0,:5]}\nTry with larger atol: {atol} and rtol: {rtol}"
        print(
            f"MATCHING torch_flash_attention AND flash_attention_kernel with atol: {atol} and rtol: {rtol}"
        )
        # if should_time:
        #     print(
        #         f"Running custom flash attention on {qp.device, kp.device, vp.device}"
        #     )
        #     triton_flash_attentiion_timer = timeit.Timer(
        #         lambda: (
        #             (out := torch.zeros_like(qp).to(qp.device)),
        #             (
        #                 flash_attention_kernel[grid](
        #                     qp,
        #                     kp,
        #                     vp,
        #                     out,
        #                     seq_len,
        #                     head_dim,
        #                     BLOCK_Q,
        #                     BLOCK_K,
        #                     num_warps=2,
        #                     num_stages=1,
        #                 )
        #             ),
        #         )
        #     )
        #     print(triton_flash_attentiion_timer.timeit(number=50))

    configs = []

    def benchmark(device="cuda"):
        atol = 1e-2
        rtol = 1e-3

        for source in ["torch", "triton"]:

            configs.append(
                triton.testing.Benchmark(
                    x_names=["M"],  # Argument names to use as an x-axis for the plot
                    x_vals=[
                        128 * i for i in range(1, 5)
                    ],  # Different possible values for `x_name`
                    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
                    # Possible values for `line_arg`
                    # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
                    line_vals=(
                        ["triton"] if source == "triton" else ["torch", "triton"]
                    ),  # Label name for the lines
                    line_names=(
                        ["Triton"] if source == "triton" else ["Torch", "Triton"]
                    ),  # Line styles
                    styles=[("green", "-"), ("blue", "-")],
                    ylabel="TFLOPS",  # Label name for the y-axis
                    plot_name="flash-attention-"
                    + (
                        source
                    ),  # Name for the plot, used also as a file name for saving the plot.
                    args={"source": source},
                )
            )

        @triton.testing.perf_report(configs)
        def benchmark(M, provider, source):
            print(M, provider, source)
            batch_size, seq_len, num_heads, head_dim = (
                10,
                M,
                10,
                512,
            )  # 5, 651, 11, 512  # 2, 5, 1, 5
            global BLOCK_K, BLOCK_Q

            BLOCK_Q = 16
            BLOCK_K = 16

            # Create sample inputs
            torch.random.manual_seed(1)
            q = torch.randn(batch_size, seq_len, num_heads, head_dim)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim)

            qp = (
                q.permute(0, 2, 1, 3).contiguous().to(device)
            )  # bs, nh, seq_len, head_dim
            kp = k.permute(0, 2, 1, 3).contiguous().to(device)
            vp = v.permute(0, 2, 1, 3).contiguous().to(device)

            quantiles = [0.5, 0.2, 0.8]
            if provider == "torch":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: torch_flash_attention(qp, kp, vp, device),
                    quantiles=quantiles,
                )
            if provider == "triton":
                grid = ((seq_len + BLOCK_Q - 1) // BLOCK_Q, batch_size * num_heads, 1)
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: (
                        (out := torch.zeros_like(qp).to(qp.device)),
                        (),
                        (
                            flash_attention_kernel[grid](
                                qp,
                                kp,
                                vp,
                                out,
                                seq_len,
                                head_dim,
                                BLOCK_Q,
                                BLOCK_K,
                                num_warps=2,
                                num_stages=1,
                            )
                        ),
                    ),
                    quantiles=quantiles,
                )
            # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
            perf = lambda ms: ms
            return perf(ms), perf(max_ms), perf(min_ms)

        benchmark.run(show_plots=True, print_data=True)

    flash_attention_example()
    # benchmark()
