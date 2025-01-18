#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublasLt.h>
#include <torch/extension.h>

#define BLOCK_SIZE 32

#define log2_e (log2(M_E))
// out : will have shape : B, NH, T,D
void permute_cpu(float *inp, int B, int T, int NH, int D)
{

    float *out = (float *)calloc(B * T * NH * D, sizeof(float));
    for (int n = 0; n < B * T * NH * D; n++)
    {
        // inp[b][t][nh][d] -> out[b][nh][t][d]
        int b = n / (T * NH * D);
        int t = (n - b * (T * NH * D)) / (NH * D);
        int nh = ((n - b * (T * NH * D)) - t * (NH * D)) / D;
        int d = n % D;
        int out_offset = b * (T * NH * D) + t * (D) + nh * (T * D) + d;
        out[out_offset] = inp[n];
    }
}

__global__ void permute_kernel(float *out,
                               const float *inp,
                               int B, int T, int NH, int D)
{
    // inp : (B,T,NH,D), out: (B,NH,T,D)
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < B * T * NH * D)
    {
        int b = n / (T * NH * D);
        int t = (n - b * (T * NH * D)) / (NH * D);
        int nh = ((n - b * (T * NH * D)) - t * (NH * D)) / D;
        int d = n % D;
        int out_offset = b * (T * NH * D) + t * (D) + nh * (T * D) + d;
        out[out_offset] = inp[n];
    }
}

__global__ void permute_kernel(float *out,
                               const float *inp,
                               int B, int T, int NH, int D,
                               int pt)
{
    // inp : (B,T,NH,D), out: (oB,oNH,oT,oC)
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < B * T * NH * D)
    {
        int b = n / (T * NH * D);
        int t = (n - b * (T * NH * D)) / (NH * D);
        int nh = ((n - b * (T * NH * D)) - t * (NH * D)) / D;
        int d = n % D;
        // t0213 -> (B,NH,T,D), not t0213 -> t0231 -> (B,NH, D,T)
        int out_offset = pt == 1 ? b * (NH * T * D) + nh * (T * D) + t * (D) + d : b * (T * NH * D) + nh * (T * D) + d * T + t;
        out[out_offset] = inp[n];
    }
}

__global__ void permute_back_kernel(float *out,
                                    const float *inp,
                                    int B, int T, int NH, int D, int pt)

{
    // inp : pt == 1 ? (B,NH,T, D)  : (B,NH,D, T) (pt==0), out: (B,T, NH,D)
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // n = pt==1 ? b*(NH*T*D) + nh * T*D + t*D + d : b*(NH*T*D) + nh*T*D + d*T + t
    if (n < B * NH * T * D)
    {
        int b = n / (NH * T * D);
        int nh = (n / (T * D)) % NH;
        int t = (n / D) % T;
        int d = n % D;
        if (pt == 0)
        {
            t = n % T;
            d = (n / T) % D;
        }
        // int b = n / (T * NH * D);
        // int nh = (n - b * (T * NH * D)) / (T * D);
        // int t = ((n - b * (T * NH * D)) - nh * (T * D)) / D;
        // int d = n % D;
        int out_offset = b * (T * NH * D) + t * (NH * D) + nh * (D) + d;
        out[out_offset] = inp[n];
    }
}

at::Tensor permute(const at::Tensor &input, bool t0213)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    int B = input.size(0);
    int T = input.size(1);
    int NH = input.size(2);
    int D = input.size(3);

    // Create output tensor

    at::Tensor output = t0213 == true ? torch::zeros({B, NH, T, D}, torch::device(torch::kCUDA)) : torch::zeros({B, NH, D, T}, torch::device(torch::kCUDA));

    // Calculate grid and block dimensions
    int block_size = BLOCK_SIZE * BLOCK_SIZE;
    dim3 blockSize(block_size);

    dim3 gridSize((B * T * NH * D + block_size - 1) / block_size); // 1dgrid
    // Get tensor data pointers
    const float *x = input.data_ptr<float>();
    float *out = output.data_ptr<float>();

    // permute_kernel<<<gridSize, blockSize>>>(out, x, B, T, NH, D);
    permute_kernel<<<gridSize, blockSize>>>(out,
                                            x,
                                            B, T, NH, D,
                                            t0213);
    return output;
}

at::Tensor permute_back(const at::Tensor &input, bool t0213)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    int B = input.size(0);
    int NH = input.size(1);
    // if t0213 is true inp : B,NH,T,D else inp : B,NH,D,T
    int T = t0213 == true ? input.size(2) : input.size(3);
    int D = t0213 == true ? input.size(3) : input.size(2);

    // Create output tensor
    at::Tensor output = torch::zeros({B, T, NH, D}, torch::device(torch::kCUDA));

    // Calculate grid and block dimensions
    int block_size = BLOCK_SIZE * BLOCK_SIZE;
    dim3 blockSize(block_size);

    dim3 gridSize((B * T * NH * D + block_size - 1) / block_size); // 1dgrid
    // Get tensor data pointers
    const float *x = input.data_ptr<float>();
    float *out = output.data_ptr<float>();

    permute_back_kernel<<<gridSize, blockSize>>>(out, x, B, T, NH, D, t0213);
    return output;
}

#define BLOCK_SIZE 32
at::Tensor matmul(const at::Tensor &input, const at::Tensor &weight)
{
    /*
    input : B*T, C
    weight : C, OC

    out = input @weight (BT, OC)
    But in cublas the arrays are readin col maj format. So what is actually seen by the library is
    input.T , weight.T which cannot be multiplied.
    Instead we pass weight, input. w
    output = weight.T @ input.T (which is actually out.T). output is laid out in memory with stride (leading dim of output = cols of weight (OC) ).
    So when read in row major format we get back the original out (input @ weight)

    ROW MAJOR
    x = [[1,2,3]],  mem(x) = [1,2,3,4,5,6]      w = [[1,1]], mem(w) = [1,1,1,1,1,1]
        [[4,5,6]]                                   [[1,1]]
                                                    [[1,1]]

    out(x@w) = [[6,6]]        mem(out) = [6,6,15,15] (ROW MAJOR)
               [[15,15]]

    For CUBLAS(Col major)
    xc = [[1,4]] -> x.T              wc = [[1,1,1]] -> w.T
         [[2,5]]                          [[1,1,1]]
         [[3,6]]

    output = wc @ xc = [[6,15]] mem(output) = [6,6,15,15] (remember output is in COL MAJOR format)
                       [[6,15]]

    We can see mem(output) is the same as mem(out). So when we read back mem(output) from the application side (in a ROW major)
    we get back the answer to the original input @ weight
    */

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(weight.is_contiguous(), "Input tensor must be contiguous");
    int BT = input.size(0);
    int C = input.size(1);
    int OC = weight.size(1);

    at::Tensor output = torch::zeros({BT, OC}, torch::device(torch::kCUDA));

    const float *x = input.data_ptr<float>();
    const float *w = weight.data_ptr<float>();
    float *out = output.data_ptr<float>();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(
        handle,      // cuBLAS handle
        CUBLAS_OP_N, // w is not transposed
        CUBLAS_OP_N, // x is not transposed
        OC,          // m,number of rows in output
        BT,          // n,number of columns in output
        C,           // k, number of columns in x / rows in w
        &alpha,      // scaling factor for multiplication
        w,           // input matrix x
        OC,          // leading dimension of x
        x,           // weight matrix
        C,           // leading dimension of w
        &beta,       // scaling factor for output
        out,         // output matrix
        OC           // leading dimension of output
    );
    cublasDestroy(handle);

    return output;
}

// __global__ void attention_kernel(float *out,
//                                  const float *qkT,
//                                  const float *v,
//                                  int B, int T, int NH, int D)
// {
//     int bh = blockIdx.x;
//     int m = blockDim.y;
// }

at::Tensor batch_mm_xy(const at::Tensor &x, const at::Tensor &y)
{
    /*
    x : B,NH,T1,C
    y : B, NH,C, T2
    out : B,NH, T1, T2
    out = x@y (B,T1,T2)
    */
    int B = x.size(0);
    int NH = x.size(1);
    int T1 = x.size(2);
    int C = x.size(3);

    int T2 = y.size(3);

    at::Tensor output = torch::zeros({B, NH, T1, T2}, torch::device(torch::kCUDA));

    const float *xx = x.data_ptr<float>();
    const float *yy = y.data_ptr<float>();
    float *out = output.data_ptr<float>();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        T2, // rows of A and C
        T1, // columns of B and C
        C,  // columns of A and rows of B
        &alpha,
        yy,
        T2,
        C * T2,
        xx,
        C,
        C * T1,
        &beta,
        out,
        T2,
        T1 * T2,
        B * NH);

    cublasDestroy(handle);

    return output;
}

__device__ float warpReduceMax(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void max_kernel(const float *x, float *m, int N, int block_size)
{
    /*
        N does not have to equal block_size.In fact in most cases it will not .eg in the attention matrix,
        N is typicaly in the thousands, while block_size is a few hundreds
        */
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp
    // m += idx * N;
    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float *maxvals = shared;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x)
    {
        maxval = fmaxf(maxval, x[idx * N + i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0)
        maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0)
    {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        m[idx] = val;
    }
    // __syncthreads();
}

__global__ void softmax_forward_kernel(float *out, float *inp, int B, int NH, int T, int block_size)
// softmax_forward_kernel(out, xx, B, NH, T, block_size);
{
    extern __shared__ float shared[];
    int idx = blockIdx.x; // (b,nh, t) = b * NH*T + nh*T + t
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    int b = idx / (NH * T);
    int nh = idx % (NH * T);
    int t = idx % T;
    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float *maxvals = shared;
    float *sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    float *x = inp + idx * T;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < T; i += blockDim.x)
    {
        // first apply the causal mask
        if (t < i)
            x[i] = -INFINITY;
        // int temp = t < i ? -INFINITY : x[i];
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0)
        maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0)
    {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < T; i += blockDim.x)
    {
        // subtract max for numerical stability
        out[idx * T + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * T;
    float sumval = 0.0f;
    for (int i = tid; i < T; i += blockDim.x)
    {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0)
        sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0)
    {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i)
        {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < T; i += blockDim.x)
    {
        out[idx * T + i] = x[i] / sum;
    }
}

at::Tensor max(at::Tensor &x)
{

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on GPU");
    int H = x.size(0);
    int W = x.size(1);
    std::cout << "H: " << H << ", W: " << W << std::endl;
    at::Tensor output = torch::zeros({H, 1},
                                     torch::device(torch::kCUDA));

    float *xx = x.data_ptr<float>();
    float *out = output.data_ptr<float>();
    max_kernel<<<H, 256, 8 * sizeof(float)>>>(xx, out, W, 256);
    // softmax_forward_kernel4<<<1, 256, 2 * 8 * sizeof(float)>>>(out, xx, 1, N);
    cudaDeviceSynchronize(); // Add error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    return output;
}

at::Tensor softmax(at::Tensor &x)
{
    /*
    x : (B,NH,T,T)
    out: (B,NH,T,T)
    */

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on GPU");
    int B = x.size(0);
    int NH = x.size(1);
    int T = x.size(2);
    at::Tensor output = torch::zeros({B, NH, T, T},
                                     torch::device(torch::kCUDA));
    float *xx = x.data_ptr<float>();
    float *out = output.data_ptr<float>();
    int block_size = 256;
    int num_blocks = (B * NH * T);
    softmax_forward_kernel<<<num_blocks, block_size, 2 * block_size * sizeof(float)>>>(out, xx, B, NH, T, block_size);
    return output;
}

at::Tensor attention_forward(at::Tensor &q, at::Tensor &k, at::Tensor &v)
{
    /*
    qkv : B,T,NH,C
    */
    int B = q.size(0);
    int T = q.size(1);
    int NH = q.size(2);
    int C = q.size(3);

    // permute q,k,v
    at::Tensor qp = torch::zeros({B, NH, T, C},
                                 torch::device(torch::kCUDA));
    at::Tensor vp = torch::zeros({B, NH, T, C},
                                 torch::device(torch::kCUDA));
    at::Tensor kp = torch::zeros({B, NH, C, T},
                                 torch::device(torch::kCUDA));
    int block_size = BLOCK_SIZE * BLOCK_SIZE;
    dim3 blockSize(block_size);

    dim3 gridSize((B * T * NH * C + block_size - 1) / block_size); // 1dgrid
    // Get tensor data pointers
    const float *qq = q.data_ptr<float>();
    const float *kk = k.data_ptr<float>();
    const float *vv = v.data_ptr<float>();

    float *qpp = qp.data_ptr<float>();
    float *kpp = kp.data_ptr<float>();
    float *vpp = vp.data_ptr<float>();

    // permute_kernel<<<gridSize, blockSize>>>(out, x, B, T, NH, D);
    permute_kernel<<<gridSize, blockSize>>>(qpp,
                                            qq,
                                            B, T, NH, C,
                                            1);
    permute_kernel<<<gridSize, blockSize>>>(kpp,
                                            kk,
                                            B, T, NH, C,
                                            0);
    permute_kernel<<<gridSize, blockSize>>>(vpp,
                                            vv,
                                            B, T, NH, C,
                                            1);

    // qp @kpT
    at::Tensor scores = torch::zeros({B, NH, T, T}, torch::device(torch::kCUDA));
    float *score = scores.data_ptr<float>();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        T, // rows of A and C
        T, // columns of B and C
        C, // columns of A and rows of B
        &alpha,
        kpp,
        T,
        C * T,
        qpp,
        C,
        C * T,
        &beta,
        score,
        T,
        T * T,
        B * NH);

    // softmax
    at::Tensor output_sm = torch::zeros({B, NH, T, T},
                                        torch::device(torch::kCUDA));
    float *out_sm = output_sm.data_ptr<float>();
    block_size = 256;
    int num_blocks = (B * NH * T);
    softmax_forward_kernel<<<num_blocks, block_size, 2 * block_size * sizeof(float)>>>(out_sm, score, B, NH, T, block_size);
    // return output_sm;

    // output_sm [B,NH,T,T](B) @ v [B, NH, T,C] (A)
    at::Tensor smv = torch::zeros({B, NH, T, C}, torch::device(torch::kCUDA));
    float *smv_ = smv.data_ptr<float>();

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        C, // rows of A and C
        T, // columns of B and C
        T, // columns of A and rows of B
        &alpha,
        vpp,
        C,
        C * T,
        out_sm,
        T,
        T * T,
        &beta,
        smv_,
        C,
        C * T,
        B * NH);

    // return smv;
    // permute back from B,NH,T,C to B, T, NH, C
    at::Tensor final_out = torch::zeros({B, T, NH, C}, torch::device(torch::kCUDA));

    // Calculate grid and block dimensions
    block_size = BLOCK_SIZE * BLOCK_SIZE;
    blockSize = block_size;

    gridSize = (B * T * NH * C + block_size - 1) / block_size; // 1dgrid
    // Get tensor data pointers
    float *out = final_out.data_ptr<float>();

    permute_back_kernel<<<gridSize, blockSize>>>(out, smv_, B, T, NH, C, 1);
    return final_out;
}

__global__ void _attention_forward_kernel1(float *q, float *k, float *v, float *out,
                                           int NH, int T, int C, int causal, int scale)
{
    /*
     qkv : B,NH,T, C
     allocate a block of work for each T,T attn. Total number of blocks B*NH
     A block will contain Br(32) threads (not T*T). So each thread will have to work on multiple elements
      q(T,C)
      ________________
      |                |
  i   |________________|
      |----------------|          Each row tile will have 32 rows (handled by the threads in the block)
      |________________|
      |                |
      |________________|

      k(T,C)
      ________________
     |________________|
  j  |----------------|
     |________________|
     |                |
     |                |
     |________________|

      v(T,C)
      ________________
     |________________|
   j |----------------|
     |________________|
     |                |
     |                |
     |________________|

       attn(T,T)
      ________________
     |         j      |
     |        __      |
     |      i|__|     |
     |                |
     |                |
     |________________|
     */
    // blockIdx.x = (B * NH),  qkv, out : B, NH, T,C
    // each row of the tile will be handled by a single tid

    // return;
    int b = blockIdx.x / NH;
    int nh = blockIdx.x % NH;
    int tid = threadIdx.x;

    int num_threads = blockDim.x;
    int Br = num_threads;
    int Bc = num_threads;
    int num_cols = (T + num_threads - 1) / num_threads;
    int num_rows = (T + num_threads - 1) / num_threads;
    extern __shared__ float smem[]; // int smem = (3 * Br * C + 2 * Br + Br * Br) * sizeof(float); q,k,v, attn, running_max, running_sum

    float *qs = smem;
    float *ks = qs + Br * C;
    float *vs = ks + Br * C;
    float *attn = vs + Br * C;
    float *running_max = attn + Br * Br;
    float *running_sum = running_max + Br;
    float scale_factor = rsqrtf(C);

    q += b * NH * T * C + nh * T * C;
    k += b * NH * T * C + nh * T * C;
    v += b * NH * T * C + nh * T * C;
    out += b * NH * T * C + nh * T * C;

    // each thread will work on a single row of every row tile (enumerated by i)
    // q[i,tid] -> out[i, tid]
    for (int i = 0; i < num_rows; i++)
    {
        running_max[tid] = -INFINITY;
        running_sum[tid] = 0.f;
        if (i * Br + tid >= T)
            break;
        for (int c = 0; c < C; c++)
            qs[tid * C + c] = q[(i * Br + tid) * C + c]; // qs[tid,x] = k[i+tid,x]
        for (int j = 0; j < num_cols; j++)
        {
            if (j * Br + tid < T)
                // break;

                for (int c = 0; c < C; c++)
                {
                    // {
                    ks[tid * C + c] = k[(j * Br + tid) * C + c]; // ks[tid,x] = k[j+tid,x]
                    vs[tid * C + c] = v[(j * Br + tid) * C + c]; // vs[tid,x] = s[j+tid,x]
                    // }
                }

            __syncthreads(); // qs, ks and vs are completely filled (by the threads in this block)
            float val;
            for (int y = 0; y < Bc; y++)
            {

                val = 0.f;
                for (int k = 0; k < C; k++)
                    val += qs[tid * C + k] * ks[y * C + k]; // qs[tid,k] * ksT[k, y] -> qs[tid,k] * ks[y,k]
                // attn[tid * Bc + y] = ((i * Br + tid < T) && (j * Br + y < T)) ? val : -INFINITY;
                attn[tid * Bc + y] = val * (scale * scale_factor + 1 - scale);
                if ((i * Br + tid >= T) || (j * Br + y >= T) || (causal && ((i * Br + tid < j * Br + y))))
                    attn[tid * Bc + y] = -INFINITY;
            }

            // online softmax
            float row_max = running_max[tid]; // max in i block along tid row
            float row_sum = running_sum[tid]; // sum in i block along tid row
            for (int y = 0; y < Bc; y++)
                if (row_max < attn[tid * Bc + y])
                    row_max = attn[tid * Bc + y]; // will be max value seen upto this point

            // scale the prev den
            row_sum *= expf(running_max[tid] - row_max);
            // __syncthreads(); // dont need this. threads are run row wise
            for (int y = 0; y < Bc; y++)
                attn[tid * Bc + y] = expf(attn[tid * Bc + y] - row_max);
            // __syncthreads();

            for (int y = 0; y < Bc; y++)
                row_sum += (attn[tid * Bc + y]);
            for (int y = 0; y < num_threads; y++)
                attn[tid * Bc + y] /= row_sum + 1e-10;

            // __syncthreads(); // softmax of block is done
            float old_val = 0.f;
            for (int k = 0; k < C; k++)
            {
                val = 0.f;
                old_val = out[(i * Br + tid) * C + k];

                for (int y = 0; y < Bc; y++)
                    val += attn[tid * Bc + y] * vs[y * C + k]; // attn[tid, y] * vs[y,k]

                val += (old_val * running_sum[tid] * expf(running_max[tid] - row_max)) / row_sum;

                // update the value
                out[(i * Br + tid) * C + k] = val;
            }
            running_sum[tid] = row_sum;
            running_max[tid] = row_max;
        }
        // __syncthreads();
    }
}

__global__ void attention_forward_kernel1(float *q, float *k, float *v, float *out,
                                          int NH, int T, int C, int causal, int scale)
{
    /*
     qkv : B,NH,T, C
     allocate a block of work for each T,T attn. Total number of blocks B*NH
     A block will contain Br(32) threads (not T*T). So each thread will have to work on multiple elements
      q(T,C)
      ________________
      |                |
  i   |________________|
      |----------------|          Each row tile will have 32 rows (handled by the threads in the block)
      |________________|
      |                |
      |________________|

      k(T,C)
      ________________
     |________________|
  j  |----------------|
     |________________|
     |                |
     |                |
     |________________|

      v(T,C)
      ________________
     |________________|
   j |----------------|
     |________________|
     |                |
     |                |
     |________________|

       attn(T,T)
      ________________
     |         j      |
     |        __      |
     |      i|__|     |
     |                |
     |                |
     |________________|
     */

    // blockIdx.x = (B * NH),  qkv, out : B, NH, T,C
    // each row of the tile will be handled by a single tid
    int b = blockIdx.x / NH;
    int nh = blockIdx.x % NH;
    int tid = threadIdx.x;

    int num_threads = blockDim.x;
    int Br = num_threads;
    int Bc = num_threads;
    int num_cols = (T + num_threads - 1) / num_threads;
    int num_rows = (T + num_threads - 1) / num_threads;
    extern __shared__ float smem[]; // int smem = (3 * Br * C + 2 * Br + Br * Br) * sizeof(float); q,k,v, attn, running_max, running_sum

    float *qs = smem;
    float *ks = qs + Br * C;
    float *vs = ks + Br * C;
    float *attn = vs + Br * C;
    float *running_max = attn + Br * Br;
    float *running_sum = running_max + Br;
    float scale_factor = rsqrtf(C);

    q += b * NH * T * C + nh * T * C;
    k += b * NH * T * C + nh * T * C;
    v += b * NH * T * C + nh * T * C;
    out += b * NH * T * C + nh * T * C;

    // each thread will work on a single row of every row tile (enumerated by i)
    // q[i,tid] -> out[i, tid]
    for (int i = 0; i < num_rows; i++)
    {
        running_max[tid] = -INFINITY;
        running_sum[tid] = 0.f;
        if (i * Br + tid >= T)
            break;
        for (int c = 0; c < C; c++)
            qs[tid * C + c] = q[(i * Br + tid) * C + c]; // qs[tid,x] = k[i+tid,x]
        for (int j = 0; j < num_cols; j++)
        {
            if (j * Br + tid < T)
                for (int c = 0; c < C; c++)
                {
                    ks[tid * C + c] = k[(j * Br + tid) * C + c]; // ks[tid,x] = k[j+tid,x]
                    vs[tid * C + c] = v[(j * Br + tid) * C + c]; // vs[tid,x] = s[j+tid,x]
                }
            __syncthreads(); // qs, ks and vs are completely filled (by the threads in this block)
            float val;
            for (int y = 0; y < Bc; y++)
            {

                val = 0.f;
                for (int k = 0; k < C; k++)
                    val += qs[tid * C + k] * ks[y * C + k]; // qs[tid,k] * ksT[k, y] -> qs[tid,k] * ks[y,k]
                attn[tid * Bc + y] = val * (scale * scale_factor + 1 - scale);
                if ((i * Br + tid >= T) || (j * Br + y >= T) || (causal && ((i * Br + tid < j * Br + y))))
                    attn[tid * Bc + y] = -INFINITY;
            }

            // online softmax
            float row_max = running_max[tid]; // max in i block along tid row
            float row_sum = running_sum[tid]; // sum in i block along tid row
            for (int y = 0; y < Bc; y++)
                if (row_max < attn[tid * Bc + y])
                    row_max = attn[tid * Bc + y]; // will be max value seen upto this point

            // scale the prev den
            row_sum *= expf(running_max[tid] - row_max);
            // __syncthreads(); // dont need this. threads are run row wise
            float _attn;
            for (int y = 0; y < Bc; y++)
            {
                _attn = attn[tid * Bc + y];
                _attn = expf(_attn - row_max);
                row_sum += _attn;
                attn[tid * Bc + y] = _attn;
            }
            for (int y = 0; y < Bc; y++)
                attn[tid * Bc + y] /= (row_sum + 1e-10);

            // __syncthreads(); // softmax of block is done
            float old_val = 0.f;
            for (int k = 0; k < C; k++)
            {
                val = 0.f;
                old_val = out[(i * Br + tid) * C + k];

                for (int y = 0; y < Bc; y++)
                    val += attn[tid * Bc + y] * vs[y * C + k]; // attn[tid, y] * vs[y,k]

                val += (old_val * running_sum[tid] * expf(running_max[tid] - row_max)) / row_sum;

                // update the value
                out[(i * Br + tid) * C + k] = val;
            }
            running_sum[tid] = row_sum;
            running_max[tid] = row_max;
        }
        // __syncthreads();
    }
    return;
}

at::Tensor attention_forward1(at::Tensor &q, at::Tensor &k, at::Tensor &v, bool causal, bool scale)
{
    assert(q.is_cuda());
    assert(k.is_cuda());
    assert(v.is_cuda());

    int B = q.size(0);
    int NH = q.size(1);
    int T = q.size(2);
    int C = q.size(3); // head dim
    int num_threads_per_block = 8;

    int row_tiles = (T + num_threads_per_block - 1) / num_threads_per_block;
    int col_tiles = row_tiles;

    dim3 blocks(B * NH);
    int Br = num_threads_per_block;
    dim3 threads(Br);

    // q_row + k_col + v_col + rows_max + rows_den + attn block (num_threads * num_threads)
    int smem = (3 * Br * C + 2 * Br + Br * Br) * sizeof(float);

    at::Tensor out = torch::zeros({B, NH, T, C}, torch::device(torch::kCUDA));
    
    float *_out = out.data_ptr<float>();
    float *_q = q.data_ptr<float>();
    float *_k = k.data_ptr<float>();
    float *_v = v.data_ptr<float>();
    attention_forward_kernel1<<<blocks, threads, smem>>>(_q, _k, _v, _out, NH, T, C, causal, scale);

    // After kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error in kernel: %s\n", cudaGetErrorString(err));
        assert(false);
    }
    else {
        return out;
    }

    // cudaDeviceSynchronize();
}