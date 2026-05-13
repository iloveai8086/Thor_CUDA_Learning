#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err__));                                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t status__ = (call);                                        \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__,         \
                    __LINE__, (int)status__);                                    \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

static void init_int8(std::vector<int8_t>& x, int scale = 1) {
    for (size_t i = 0; i < x.size(); ++i) {
        int t = static_cast<int>(i % 17) - 8;  // [-8, 8]
        int v = t * scale;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        x[i] = static_cast<int8_t>(v);
    }
}

static void print_sample_i8(const char* name, const std::vector<int8_t>& x, int n = 8) {
    std::cout << name << ": ";
    int limit = std::min<int>(n, static_cast<int>(x.size()));
    for (int i = 0; i < limit; ++i) {
        std::cout << static_cast<int>(x[i]) << " ";
    }
    std::cout << "\n";
}

static void print_sample_i32(const char* name, const std::vector<int32_t>& x, int n = 8) {
    std::cout << name << ": ";
    int limit = std::min<int>(n, static_cast<int>(x.size()));
    for (int i = 0; i < limit; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    int warmup = 20;
    int iters  = 100;
    if (argc >= 2) warmup = std::atoi(argv[1]);
    if (argc >= 3) iters  = std::atoi(argv[2]);

    std::cout << "Benchmark cuBLAS INT8 GEMM on Thor/Blackwell-style path\n";
    std::cout << "Compute: C = A * B^T\n";
    std::cout << "A: row-major [" << M << ", " << K << "]\n";
    std::cout << "B: row-major [" << N << ", " << K << "]\n";
    std::cout << "C: row-major [" << M << ", " << N << "]\n";
    std::cout << "warmup=" << warmup << ", iters=" << iters << "\n";

    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";

    // host row-major buffers
    std::vector<int8_t>  hA(static_cast<size_t>(M) * K);
    std::vector<int8_t>  hB(static_cast<size_t>(N) * K);   // B stored as [N, K]
    std::vector<int32_t> hC(static_cast<size_t>(M) * N);

    init_int8(hA, 1);
    init_int8(hB, 1);

    print_sample_i8("Sample A(row-major MxK)", hA, 8);
    print_sample_i8("Sample B(row-major NxK)", hB, 8);

    int8_t*  dA = nullptr;
    int8_t*  dB = nullptr;
    int32_t* dC = nullptr;

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(int8_t);
    const size_t bytesB = static_cast<size_t>(N) * K * sizeof(int8_t);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(int32_t);

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 对 int8 Tensor Core 路径通常设置这个更稳妥
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const int32_t alpha = 1;
    const int32_t beta  = 0;

    // 关键说明：
    // 我们的输入/输出都是 row-major。
    // cuBLAS 是 column-major 接口，因此这里使用“交换矩阵 + 转置解释”的经典技巧：
    //
    //   C_row(M,N) = A_row(M,K) * B_row(N,K)^T
    //
    // 对应到 column-major 可写成：
    //
    //   C_col(N,M) = op(dB) * op(dA)
    //
    // 其中：
    //   dB 这块 row-major [N,K] 内存，按 column-major 看，相当于 [K,N]，所以需要 TRANSPOSE 得到 [N,K]
    //   dA 这块 row-major [M,K] 内存，按 column-major 看，相当于 [K,M]，直接用 NON-TRANSPOSE 即可
    //
    // 因此调用：
    //   C_col(N,M) = transpose(B_buf_as_col[K,N]) * nontranspose(A_buf_as_col[K,M])
    //
    // 最终 dC 的内存布局正好就是我们想要的 row-major [M,N]
    //
    // gemm 参数：
    //   m = N
    //   n = M
    //   k = K
    //   A operand = dB, transA = T, lda = K
    //   B operand = dA, transB = N, ldb = K
    //   C operand = dC, ldc = N
    //
    // 这里要求 K/N/M 都最好满足 int8 路径的对齐要求。你当前 4096 全都满足。
    const cublasOperation_t transA = CUBLAS_OP_T;
    const cublasOperation_t transB = CUBLAS_OP_N;

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            transA, transB,
            N, M, K,
            &alpha,
            dB, CUDA_R_8I, K,
            dA, CUDA_R_8I, K,
            &beta,
            dC, CUDA_R_32I, N,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            transA, transB,
            N, M, K,
            &alpha,
            dB, CUDA_R_8I, K,
            dA, CUDA_R_8I, K,
            &beta,
            dC, CUDA_R_32I, N,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(iters);

    const double ops  = 2.0 * static_cast<double>(M) * N * K;
    const double tops = ops / (avg_ms * 1e-3) / 1e12;

    std::cout << "Total time: " << total_ms << " ms\n";
    std::cout << "Avg time  : " << avg_ms << " ms\n";
    std::cout << "TOPS      : " << tops << "\n";

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    print_sample_i32("Sample C(row-major)", hC, 8);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}
// nvcc i8_cublas.cu -O3 -arch=sm_110a -lcublasLt -lcublas -o i8_cublas