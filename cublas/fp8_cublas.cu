#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

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
            fprintf(stderr, "cuBLASLt error %s:%d: status=%d\n", __FILE__,       \
                    __LINE__, (int)status__);                                    \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

static void init_f32(std::vector<float>& x) {
    for (size_t i = 0; i < x.size(); ++i) {
        int t = static_cast<int>(i % 17) - 8;   // [-8, 8]
        x[i] = static_cast<float>(t) / 8.0f;    // [-1.0, 1.0]
    }
}

static void f32_to_fp8_e4m3(const std::vector<float>& in, std::vector<__nv_fp8_e4m3>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = __nv_fp8_e4m3(in[i]);
    }
}

static float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

static void print_sample_f32(const char* name, const std::vector<float>& x, int n = 8) {
    std::cout << name << ": ";
    int limit = std::min<int>(n, static_cast<int>(x.size()));
    for (int i = 0; i < limit; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";
}

static void print_sample_fp8(const char* name, const std::vector<__nv_fp8_e4m3>& x, int n = 8) {
    std::cout << name << ": ";
    int limit = std::min<int>(n, static_cast<int>(x.size()));
    for (int i = 0; i < limit; ++i) {
        std::cout << static_cast<float>(x[i]) << " ";
    }
    std::cout << "\n";
}

static void print_sample_bf16(const char* name, const std::vector<__nv_bfloat16>& x, int n = 8) {
    std::cout << name << ": ";
    int limit = std::min<int>(n, static_cast<int>(x.size()));
    for (int i = 0; i < limit; ++i) {
        std::cout << bf16_to_f32(x[i]) << " ";
    }
    std::cout << "\n";
}

static float cpu_dot_rowmajor_A_mul_BT(
    const std::vector<float>& A,  // [M, K]
    const std::vector<float>& B,  // [N, K]
    int K,
    int row_m, int col_n)
{
    float acc = 0.0f;
    const float* a = &A[static_cast<size_t>(row_m) * K];
    const float* b = &B[static_cast<size_t>(col_n) * K];
    for (int k = 0; k < K; ++k) {
        acc += a[k] * b[k];
    }
    return acc;
}

int main(int argc, char** argv) {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    int warmup = 20;
    int iters  = 200;
    if (argc >= 2) warmup = std::atoi(argv[1]);
    if (argc >= 3) iters  = std::atoi(argv[2]);

    std::cout << "Benchmark cuBLASLt FP8(E4M3) GEMM with BF16 output\n";
    std::cout << "Compute: D = A * B^T\n";
    std::cout << "A: row-major [" << M << ", " << K << "] FP8 E4M3\n";
    std::cout << "B: row-major [" << N << ", " << K << "] FP8 E4M3\n";
    std::cout << "D: row-major [" << M << ", " << N << "] BF16\n";
    std::cout << "warmup=" << warmup << ", iters=" << iters << "\n";

    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";

    std::vector<float> hA_f32(static_cast<size_t>(M) * K);
    std::vector<float> hB_f32(static_cast<size_t>(N) * K);
    init_f32(hA_f32);
    init_f32(hB_f32);

    std::vector<__nv_fp8_e4m3> hA_fp8;
    std::vector<__nv_fp8_e4m3> hB_fp8;
    f32_to_fp8_e4m3(hA_f32, hA_fp8);
    f32_to_fp8_e4m3(hB_f32, hB_fp8);

    std::vector<__nv_bfloat16> hD_bf16(static_cast<size_t>(M) * N);

    print_sample_f32("Sample A(f32,row-major MxK)", hA_f32, 8);
    print_sample_f32("Sample B(f32,row-major NxK)", hB_f32, 8);
    print_sample_fp8("Sample A(fp8,row-major MxK)", hA_fp8, 8);
    print_sample_fp8("Sample B(fp8,row-major NxK)", hB_fp8, 8);

    __nv_fp8_e4m3* dA = nullptr;   // row-major [M,K]
    __nv_fp8_e4m3* dB = nullptr;   // row-major [N,K]
    __nv_bfloat16* dC = nullptr;   // col-major [N,M] backing row-major [M,N]
    __nv_bfloat16* dD = nullptr;   // col-major [N,M] backing row-major [M,N]

    float* dScaleA = nullptr;
    float* dScaleB = nullptr;

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(__nv_fp8_e4m3);
    const size_t bytesB = static_cast<size_t>(N) * K * sizeof(__nv_fp8_e4m3);
    const size_t bytesD = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesD));
    CHECK_CUDA(cudaMalloc(&dD, bytesD));
    CHECK_CUDA(cudaMalloc(&dScaleA, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dScaleB, sizeof(float)));

    const float one = 1.0f;
    CHECK_CUDA(cudaMemcpy(dA, hA_fp8.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB_fp8.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesD));
    CHECK_CUDA(cudaMemset(dD, 0, bytesD));
    CHECK_CUDA(cudaMemcpy(dScaleA, &one, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dScaleB, &one, sizeof(float), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt;
    CHECK_CUBLAS(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // 目标：
    //   D_row(M,N) = A_row(M,K) * B_row(N,K)^T
    //
    // 用列主序重解释：
    //   dB 的 row-major [N,K] 内存 -> 视为 col-major [K,N]
    //   dA 的 row-major [M,K] 内存 -> 视为 col-major [K,M]
    //
    // 调 cublasLt 时做：
    //   D_col(N,M) = transpose(dB_as_col[K,N]) * nontranspose(dA_as_col[K,M])
    //
    // 这样 dD 的底层内存正好等价于 row-major [M,N]
    const cublasOperation_t transA = CUBLAS_OP_T;
    const cublasOperation_t transB = CUBLAS_OP_N;

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB)));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;

    // A operand = dB buffer, viewed as col-major [K, N], lda = K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, N, K));

    // B operand = dA buffer, viewed as col-major [K, M], ldb = K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K));

    // C/D = BF16, viewed as col-major [N, M], ldc/ldd = N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, N, M, N));

    const size_t workspaceSize = 32ull * 1024 * 1024;
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof(workspaceSize)));

    constexpr int kRequestedAlgoCount = 16;
    cublasLtMatmulHeuristicResult_t heuristicResults[kRequestedAlgoCount];
    int returnedResults = 0;

    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        lt,
        opDesc,
        Adesc,
        Bdesc,
        Cdesc,
        Ddesc,
        preference,
        kRequestedAlgoCount,
        heuristicResults,
        &returnedResults));

    if (returnedResults == 0) {
        std::fprintf(stderr, "No FP8 heuristic found.\n");
        std::exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    auto run_once = [&](const cublasLtMatmulHeuristicResult_t& h) -> cublasStatus_t {
        return cublasLtMatmul(
            lt,
            opDesc,
            &alpha,
            dB, Adesc,
            dA, Bdesc,
            &beta,
            dC, Cdesc,
            dD, Ddesc,
            &h.algo,
            workspace,
            workspaceSize,
            0);
    };

    int algoIndex = -1;
    for (int i = 0; i < returnedResults; ++i) {
        cublasStatus_t st = run_once(heuristicResults[i]);
        if (st == CUBLAS_STATUS_SUCCESS) {
            algoIndex = i;
            break;
        }
    }
    if (algoIndex < 0) {
        std::fprintf(stderr, "No runnable FP8 heuristic found.\n");
        std::exit(EXIT_FAILURE);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(run_once(heuristicResults[algoIndex]));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(run_once(heuristicResults[algoIndex]));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(iters);

    const double flops  = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = flops / (avg_ms * 1e-3) / 1e12;

    std::cout << "Selected heuristic index: " << algoIndex << "\n";
    std::cout << "Total time: " << total_ms << " ms\n";
    std::cout << "Avg time  : " << avg_ms << " ms\n";
    std::cout << "TFLOPS    : " << tflops << "\n";

    CHECK_CUDA(cudaMemcpy(hD_bf16.data(), dD, bytesD, cudaMemcpyDeviceToHost));
    print_sample_bf16("Sample D(bf16,row-major)", hD_bf16, 8);

    std::cout << "CPU ref first row (float): ";
    for (int j = 0; j < 8; ++j) {
        float ref = cpu_dot_rowmajor_A_mul_BT(hA_f32, hB_f32, K, 0, j);
        std::cout << ref << " ";
    }
    std::cout << "\n";

    std::cout << "GPU first row (bf16->f32): ";
    for (int j = 0; j < 8; ++j) {
        std::cout << bf16_to_f32(hD_bf16[j]) << " ";
    }
    std::cout << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc));
    CHECK_CUBLAS(cublasLtDestroy(lt));

    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dD));
    CHECK_CUDA(cudaFree(dScaleA));
    CHECK_CUDA(cudaFree(dScaleB));

    return 0;
}