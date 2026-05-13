#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
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

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static void init_f32(std::vector<float>& x) {
    for (size_t i = 0; i < x.size(); ++i) {
        int t = static_cast<int>(i % 17) - 8;
        x[i] = static_cast<float>(t) / 8.0f;   // [-1, 1]
    }
}

static float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// 把 row-major [rows, cols] 转成“底层按 col-major [cols, rows] 存”的线性数组。
// 用于之后做 TN:
//   A operand: underlying col-major [K, M], op(A)=T -> [M, K]
//   B operand: underlying col-major [K, N], op(B)=N -> [K, N]
static void row_major_to_col_major_underlying(
    const std::vector<float>& in_row,
    int rows, int cols,
    std::vector<float>& out_col_underlying)
{
    out_col_underlying.resize(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // row-major [rows, cols]
            float v = in_row[static_cast<size_t>(r) * cols + c];
            // write as col-major [cols, rows] => element(c, r)
            out_col_underlying[static_cast<size_t>(c) + static_cast<size_t>(r) * cols] = v;
        }
    }
}

// 将 float 序列按相邻两个值打包成 fp4x2
static void pack_f32_to_fp4x2(
    const std::vector<float>& in,
    std::vector<__nv_fp4x2_e2m1>& out)
{
    if (in.size() % 2 != 0) {
        std::fprintf(stderr, "Input size for fp4x2 packing must be even.\n");
        std::exit(EXIT_FAILURE);
    }
    out.resize(in.size() / 2);
    for (size_t i = 0; i < in.size(); i += 2) {
        float2 v;
        v.x = in[i + 0];
        v.y = in[i + 1];
        out[i / 2] = __nv_fp4x2_e2m1(v);
    }
}

static void print_sample_f32(const char* name, const std::vector<float>& x, int n = 8) {
    std::cout << name << ": ";
    int limit = std::min<int>(n, static_cast<int>(x.size()));
    for (int i = 0; i < limit; ++i) std::cout << x[i] << " ";
    std::cout << "\n";
}

static void print_sample_bf16(const char* name, const std::vector<__nv_bfloat16>& x, int n = 8) {
    std::cout << name << ": ";
    int limit = std::min<int>(n, static_cast<int>(x.size()));
    for (int i = 0; i < limit; ++i) std::cout << bf16_to_f32(x[i]) << " ";
    std::cout << "\n";
}

static float cpu_ref_rowmajor_A_mul_B(
    const std::vector<float>& A_row,  // [M,K]
    const std::vector<float>& B_col,  // underlying col-major [K,N] stored linearly
    int M, int N, int K,
    int i, int j)
{
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A_row[static_cast<size_t>(i) * K + k];
        // B_col is col-major [K,N]
        float b = B_col[static_cast<size_t>(k) + static_cast<size_t>(j) * K];
        acc += a * b;
    }
    return acc;
}

// 文档要求 FP4 scale 使用 tiled tensor 布局，起始地址 16B 对齐。
// 对于“全 1 scale”的最小样例，只要整个分配区都填成 1，布局内容就不会影响数值，
// 关键是分配足够大的 tiled buffer。
// 文档给出：一个 scale tile 对应一个 128x64 的 FP4 数据块。:contentReference[oaicite:7]{index=7}
static size_t fp4_scale_bytes_for_matrix(int inner_dim, int outer_dim) {
    const int tiles_inner = ceil_div(inner_dim, 128);
    const int tiles_outer = ceil_div(outer_dim, 64);
    const size_t bytes_per_tile = 512; // 128x4 scale tile in memory
    return static_cast<size_t>(tiles_inner) * tiles_outer * bytes_per_tile;
}

int main(int argc, char** argv) {
    // 为了满足 FP4 路径的要求，维度最好都是 16B 对齐友好的，4096 没问题
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    int warmup = 20;
    int iters  = 200;
    if (argc >= 2) warmup = std::atoi(argv[1]);
    if (argc >= 3) iters  = std::atoi(argv[2]);

    std::cout << "Benchmark cuBLASLt NVFP4(E2M1) GEMM with BF16 output\n";
    std::cout << "Compute: D = A * B\n";
    std::cout << "A logical: [" << M << ", " << K << "]\n";
    std::cout << "B logical: [" << K << ", " << N << "]\n";
    std::cout << "D logical: [" << M << ", " << N << "]\n";
    std::cout << "warmup=" << warmup << ", iters=" << iters << "\n";

    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";

    // A: row-major [M,K]
    // B: row-major [K,N]
    std::vector<float> hA_row(static_cast<size_t>(M) * K);
    std::vector<float> hB_row(static_cast<size_t>(K) * N);
    init_f32(hA_row);
    init_f32(hB_row);

    print_sample_f32("Sample A(row-major MxK)", hA_row, 8);
    print_sample_f32("Sample B(row-major KxN)", hB_row, 8);

    // 变成底层 col-major:
    // A_under: col-major [K,M]
    // B_under: col-major [K,N]
    std::vector<float> hA_under_f32, hB_under_f32;
    row_major_to_col_major_underlying(hA_row, M, K, hA_under_f32);
    row_major_to_col_major_underlying(hB_row, K, N, hB_under_f32);

    // 打包成 fp4x2
    std::vector<__nv_fp4x2_e2m1> hA_packed, hB_packed;
    pack_f32_to_fp4x2(hA_under_f32, hA_packed);
    pack_f32_to_fp4x2(hB_under_f32, hB_packed);

    std::vector<__nv_bfloat16> hD(static_cast<size_t>(M) * N);

    // scale buffer: UE4M3，全 1
    const size_t aScaleBytes = fp4_scale_bytes_for_matrix(K, M);
    const size_t bScaleBytes = fp4_scale_bytes_for_matrix(K, N);

    std::vector<__nv_fp8_e4m3> hAScale(aScaleBytes);
    std::vector<__nv_fp8_e4m3> hBScale(bScaleBytes);
    for (size_t i = 0; i < hAScale.size(); ++i) hAScale[i] = __nv_fp8_e4m3(1.0f);
    for (size_t i = 0; i < hBScale.size(); ++i) hBScale[i] = __nv_fp8_e4m3(1.0f);

    void* dA = nullptr;
    void* dB = nullptr;
    __nv_bfloat16* dC = nullptr;
    __nv_bfloat16* dD = nullptr;
    void* dAScale = nullptr;
    void* dBScale = nullptr;

    const size_t bytesA = hA_packed.size() * sizeof(__nv_fp4x2_e2m1);
    const size_t bytesB = hB_packed.size() * sizeof(__nv_fp4x2_e2m1);
    const size_t bytesD = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesD));
    CHECK_CUDA(cudaMalloc(&dD, bytesD));
    CHECK_CUDA(cudaMalloc(&dAScale, aScaleBytes));
    CHECK_CUDA(cudaMalloc(&dBScale, bScaleBytes));

    CHECK_CUDA(cudaMemcpy(dA, hA_packed.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB_packed.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesD));
    CHECK_CUDA(cudaMemset(dD, 0, bytesD));
    CHECK_CUDA(cudaMemcpy(dAScale, hAScale.data(), aScaleBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBScale, hBScale.data(), bScaleBytes, cudaMemcpyHostToDevice));

    cublasLtHandle_t lt;
    CHECK_CUBLAS(cublasLtCreate(&lt));

    // NVFP4 要求:
    // - computeType = CUBLAS_COMPUTE_32F
    // - scaleType   = CUDA_R_32F
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // FP4 要求 TN: A 转置, B 非转置
    const cublasOperation_t transA = CUBLAS_OP_T;
    const cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // A/B 的 scale mode 必须是 VEC16_UE4M3
    const cublasLtMatmulMatrixScale_t fp4ScaleMode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &fp4ScaleMode, sizeof(fp4ScaleMode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &fp4ScaleMode, sizeof(fp4ScaleMode)));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAScale, sizeof(dAScale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBScale, sizeof(dBScale)));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;

    // A underlying: col-major [K, M], lda=K, type=FP4
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, K, M, K));

    // B underlying: col-major [K, N], ldb=K, type=FP4
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, N, K));

    // C/D output: BF16 col-major [M, N], ldc/ldd=M
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, M, N, M));

    const size_t workspaceSize = 64ull * 1024 * 1024;
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof(workspaceSize)));

    constexpr int kAlgoCount = 16;
    cublasLtMatmulHeuristicResult_t heuristic[kAlgoCount];
    int returned = 0;

    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        lt, opDesc, Adesc, Bdesc, Cdesc, Ddesc,
        preference, kAlgoCount, heuristic, &returned));

    if (returned == 0) {
        std::fprintf(stderr, "No NVFP4 heuristic found.\n");
        std::exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    auto run_once = [&](const cublasLtMatmulHeuristicResult_t& h) -> cublasStatus_t {
        return cublasLtMatmul(
            lt,
            opDesc,
            &alpha,
            dA, Adesc,
            dB, Bdesc,
            &beta,
            dC, Cdesc,
            dD, Ddesc,
            &h.algo,
            workspace,
            workspaceSize,
            0);
    };

    int algoIndex = -1;
    for (int i = 0; i < returned; ++i) {
        cublasStatus_t st = run_once(heuristic[i]);
        if (st == CUBLAS_STATUS_SUCCESS) {
            algoIndex = i;
            break;
        }
    }
    if (algoIndex < 0) {
        std::fprintf(stderr, "No runnable NVFP4 heuristic found.\n");
        std::exit(EXIT_FAILURE);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(run_once(heuristic[algoIndex]));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(run_once(heuristic[algoIndex]));
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

    CHECK_CUDA(cudaMemcpy(hD.data(), dD, bytesD, cudaMemcpyDeviceToHost));
    print_sample_bf16("Sample D(bf16,col-major)", hD, 8);

    std::cout << "CPU ref first row: ";
    for (int j = 0; j < 8; ++j) {
        float ref = cpu_ref_rowmajor_A_mul_B(hA_row, hB_under_f32, M, N, K, 0, j);
        std::cout << ref << " ";
    }
    std::cout << "\n";

    std::cout << "GPU first row: ";
    for (int j = 0; j < 8; ++j) {
        // hD is col-major [M,N]
        std::cout << bf16_to_f32(hD[static_cast<size_t>(0) + static_cast<size_t>(j) * M]) << " ";
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
    CHECK_CUDA(cudaFree(dAScale));
    CHECK_CUDA(cudaFree(dBScale));

    return 0;
}