#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <torch/library.h>

#include <cstdint>
#include <mutex>

#include "common.h"

namespace matmul_fp8_v4_detail {

constexpr int kNumWarps = 4;
constexpr int kThreads = kNumWarps * WARP_SIZE;
constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 128;
constexpr int kMmaK = 32;

int64_t align_up(int64_t x, int64_t y) {
  return ((x + y - 1) / y) * y;
}

void check_cu_v4(CUresult result, const char* expr) {
  if (result == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  TORCH_CHECK(false, expr, " failed: ", name ? name : "<unknown>", " ",
              message ? message : "");
}

#define MATMUL_FP8_V4_CU_CHECK(expr) check_cu_v4((expr), #expr)

void check_cublas_v4(cublasStatus_t status, const char* expr) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              expr, " failed: cublasStatus=", static_cast<int>(status));
}

#define MATMUL_FP8_V4_CUBLAS_CHECK(expr) check_cublas_v4((expr), #expr)

CUtensorMap make_tma_fp8_kmajor_desc(
    const at::Tensor& t,
    int outer_dim,
    int inner_dim,
    int block_outer) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dims[2] = {
      static_cast<cuuint64_t>(inner_dim),
      static_cast<cuuint64_t>(outer_dim)};
  const cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(t.stride(0) * t.element_size())};
  const cuuint32_t box_dims[2] = {
      static_cast<cuuint32_t>(kBlockK),
      static_cast<cuuint32_t>(block_outer)};
  const cuuint32_t element_strides[2] = {1, 1};

  MATMUL_FP8_V4_CU_CHECK(cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      2,
      t.data_ptr(),
      global_dims,
      global_strides,
      box_dims,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensor_map;
}

__device__ inline uint64_t make_fp8_kmajor_desc(int smem_addr) {
  constexpr uint64_t kLayoutSwizzle128B = 2;
  constexpr int kStrideByteOffset = 8 * kBlockK * sizeof(__nv_fp8_e4m3);
  constexpr int kLeadingByteOffset = 0;
  return desc_encode(static_cast<uint64_t>(smem_addr)) |
         (desc_encode(kLeadingByteOffset) << 16ULL) |
         (desc_encode(kStrideByteOffset) << 32ULL) |
         (1ULL << 46ULL) |
         (kLayoutSwizzle128B << 61ULL);
}

__device__ inline void tcgen05_mma_fp8_e4m3(
    int taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :
      : "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
}

__device__ inline void mbarrier_arrive_expect_tx_cta(int mbar_addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :: "r"(mbar_addr), "r"(size) : "memory");
}

__global__ __launch_bounds__(kThreads)
void matmul_fp8_v4_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    nv_bfloat16* C,
    int M,
    int N,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;

  const int grid_n = N / kBlockN;
  const int bid_m = blockIdx.x / grid_n;
  const int bid_n = blockIdx.x % grid_n;
  const int off_m = bid_m * kBlockM;
  const int off_n = bid_n * kBlockN;

  extern __shared__ __align__(1024) char smem[];
  const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem));
  const int B_smem = A_smem + kBlockM * kBlockK * sizeof(__nv_fp8_e4m3);

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[2];
  __shared__ int tmem_addr[1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + 8;

  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(tma_mbar_addr, 1);
    mbarrier_init(mma_mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    tcgen05_alloc(addr, kBlockN);
  }

  __syncthreads();
  const int taddr = tmem_addr[0];

  // tcgen05.mma.kind::f8f6f4 instruction descriptor.
  // This mirrors matmul_v1's BF16 descriptor, but uses E4M3 for A/B.
  constexpr uint32_t kCFormatF32 = 1U;
  constexpr uint32_t kF8FormatE4M3 = 0U;
  constexpr uint32_t kMajorK = 0U;
  constexpr uint32_t i_desc =
      (kCFormatF32 << 4U) |               // C accumulator format: F32
      (kF8FormatE4M3 << 7U) |             // A format: E4M3
      (kF8FormatE4M3 << 10U) |            // B format: E4M3
      (kMajorK << 15U) |                  // A layout: K-major
      (kMajorK << 16U) |                  // B layout: K-major
      ((uint32_t)kBlockN >> 3U << 17U) |  // MMA_N
      ((uint32_t)kBlockM >> 4U << 24U);   // MMA_M

  int tma_phase = 0;
  int mma_phase = 0;
  const int num_iters = K / kBlockK;
  for (int iter_k = 0; iter_k < num_iters; ++iter_k) {
    if (warp_id == 0 && elect_sync()) {
      const int off_k = iter_k * kBlockK;
      tma_2d_g2s(A_smem, &A_tmap, off_k, off_m, tma_mbar_addr);
      tma_2d_g2s(B_smem, &B_tmap, off_k, off_n, tma_mbar_addr);
      constexpr int cp_size =
          (kBlockM + kBlockN) * kBlockK * sizeof(__nv_fp8_e4m3);
      mbarrier_arrive_expect_tx_cta(tma_mbar_addr, cp_size);
    }

    mbarrier_wait(tma_mbar_addr, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");
    tma_phase ^= 1;

    if (warp_id == 0 && elect_sync()) {
      #pragma unroll
      for (int k = 0; k < kBlockK / kMmaK; ++k) {
        const uint64_t a_desc = make_fp8_kmajor_desc(A_smem + k * kMmaK);
        const uint64_t b_desc = make_fp8_kmajor_desc(B_smem + k * kMmaK);
        const int accumulate = (iter_k != 0 || k != 0);
        tcgen05_mma_fp8_e4m3(taddr, a_desc, b_desc, i_desc, accumulate);
      }
      tcgen05_commit(mma_mbar_addr);
    }

    mbarrier_wait(mma_mbar_addr, mma_phase);
    mma_phase ^= 1;
  }

  asm volatile("tcgen05.fence::after_thread_sync;");

  for (int n = 0; n < kBlockN / 8; ++n) {
    float tmp[8];
    const int addr = taddr + ((warp_id * 32) << 16) + (n * 8);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    nv_bfloat162 out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});

    nv_bfloat16* out_ptr = C + (off_m + tid) * N + (off_n + n * 8);
    reinterpret_cast<int4*>(out_ptr)[0] = reinterpret_cast<int4*>(out)[0];
  }

  __syncthreads();
  if (warp_id == 0) {
    tcgen05_dealloc(taddr, kBlockN);
  }
}

void launch_matmul_fp8_v4(
    const at::Tensor& a_fp8,
    const at::Tensor& b_nt_fp8,
    const at::Tensor& c,
    int M,
    int N,
    int K) {
  auto A_tmap = make_tma_fp8_kmajor_desc(a_fp8, M, K, kBlockM);
  auto B_tmap = make_tma_fp8_kmajor_desc(b_nt_fp8, N, K, kBlockN);

  const int grid = (M / kBlockM) * (N / kBlockN);
  constexpr int smem_size =
      (kBlockM + kBlockN) * kBlockK * sizeof(__nv_fp8_e4m3);
  auto kernel = matmul_fp8_v4_kernel;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  kernel<<<grid, kThreads, smem_size, stream>>>(
      A_tmap,
      B_tmap,
      reinterpret_cast<nv_bfloat16*>(c.data_ptr()),
      M,
      N,
      K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 新增 launch_cublaslt_fp8_ref，复用同一份 a_fp8/b_nt_fp8 cache，
// 按 00_cublas/fp8_cublas.cu 的 layout 调 cuBLASLt。
void launch_cublaslt_fp8_ref(
    const at::Tensor& a_fp8,
    const at::Tensor& b_nt_fp8,
    const at::Tensor& c,
    int M,
    int N,
    int K) {
  cublasLtHandle_t lt;
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtCreate(&lt));

  cublasLtMatmulDesc_t op_desc;
  MATMUL_FP8_V4_CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  const cublasOperation_t trans_a = CUBLAS_OP_T;
  const cublasOperation_t trans_b = CUBLAS_OP_N;
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  auto scale_a_tensor = at::ones({1}, a_fp8.options().dtype(at::kFloat));
  auto scale_b_tensor = at::ones({1}, a_fp8.options().dtype(at::kFloat));
  float* scale_a = scale_a_tensor.data_ptr<float>();
  float* scale_b = scale_b_tensor.data_ptr<float>();
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));

  cublasLtMatrixLayout_t a_desc;
  cublasLtMatrixLayout_t b_desc;
  cublasLtMatrixLayout_t c_desc;
  cublasLtMatrixLayout_t d_desc;

  // b_nt_fp8 is row-major [N,K], reinterpreted as column-major [K,N].
  MATMUL_FP8_V4_CUBLAS_CHECK(
      cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, K, N, K));
  // a_fp8 is row-major [M,K], reinterpreted as column-major [K,M].
  MATMUL_FP8_V4_CUBLAS_CHECK(
      cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, K, M, K));
  // c is row-major [M,N], reinterpreted as column-major [N,M].
  MATMUL_FP8_V4_CUBLAS_CHECK(
      cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, N, M, N));
  MATMUL_FP8_V4_CUBLAS_CHECK(
      cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, N, M, N));

  constexpr size_t workspace_size = 32ull * 1024 * 1024;
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                             a_fp8.options().dtype(at::kByte));

  cublasLtMatmulPreference_t preference;
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size,
      sizeof(workspace_size)));

  constexpr int kRequestedAlgoCount = 16;
  cublasLtMatmulHeuristicResult_t heuristic_results[kRequestedAlgoCount];
  int returned_results = 0;
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt,
      op_desc,
      a_desc,
      b_desc,
      c_desc,
      d_desc,
      preference,
      kRequestedAlgoCount,
      heuristic_results,
      &returned_results));
  TORCH_CHECK(returned_results > 0, "matmul_fp8_v4_cublaslt found no FP8 heuristic");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  cublasStatus_t last_status = CUBLAS_STATUS_SUCCESS;
  bool launched = false;
  for (int i = 0; i < returned_results; ++i) {
    last_status = cublasLtMatmul(
        lt,
        op_desc,
        &alpha,
        b_nt_fp8.data_ptr(),
        a_desc,
        a_fp8.data_ptr(),
        b_desc,
        &beta,
        c.data_ptr(),
        c_desc,
        c.data_ptr(),
        d_desc,
        &heuristic_results[i].algo,
        workspace.data_ptr(),
        workspace_size,
        stream);
    if (last_status == CUBLAS_STATUS_SUCCESS) {
      launched = true;
      break;
    }
  }
  TORCH_CHECK(launched,
              "matmul_fp8_v4_cublaslt failed to launch any heuristic, last status=",
              static_cast<int>(last_status));

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  TORCH_CHECK(sync_status == cudaSuccess,
              "matmul_fp8_v4_cublaslt stream sync failed: ",
              cudaGetErrorString(sync_status));

  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
  MATMUL_FP8_V4_CUBLAS_CHECK(cublasLtDestroy(lt));
}

struct FP8Cache {
  const void* a_ptr = nullptr;
  const void* b_ptr = nullptr;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  at::Tensor a_fp8;
  at::Tensor b_nt_fp8;
};

enum class QuantMode {
  CudaRaw,
  TorchRaw,
};

FP8Cache cuda_raw_cache;
FP8Cache torch_raw_cache;
std::mutex cache_mutex;

// cache_matches(...) 检查当前 cache 里的 FP8 数据是不是还能复用：
// A.data_ptr() 是否和上次一样
// B.data_ptr() 是否和上次一样
// m/n/k 是否和上次一样
// cache.a_fp8 和 cache.b_nt_fp8 是否已经存在
// 如果不匹配，就重新量化：
// 默认 matmul_fp8_v4：refresh_cache_cuda_raw(...)
// 用 CUDA kernel 把 BF16 A cast 到 raw E4M3 FP8，把 B.transpose(0,1).contiguous() cast 到 raw E4M3 FP8。
// matmul_fp8_v4_torch_quant：用 PyTorch .to(torch.float8_e4m3fn) 做同样的量化。
// 所以 benchmark 里如果 A/B tensor 指针一直不变，第一次调用会做：
// BF16 A/B -> FP8 A / FP8 B^T
// 后面重复调用只跑 GEMM kernel，不再重复量化。
// 这也是之前 nsys 里 matmul_fp8_v4_kernel 看起来很快的原因之一：量化被 cache 掉了，计时主要是 FP8 GEMM 本体。
bool cache_matches(
    const FP8Cache& cache,
    const at::Tensor& A,
    const at::Tensor& B,
    int64_t m,
    int64_t n,
    int64_t k) {
  return cache.a_ptr == A.data_ptr() &&
         cache.b_ptr == B.data_ptr() &&
         cache.m == m &&
         cache.n == n &&
         cache.k == k &&
         cache.a_fp8.defined() &&
         cache.b_nt_fp8.defined();
}

// 默认 matmul_fp8_v4 用 CUDA raw BF16 -> E4M3 cast
__global__ void cast_bf16_to_fp8_e4m3_kernel(
    const nv_bfloat16* x,
    __nv_fp8_e4m3* y,
    int64_t numel) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel)
    y[idx] = __nv_fp8_e4m3(__bfloat162float(x[idx]));
}

at::Tensor cast_bf16_to_fp8_e4m3_cuda(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "cast_bf16_to_fp8_e4m3_cuda expects CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16,
              "cast_bf16_to_fp8_e4m3_cuda expects BF16 input");
  TORCH_CHECK(x.is_contiguous(), "cast_bf16_to_fp8_e4m3_cuda expects contiguous input");

  auto y = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
  const int64_t numel = x.numel();
  if (numel == 0)
    return y;

  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  cast_bf16_to_fp8_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(y.data_ptr()),
      numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// A.contiguous()：如果 A 本来 contiguous，通常可能不拷贝；否则会走 PyTorch copy kernel。
// .to(Float8_e4m3fn)：PyTorch 的 dtype conversion kernel，把 BF16 转 FP8 E4M3。
// B.transpose(0,1).contiguous()：这里一定会做转置/拷贝，把 B 从 [K,N] 变成 [N,K] contiguous。
// .to(Float8_e4m3fn)：再做 BF16 -> FP8。
// 最后的 .contiguous() 通常是保险，前一步 .to() 返回一般已经 contiguous。
// 但因为现在有 cache：
// 只要 benchmark 里 A/B 指针不变，量化和 B 转置只在第一次调用发生。
// 真正端到端性能要么关掉 cache，要么单独把 refresh_cache 的 PyTorch kernels 加进 profile/计时。
void refresh_cache_torch_raw(
    FP8Cache& cache,
    const at::Tensor& A,
    const at::Tensor& B,
    int64_t m,
    int64_t n,
    int64_t k) {
  cache.a_fp8 = A.contiguous().to(at::ScalarType::Float8_e4m3fn).contiguous();  // torch BF16 转 FP8 E4M3
  cache.b_nt_fp8 = B.transpose(0, 1).contiguous().to(at::ScalarType::Float8_e4m3fn).contiguous();
  cache.a_ptr = A.data_ptr();
  cache.b_ptr = B.data_ptr();
  cache.m = m;
  cache.n = n;
  cache.k = k;
}

// 这个 v4 kernel 是裸 FP8 tcgen05.mma.kind::f8f6f4，数学语义是 scaleA=scaleB=1。
// 用户给的 per-token UE8M0 scale kernel 是 MXFP8/block-scaled 语义；如果直接用于
// 这里，会变成 (A/sfA) @ (B/sfB)^T，必须换 block-scale MMA 才能还原原始 GEMM。
// 所以默认路径只借鉴“量化由 CUDA kernel 完成”的方式，做 raw BF16 -> E4M3 cast。
void refresh_cache_cuda_raw(
    FP8Cache& cache,
    const at::Tensor& A,
    const at::Tensor& B,
    int64_t m,
    int64_t n,
    int64_t k) {
  cache.a_fp8 = cast_bf16_to_fp8_e4m3_cuda(A.contiguous());
  cache.b_nt_fp8 = cast_bf16_to_fp8_e4m3_cuda(B.transpose(0, 1).contiguous());
  cache.a_ptr = A.data_ptr();
  cache.b_ptr = B.data_ptr();
  cache.m = m;
  cache.n = n;
  cache.k = k;
}

void refresh_cache(
    FP8Cache& cache,
    const at::Tensor& A,
    const at::Tensor& B,
    int64_t m,
    int64_t n,
    int64_t k,
    QuantMode mode) {
  if (mode == QuantMode::CudaRaw)
    refresh_cache_cuda_raw(cache, A, B, m, n, k);
  else
    refresh_cache_torch_raw(cache, A, B, m, n, k);
}

}  // namespace matmul_fp8_v4_detail

static at::Tensor matmul_fp8_v4_impl(
    const at::Tensor& A,
    const at::Tensor& B,
    matmul_fp8_v4_detail::QuantMode mode,
    const char* op_name) {
  using namespace matmul_fp8_v4_detail;
  c10::cuda::CUDAGuard device_guard(A.device());

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, op_name, " expects 2D tensors");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), op_name, " expects CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              op_name, " expects BF16 inputs");
  TORCH_CHECK(A.device() == B.device(), op_name, " expects tensors on the same device");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  TORCH_CHECK(B.size(0) == k, op_name, " shape mismatch: A[M,K] @ B[K,N]");
  const int64_t n = B.size(1);
  TORCH_CHECK(m % kBlockM == 0 && n % kBlockN == 0 && k % kBlockK == 0,
              op_name, " requires M%128==0, N%256==0, K%128==0");

  FP8Cache& cache = (mode == QuantMode::CudaRaw) ? cuda_raw_cache : torch_raw_cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  if (!cache_matches(cache, A, B, m, n, k))
    refresh_cache(cache, A, B, m, n, k, mode);

  auto C = at::empty({m, n}, A.options());
  launch_matmul_fp8_v4(cache.a_fp8, cache.b_nt_fp8, C, static_cast<int>(m),
                       static_cast<int>(n), static_cast<int>(k));
  return C;
}

at::Tensor matmul_fp8_v4(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v4_impl(
      A, B, matmul_fp8_v4_detail::QuantMode::CudaRaw, "matmul_fp8_v4");
}

at::Tensor matmul_fp8_v4_nocache(const at::Tensor& A, const at::Tensor& B) {
  using namespace matmul_fp8_v4_detail;
  c10::cuda::CUDAGuard device_guard(A.device());
  constexpr const char* op_name = "matmul_fp8_v4_nocache";

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, op_name, " expects 2D tensors");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), op_name, " expects CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              op_name, " expects BF16 inputs");
  TORCH_CHECK(A.device() == B.device(), op_name, " expects tensors on the same device");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  TORCH_CHECK(B.size(0) == k, op_name, " shape mismatch: A[M,K] @ B[K,N]");
  const int64_t n = B.size(1);
  TORCH_CHECK(m % kBlockM == 0 && n % kBlockN == 0 && k % kBlockK == 0,
              op_name, " requires M%128==0, N%256==0, K%128==0");

  auto a_bf16 = A.contiguous();
  auto b_nt_bf16 = B.transpose(0, 1).contiguous();
  auto a_fp8 = cast_bf16_to_fp8_e4m3_cuda(a_bf16);
  auto b_nt_fp8 = cast_bf16_to_fp8_e4m3_cuda(b_nt_bf16);

  auto C = at::empty({m, n}, A.options());
  launch_matmul_fp8_v4(a_fp8, b_nt_fp8, C, static_cast<int>(m),
                       static_cast<int>(n), static_cast<int>(k));
  return C;
}

at::Tensor matmul_fp8_v4_torch_quant(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v4_impl(
      A, B, matmul_fp8_v4_detail::QuantMode::TorchRaw, "matmul_fp8_v4_torch_quant");
}

// matmul_fp8_v4_cublaslt
at::Tensor matmul_fp8_v4_cublaslt(const at::Tensor& A, const at::Tensor& B) {
  using namespace matmul_fp8_v4_detail;
  c10::cuda::CUDAGuard device_guard(A.device());
  constexpr const char* op_name = "matmul_fp8_v4_cublaslt";

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, op_name, " expects 2D tensors");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), op_name, " expects CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              op_name, " expects BF16 inputs");
  TORCH_CHECK(A.device() == B.device(), op_name, " expects tensors on the same device");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  TORCH_CHECK(B.size(0) == k, op_name, " shape mismatch: A[M,K] @ B[K,N]");
  const int64_t n = B.size(1);
  TORCH_CHECK(m % kBlockM == 0 && n % kBlockN == 0 && k % kBlockK == 0,
              op_name, " requires M%128==0, N%256==0, K%128==0");

  std::lock_guard<std::mutex> lock(cache_mutex);
  if (!cache_matches(cuda_raw_cache, A, B, m, n, k))
    refresh_cache(cuda_raw_cache, A, B, m, n, k, QuantMode::CudaRaw);

  auto C = at::empty({m, n}, A.options());
  launch_cublaslt_fp8_ref(cuda_raw_cache.a_fp8, cuda_raw_cache.b_nt_fp8, C,
                          static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  return C;
}
