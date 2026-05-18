#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>

#include "common.h"

namespace matmul_fp8_v7_detail {

constexpr int kNumWarps = 4;
constexpr int kThreads = kNumWarps * WARP_SIZE;
constexpr int kBlockM = 128;
constexpr int kMmaK = 32;
constexpr int kTmaK = 128;

void check_cu_v7(CUresult result, const char* expr) {
  if (result == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  TORCH_CHECK(false, expr, " failed: ", name ? name : "<unknown>", " ",
              message ? message : "");
}

#define MATMUL_FP8_V7_CU_CHECK(expr) check_cu_v7((expr), #expr)

CUtensorMap make_tma_fp8_3d_128b_desc(
    const at::Tensor& t,
    int outer_dim,
    int inner_dim,
    int block_outer,
    int block_k) {
  TORCH_CHECK(inner_dim % kTmaK == 0, "FP8 3D TMA requires K%128==0");
  TORCH_CHECK(block_k % kTmaK == 0, "FP8 3D TMA requires BLOCK_K%128==0");

  CUtensorMap tensor_map;
  const cuuint64_t global_dims[3] = {
      static_cast<cuuint64_t>(kTmaK),
      static_cast<cuuint64_t>(outer_dim),
      static_cast<cuuint64_t>(inner_dim / kTmaK)};
  const cuuint64_t global_strides[2] = {
      static_cast<cuuint64_t>(t.stride(0) * t.element_size()),
      static_cast<cuuint64_t>(kTmaK * t.element_size())};
  const cuuint32_t box_dims[3] = {
      static_cast<cuuint32_t>(kTmaK),
      static_cast<cuuint32_t>(block_outer),
      static_cast<cuuint32_t>(block_k / kTmaK)};
  const cuuint32_t element_strides[3] = {1, 1, 1};

  MATMUL_FP8_V7_CU_CHECK(cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      3,
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

__device__ inline uint64_t make_fp8_128b_desc(int smem_addr) {
  constexpr uint64_t kLayoutSwizzle128B = 2;
  constexpr int kStrideByteOffset = 8 * kTmaK * sizeof(__nv_fp8_e4m3);
  constexpr int kLeadingByteOffset = 0;
  return desc_encode(static_cast<uint64_t>(smem_addr)) |
         (desc_encode(kLeadingByteOffset) << 16ULL) |
         (desc_encode(kStrideByteOffset) << 32ULL) |
         (1ULL << 46ULL) |
         (kLayoutSwizzle128B << 61ULL);
}

template <int CTA_GROUP>
__device__ inline void tcgen05_mma_fp8_e4m3(
    int taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int enable_input_d) {
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :
        : "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
  } else {
    static_assert(CTA_GROUP == 2);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :
        : "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
  }
}

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, int NUM_STAGES>
__global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(kThreads)
void matmul_fp8_v7_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    nv_bfloat16* C,
    int M,
    int N,
    int K) {
  static_assert(BLOCK_K % kTmaK == 0);
  static_assert(BLOCK_N % CTA_GROUP == 0);
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2);
  static_assert(NUM_STAGES >= 1);

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int cta_rank = blockIdx.x % CTA_GROUP;

  const int grid_n = N / BLOCK_N;
  const int bid = blockIdx.x;
  const int bid_m = bid / (grid_n * CTA_GROUP) * CTA_GROUP + (bid % CTA_GROUP);
  const int bid_n = (bid / CTA_GROUP) % grid_n;
  const int off_m = bid_m * kBlockM;
  const int off_n = bid_n * BLOCK_N;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = kBlockM * BLOCK_K * sizeof(__nv_fp8_e4m3);
  constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(__nv_fp8_e4m3);
  constexpr int Stage_size = A_size + B_size;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
  __shared__ int tmem_addr[1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  if (warp_id == 0 && elect_sync()) {
    #pragma unroll
    for (int i = 0; i < NUM_STAGES; ++i) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }
    mbarrier_init(mainloop_mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    tcgen05_alloc<CTA_GROUP>(addr, BLOCK_N);
  }

  if constexpr (CTA_GROUP > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  } else {
    __syncthreads();
  }
  const int taddr = tmem_addr[0];

  constexpr int MMA_M = kBlockM * CTA_GROUP;
  constexpr uint32_t kCFormatF32 = 1U;
  constexpr uint32_t kF8FormatE4M3 = 0U;
  constexpr uint32_t kMajorK = 0U;
  constexpr uint32_t i_desc =
      (kCFormatF32 << 4U) |
      (kF8FormatE4M3 << 7U) |
      (kF8FormatE4M3 << 10U) |
      (kMajorK << 15U) |
      (kMajorK << 16U) |
      ((uint32_t)BLOCK_N >> 3U << 17U) |
      ((uint32_t)MMA_M >> 4U << 24U);

  int phase = 0;

  auto load = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;
    mbarrier_wait(mma_mbar_addr + stage_id * 8, phase ^ 1);
    if (stage_id == NUM_STAGES - 1)
      phase ^= 1;

    const int local_tma_mbar = tma_mbar_addr + stage_id * 8;
    const int mbar_addr = CTA_GROUP > 1 ? (local_tma_mbar & 0xFEFFFFFF) : local_tma_mbar;
    const int A_smem = smem + stage_id * Stage_size;
    const int B_smem = A_smem + A_size;
    const int off_k = iter_k * BLOCK_K;

    tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, off_k / kTmaK, mbar_addr);
    tma_3d_gmem2smem<CTA_GROUP>(
        B_smem,
        &B_tmap,
        0,
        off_n + cta_rank * (BLOCK_N / CTA_GROUP),
        off_k / kTmaK,
        mbar_addr);
    mbarrier_arrive_expect_tx(mbar_addr, Stage_size);
  };

  auto compute = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;
    mbarrier_wait(tma_mbar_addr + stage_id * 8, phase);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (stage_id == NUM_STAGES - 1)
      phase ^= 1;

    const int A_smem = smem + stage_id * Stage_size;
    const int B_smem = A_smem + A_size;

    int mma_idx = 0;
    #pragma unroll
    for (int k1 = 0; k1 < BLOCK_K / kTmaK; ++k1) {
      #pragma unroll
      for (int k2 = 0; k2 < kTmaK / kMmaK; ++k2) {
        const uint64_t a_desc =
            make_fp8_128b_desc(A_smem + k1 * kBlockM * kTmaK + k2 * kMmaK);
        const uint64_t b_desc =
            make_fp8_128b_desc(B_smem + k1 * (BLOCK_N / CTA_GROUP) * kTmaK + k2 * kMmaK);
        const int accumulate = (iter_k != 0 || mma_idx != 0);
        tcgen05_mma_fp8_e4m3<CTA_GROUP>(taddr, a_desc, b_desc, i_desc, accumulate);
        ++mma_idx;
      }
    }

    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    if constexpr (CTA_GROUP > 1)
      tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + stage_id * 8, cta_mask);
    else
      tcgen05_commit<CTA_GROUP>(mma_mbar_addr + stage_id * 8);
  };

  const int num_iters = K / BLOCK_K;
  if (warp_id == 0 && elect_sync()) {
    for (int iter_k = 0; iter_k < num_iters; ++iter_k)
      load(iter_k);
  } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
    for (int iter_k = 0; iter_k < num_iters; ++iter_k)
      compute(iter_k);

    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    if constexpr (CTA_GROUP > 1)
      tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr, cta_mask);
    else
      tcgen05_commit<CTA_GROUP>(mainloop_mbar_addr);
  }

  __syncthreads();
  mbarrier_wait(mainloop_mbar_addr, 0);
  asm volatile("tcgen05.fence::after_thread_sync;");

  for (int n = 0; n < BLOCK_N / 8; ++n) {
    float tmp[8];
    const int addr = taddr + ((cta_rank * kBlockM + warp_id * 32) << 16) + (n * 8);
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

  if constexpr (CTA_GROUP > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  } else {
    __syncthreads();
  }
  if (warp_id == 0)
    tcgen05_dealloc<CTA_GROUP>(taddr, BLOCK_N);
}

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, int NUM_STAGES>
void launch_matmul_fp8_v7(
    const at::Tensor& a_fp8,
    const at::Tensor& b_nt_fp8,
    const at::Tensor& c,
    int M,
    int N,
    int K) {
  auto A_tmap = make_tma_fp8_3d_128b_desc(a_fp8, M, K, kBlockM, BLOCK_K);
  auto B_tmap =
      make_tma_fp8_3d_128b_desc(b_nt_fp8, N, K, BLOCK_N / CTA_GROUP, BLOCK_K);

  const int grid = (M / kBlockM) * (N / BLOCK_N);
  constexpr int smem_size =
      NUM_STAGES * (kBlockM + BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(__nv_fp8_e4m3);
  auto kernel = matmul_fp8_v7_kernel<BLOCK_N, BLOCK_K, CTA_GROUP, NUM_STAGES>;
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

}  // namespace matmul_fp8_v7_detail

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, int NUM_STAGES>
static at::Tensor matmul_fp8_v7_impl(
    const at::Tensor& A,
    const at::Tensor& B,
    const char* op_name) {
  using namespace matmul_fp8_v7_detail;
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
  TORCH_CHECK(m % (kBlockM * CTA_GROUP) == 0 && n % BLOCK_N == 0 && k % BLOCK_K == 0,
              op_name, " requires M%(128*CTA_GROUP)==0, N%BLOCK_N==0, K%BLOCK_K==0");

  auto a_bf16 = A.contiguous();
  auto b_nt_bf16 = B.transpose(0, 1).contiguous();
  auto a_fp8 = cast_bf16_to_fp8_e4m3_cuda(a_bf16);
  auto b_nt_fp8 = cast_bf16_to_fp8_e4m3_cuda(b_nt_bf16);

  auto C = at::empty({m, n}, A.options());
  launch_matmul_fp8_v7<BLOCK_N, BLOCK_K, CTA_GROUP, NUM_STAGES>(
      a_fp8, b_nt_fp8, C,
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  return C;
}

at::Tensor matmul_fp8_v7(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v7_impl<256, 128, 2, 7>(A, B, "matmul_fp8_v7");
}

at::Tensor matmul_fp8_v7_n256_k128_c2_s4(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v7_impl<256, 128, 2, 4>(A, B, "matmul_fp8_v7_n256_k128_c2_s4");
}

at::Tensor matmul_fp8_v7_n256_k128_c2_s5(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v7_impl<256, 128, 2, 5>(A, B, "matmul_fp8_v7_n256_k128_c2_s5");
}

at::Tensor matmul_fp8_v7_n256_k128_c2_s6(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v7_impl<256, 128, 2, 6>(A, B, "matmul_fp8_v7_n256_k128_c2_s6");
}

at::Tensor matmul_fp8_v7_n256_k128_c2_s7(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v7_impl<256, 128, 2, 7>(A, B, "matmul_fp8_v7_n256_k128_c2_s7");
}

at::Tensor matmul_fp8_v7_n256_k256_c2_s2(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v7_impl<256, 256, 2, 2>(A, B, "matmul_fp8_v7_n256_k256_c2_s2");
}

at::Tensor matmul_fp8_v7_n256_k256_c2_s3(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v7_impl<256, 256, 2, 3>(A, B, "matmul_fp8_v7_n256_k256_c2_s3");
}
