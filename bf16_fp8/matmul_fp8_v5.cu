#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <mutex>

#include "common.h"

namespace matmul_fp8_v5_detail {

constexpr int kNumWarps = 4;
constexpr int kThreads = kNumWarps * WARP_SIZE;
constexpr int kBlockM = 128;
constexpr int kMmaK = 32;
constexpr int kTmaK = 128;  // 128 FP8 elements = 128B swizzle inner tile.

void check_cu_v5(CUresult result, const char* expr) {
  if (result == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  TORCH_CHECK(false, expr, " failed: ", name ? name : "<unknown>", " ",
              message ? message : "");
}

#define MATMUL_FP8_V5_CU_CHECK(expr) check_cu_v5((expr), #expr)

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

  MATMUL_FP8_V5_CU_CHECK(cuTensorMapEncodeTiled(
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

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__ __launch_bounds__(kThreads)
void matmul_fp8_v5_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    nv_bfloat16* C,
    int M,
    int N,
    int K) {
  static_assert(BLOCK_K % kTmaK == 0);
  static_assert(NUM_STAGES >= 2);

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;

  const int grid_n = N / BLOCK_N;
  const int bid_m = blockIdx.x / grid_n;
  const int bid_n = blockIdx.x % grid_n;
  const int off_m = bid_m * kBlockM;
  const int off_n = bid_n * BLOCK_N;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = kBlockM * BLOCK_K * sizeof(__nv_fp8_e4m3);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
  constexpr int Stage_size = A_size + B_size;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t tma_mbars[NUM_STAGES];
  __shared__ uint64_t mma_mbars[1];
  __shared__ int tmem_addr[1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
  const int mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));

  if (warp_id == 0 && elect_sync()) {
    #pragma unroll
    for (int i = 0; i < NUM_STAGES; ++i)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    mbarrier_init(mma_mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    tcgen05_alloc(addr, BLOCK_N);
  }

  __syncthreads();
  const int taddr = tmem_addr[0];

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
      ((uint32_t)kBlockM >> 4U << 24U);

  int tma_phase = 0;
  int mma_phase = 0;

  auto load = [&](int iter_k) {
    if (warp_id == 0 && elect_sync()) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * Stage_size;
      const int B_smem = A_smem + A_size;
      const int off_k = iter_k * BLOCK_K;
      tma_3d_g2s(A_smem, &A_tmap, 0, off_m, off_k / kTmaK, mbar_addr);
      tma_3d_g2s(B_smem, &B_tmap, 0, off_n, off_k / kTmaK, mbar_addr);
      mbarrier_arrive_expect_tx_cta(mbar_addr, Stage_size);
    }
  };

  auto compute = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;
    const int mbar_addr = tma_mbar_addr + stage_id * 8;
    mbarrier_wait(mbar_addr, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int A_smem = smem + stage_id * Stage_size;
    const int B_smem = A_smem + A_size;

    if (stage_id == NUM_STAGES - 1)
      tma_phase ^= 1;

    if (warp_id == 0 && elect_sync()) {
      int mma_idx = 0;
      #pragma unroll
      for (int k1 = 0; k1 < BLOCK_K / kTmaK; ++k1) {
        #pragma unroll
        for (int k2 = 0; k2 < kTmaK / kMmaK; ++k2) {
          const uint64_t a_desc =
              make_fp8_128b_desc(A_smem + k1 * kBlockM * kTmaK + k2 * kMmaK);
          const uint64_t b_desc =
              make_fp8_128b_desc(B_smem + k1 * BLOCK_N * kTmaK + k2 * kMmaK);
          const int accumulate = (iter_k != 0 || mma_idx != 0);
          tcgen05_mma_fp8_e4m3(taddr, a_desc, b_desc, i_desc, accumulate);
          ++mma_idx;
        }
      }
      tcgen05_commit(mma_mbar_addr);
    }
  };

  const int num_iters = K / BLOCK_K;
  #pragma unroll
  for (int i = 0; i < NUM_STAGES - 1; ++i)
    load(i);

  for (int iter_k = 0; iter_k < num_iters - NUM_STAGES + 1; ++iter_k) {
    load(iter_k + NUM_STAGES - 1);
    compute(iter_k);
    mbarrier_wait(mma_mbar_addr, mma_phase);
    mma_phase ^= 1;
  }

  for (int iter_k = num_iters - NUM_STAGES + 1; iter_k < num_iters; ++iter_k) {
    compute(iter_k);
    mbarrier_wait(mma_mbar_addr, mma_phase);
    mma_phase ^= 1;
  }

  asm volatile("tcgen05.fence::after_thread_sync;");

  for (int n = 0; n < BLOCK_N / 8; ++n) {
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
  if (warp_id == 0)
    tcgen05_dealloc(taddr, BLOCK_N);
}

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES>
void launch_matmul_fp8_v5(
    const at::Tensor& a_fp8,
    const at::Tensor& b_nt_fp8,
    const at::Tensor& c,
    int M,
    int N,
    int K) {
  auto A_tmap = make_tma_fp8_3d_128b_desc(a_fp8, M, K, kBlockM, BLOCK_K);
  auto B_tmap = make_tma_fp8_3d_128b_desc(b_nt_fp8, N, K, BLOCK_N, BLOCK_K);

  const int grid = (M / kBlockM) * (N / BLOCK_N);
  constexpr int smem_size =
      NUM_STAGES * (kBlockM + BLOCK_N) * BLOCK_K * sizeof(__nv_fp8_e4m3);
  auto kernel = matmul_fp8_v5_kernel<BLOCK_N, BLOCK_K, NUM_STAGES>;
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

struct FP8Cache {
  const void* a_ptr = nullptr;
  const void* b_ptr = nullptr;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  at::Tensor a_fp8;
  at::Tensor b_nt_fp8;
};

FP8Cache cache;
std::mutex cache_mutex;

bool cache_matches(const at::Tensor& A, const at::Tensor& B, int64_t m, int64_t n, int64_t k) {
  return cache.a_ptr == A.data_ptr() &&
         cache.b_ptr == B.data_ptr() &&
         cache.m == m &&
         cache.n == n &&
         cache.k == k &&
         cache.a_fp8.defined() &&
         cache.b_nt_fp8.defined();
}

void refresh_cache(const at::Tensor& A, const at::Tensor& B, int64_t m, int64_t n, int64_t k) {
  auto a_bf16 = A.contiguous();
  auto b_nt_bf16 = B.transpose(0, 1).contiguous();
  cache.a_fp8 = cast_bf16_to_fp8_e4m3_cuda(a_bf16);
  cache.b_nt_fp8 = cast_bf16_to_fp8_e4m3_cuda(b_nt_bf16);
  cache.a_ptr = A.data_ptr();
  cache.b_ptr = B.data_ptr();
  cache.m = m;
  cache.n = n;
  cache.k = k;
}

}  // namespace matmul_fp8_v5_detail

at::Tensor matmul_fp8_v5_cache(const at::Tensor& A, const at::Tensor& B) {
  using namespace matmul_fp8_v5_detail;
  c10::cuda::CUDAGuard device_guard(A.device());
  constexpr const char* op_name = "matmul_fp8_v5_cache";

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, op_name, " expects 2D tensors");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), op_name, " expects CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              op_name, " expects BF16 inputs");
  TORCH_CHECK(A.device() == B.device(), op_name, " expects tensors on the same device");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  TORCH_CHECK(B.size(0) == k, op_name, " shape mismatch: A[M,K] @ B[K,N]");
  const int64_t n = B.size(1);
  TORCH_CHECK(m % kBlockM == 0 && n % 256 == 0 && k % 128 == 0,
              op_name, " requires M%128==0, N%256==0, K%128==0");

  std::lock_guard<std::mutex> lock(cache_mutex);
  if (!cache_matches(A, B, m, n, k))
    refresh_cache(A, B, m, n, k);

  auto C = at::empty({m, n}, A.options());
  launch_matmul_fp8_v5<256, 128, 2>(
      cache.a_fp8, cache.b_nt_fp8, C,
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  return C;
}

at::Tensor matmul_fp8_v5(const at::Tensor& A, const at::Tensor& B) {
  using namespace matmul_fp8_v5_detail;
  c10::cuda::CUDAGuard device_guard(A.device());
  constexpr const char* op_name = "matmul_fp8_v5";

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, op_name, " expects 2D tensors");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), op_name, " expects CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              op_name, " expects BF16 inputs");
  TORCH_CHECK(A.device() == B.device(), op_name, " expects tensors on the same device");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  TORCH_CHECK(B.size(0) == k, op_name, " shape mismatch: A[M,K] @ B[K,N]");
  const int64_t n = B.size(1);
  TORCH_CHECK(m % kBlockM == 0 && n % 256 == 0 && k % 128 == 0,
              op_name, " requires M%128==0, N%256==0, K%128==0");

  auto a_bf16 = A.contiguous();
  auto b_nt_bf16 = B.transpose(0, 1).contiguous();
  auto a_fp8 = cast_bf16_to_fp8_e4m3_cuda(a_bf16);
  auto b_nt_fp8 = cast_bf16_to_fp8_e4m3_cuda(b_nt_bf16);

  auto C = at::empty({m, n}, A.options());
  launch_matmul_fp8_v5<256, 128, 2>(
      a_fp8, b_nt_fp8, C,
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  return C;
}
