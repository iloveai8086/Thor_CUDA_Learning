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
#include "profiler.h"

namespace matmul_fp8_v8_detail {

constexpr int kNumWarps = 6;
constexpr int kThreads = kNumWarps * WARP_SIZE;
constexpr int kBlockM = 128;
constexpr int kMmaK = 32;
constexpr int kTmaK = 128;
constexpr int kPersistentGrid = 20;

void check_cu_v8(CUresult result, const char* expr) {
  if (result == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  TORCH_CHECK(false, expr, " failed: ", name ? name : "<unknown>", " ",
              message ? message : "");
}

#define MATMUL_FP8_V8_CU_CHECK(expr) check_cu_v8((expr), #expr)

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

  MATMUL_FP8_V8_CU_CHECK(cuTensorMapEncodeTiled(
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

__device__ __forceinline__ int min_int(int a, int b) {
  return a < b ? a : b;
}

__device__ __forceinline__ bool is_power_of_two(int x) {
  return x > 0 && (x & (x - 1)) == 0;
}

__device__ __forceinline__ void hilbert_rot(int n, int& x, int& y, int rx, int ry) {
  if (ry == 0) {
    if (rx == 1) {
      x = n - 1 - x;
      y = n - 1 - y;
    }
    const int t = x;
    x = y;
    y = t;
  }
}

__device__ __forceinline__ void hilbert_d2xy(int n, int d, int& x, int& y) {
  int t = d;
  x = 0;
  y = 0;
  for (int s = 1; s < n; s *= 2) {
    const int rx = 1 & (t / 2);
    const int ry = 1 & (t ^ rx);
    hilbert_rot(s, x, y, rx, ry);
    x += s * rx;
    y += s * ry;
    t /= 4;
  }
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

template <
    int BLOCK_N,
    int BLOCK_K,
    int CTA_GROUP,
    int NUM_STAGES,
    int L2_GROUP_SIZE,
    bool DO_PROFILE,
    bool USE_L2_SWIZZLE,
    bool USE_HILBERT_SWIZZLE>
__global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(kThreads)
void matmul_fp8_v8_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    nv_bfloat16* C,
    int M,
    int N,
    int K,
    int64_t* profiler_ptr,
    int num_entries) {
  static_assert(BLOCK_K % kTmaK == 0);
  static_assert(BLOCK_N % CTA_GROUP == 0);
  static_assert(CTA_GROUP == 2);
  static_assert(NUM_STAGES >= 1);
  static_assert(BLOCK_N * 2 <= 512);

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  int cta_rank = 0;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  const int grid_m = M / kBlockM;
  const int grid_n = N / BLOCK_N;

  Profiler profiler;
  if constexpr (DO_PROFILE) {
    if (elect_sync()) {
      profiler.init(num_entries, profiler_ptr, bid * kNumWarps + warp_id);
      profiler.start(ProfilerTag::Setup,
                     make_profiler_meta(warp_id, cta_rank, -1, -1, bid));
    }
  }

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = kBlockM * BLOCK_K * sizeof(__nv_fp8_e4m3);
  constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(__nv_fp8_e4m3);
  constexpr int Stage_size = A_size + B_size;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[NUM_STAGES * 2 + 4];
  __shared__ int tmem_addr[1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

  if (warp_id == 0 && elect_sync()) {
    #pragma unroll
    for (int i = 0; i < NUM_STAGES; ++i) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      mbarrier_init(mainloop_mbar_addr + i * 8, 1);
      mbarrier_init(epilogue_mbar_addr + i * 8, 4 * CTA_GROUP);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    tcgen05_alloc<CTA_GROUP>(addr, BLOCK_N * 2);
  }

  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");
  const int taddr = tmem_addr[0];
  if constexpr (DO_PROFILE) {
    if (elect_sync())
      profiler.stop();
  }

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

  auto compute_bid = [&](int linear_bid, int& bid_m, int& bid_n) {
    constexpr int GROUP_M = CTA_GROUP;
    if constexpr (USE_HILBERT_SWIZZLE) {
      const int cluster_grid_m = grid_m / GROUP_M;
      const int cluster_grid_n = grid_n;
      const int cluster_bid = linear_bid / GROUP_M;
      const int cta_rank_in_cluster_tile = linear_bid % GROUP_M;

      if (cluster_grid_m == cluster_grid_n && is_power_of_two(cluster_grid_m)) {
        int cluster_m = 0;
        int n_tile = 0;
        hilbert_d2xy(cluster_grid_m, cluster_bid, cluster_m, n_tile);
        bid_m = cluster_m * GROUP_M + cta_rank_in_cluster_tile;
        bid_n = n_tile;
      } else {
        constexpr int GROUP_SIZE = L2_GROUP_SIZE;
        const int num_blocks_per_group = grid_m * GROUP_SIZE;
        const int group_idx = linear_bid / num_blocks_per_group;
        const int first_n = group_idx * GROUP_SIZE;
        const int in_group = linear_bid % num_blocks_per_group;
        const int num_in_group = min_int(GROUP_SIZE, grid_n - first_n);
        const int m_group_width = num_in_group * GROUP_M;
        const int m_group = in_group / m_group_width;
        const int rem = in_group % m_group_width;
        bid_m = m_group * GROUP_M + (rem % GROUP_M);
        bid_n = first_n + (rem / GROUP_M);
      }
    } else if constexpr (USE_L2_SWIZZLE) {
      constexpr int GROUP_SIZE = L2_GROUP_SIZE;
      const int num_blocks_per_group = grid_m * GROUP_SIZE;
      const int group_idx = linear_bid / num_blocks_per_group;
      const int first_n = group_idx * GROUP_SIZE;
      const int in_group = linear_bid % num_blocks_per_group;
      const int num_in_group = min_int(GROUP_SIZE, grid_n - first_n);
      const int m_group_width = num_in_group * GROUP_M;
      const int m_group = in_group / m_group_width;
      const int rem = in_group % m_group_width;
      bid_m = m_group * GROUP_M + (rem % GROUP_M);
      bid_n = first_n + (rem / GROUP_M);
    } else {
      bid_m = linear_bid / (grid_n * GROUP_M) * GROUP_M + (linear_bid % GROUP_M);
      bid_n = (linear_bid / GROUP_M) % grid_n;
    }
  };

  auto load = [&](int tma_stage, int mma_phase, int iter_k, int bid_tile, int bid_m, int bid_n) {
    if constexpr (DO_PROFILE) {
      profiler.start(ProfilerTag::WaitMMA,
                     make_profiler_meta(warp_id, cta_rank, tma_stage, mma_phase,
                                        bid_tile, bid_m, bid_n, iter_k));
    }
    mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);
    if constexpr (DO_PROFILE)
      profiler.stop();

    if constexpr (DO_PROFILE) {
      profiler.start(ProfilerTag::IssueTMA,
                     make_profiler_meta(warp_id, cta_rank, tma_stage, mma_phase,
                                        bid_tile, bid_m, bid_n, iter_k));
    }
    const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;
    const int A_smem = smem + tma_stage * Stage_size;
    const int B_smem = A_smem + A_size;
    const int off_m = bid_m * kBlockM;
    const int off_n = bid_n * BLOCK_N + cta_rank * (BLOCK_N / CTA_GROUP);
    const int off_k = iter_k * BLOCK_K;
    tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, off_k / kTmaK, mbar_addr);
    tma_3d_gmem2smem<CTA_GROUP>(B_smem, &B_tmap, 0, off_n, off_k / kTmaK, mbar_addr);
    mbarrier_arrive_expect_tx(mbar_addr, Stage_size);
    if constexpr (DO_PROFILE)
      profiler.stop();
  };

  auto compute = [&](int tma_stage,
                     int tma_phase,
                     int mainloop_stage,
                     int enable_input_d,
                     int bid_tile,
                     int bid_m,
                     int bid_n) {
    if constexpr (DO_PROFILE) {
      profiler.start(ProfilerTag::WaitTMA,
                     make_profiler_meta(warp_id, cta_rank, tma_stage, tma_phase,
                                        bid_tile, bid_m, bid_n, enable_input_d));
    }
    mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if constexpr (DO_PROFILE)
      profiler.stop();

    if constexpr (DO_PROFILE) {
      profiler.start(ProfilerTag::IssueMMA,
                     make_profiler_meta(warp_id, cta_rank, tma_stage, mainloop_stage,
                                        bid_tile, bid_m, bid_n, enable_input_d));
    }
    const int A_smem = smem + tma_stage * Stage_size;
    const int B_smem = A_smem + A_size;
    const int tmem = taddr + mainloop_stage * BLOCK_N;

    int mma_idx = 0;
    #pragma unroll
    for (int k1 = 0; k1 < BLOCK_K / kTmaK; ++k1) {
      #pragma unroll
      for (int k2 = 0; k2 < kTmaK / kMmaK; ++k2) {
        const uint64_t a_desc =
            make_fp8_128b_desc(A_smem + k1 * kBlockM * kTmaK + k2 * kMmaK);
        const uint64_t b_desc =
            make_fp8_128b_desc(B_smem + k1 * (BLOCK_N / CTA_GROUP) * kTmaK + k2 * kMmaK);
        const int accumulate = (enable_input_d || mma_idx != 0);
        tcgen05_mma_fp8_e4m3<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, accumulate);
        ++mma_idx;
      }
    }

    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + tma_stage * 8, cta_mask);
    if constexpr (DO_PROFILE)
      profiler.stop();
  };

  auto epilogue = [&](int mainloop_stage, int bid_tile, int bid_m, int bid_n) {
    if constexpr (DO_PROFILE) {
      if (elect_sync()) {
        profiler.start(ProfilerTag::Epilogue,
                       make_profiler_meta(warp_id, cta_rank, mainloop_stage, -1,
                                          bid_tile, bid_m, bid_n));
      }
    }

    const int epilogue_warp_id = warp_id % 4;
    const int epilogue_tid = epilogue_warp_id * WARP_SIZE + lane_id;
    #pragma unroll
    for (int n = 0; n < BLOCK_N / 8; ++n) {
      float tmp[8];
      const int row = cta_rank * kBlockM + epilogue_warp_id * 32;
      const int col = mainloop_stage * BLOCK_N + n * 8;
      const int addr = taddr + (row << 16) + col;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
          : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
            "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
          : "r"(addr));
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      nv_bfloat162 out[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i)
        out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});

      nv_bfloat16* out_ptr =
          C + (bid_m * kBlockM + epilogue_tid) * N + (bid_n * BLOCK_N + n * 8);
      reinterpret_cast<int4*>(out_ptr)[0] = reinterpret_cast<int4*>(out)[0];
    }

    if constexpr (DO_PROFILE) {
      if (elect_sync())
        profiler.stop();
    }
  };

  const int num_tiles = grid_m * grid_n;
  const int num_iters = K / BLOCK_K;

  if (warp_id == 0 && elect_sync()) {
    int tma_stage = 0;
    int mma_phase = 1;
    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      int bid_m = 0;
      int bid_n = 0;
      compute_bid(this_bid, bid_m, bid_n);
      for (int iter_k = 0; iter_k < num_iters; ++iter_k) {
        load(tma_stage, mma_phase, iter_k, this_bid, bid_m, bid_n);
        tma_stage = (tma_stage + 1) % NUM_STAGES;
        if (tma_stage == 0)
          mma_phase ^= 1;
      }
    }
  } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
    int tma_stage = 0;
    int tma_phase = 0;
    int mainloop_stage = 0;
    int epilogue_phase = 1;

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      int bid_m = 0;
      int bid_n = 0;
      compute_bid(this_bid, bid_m, bid_n);

      if constexpr (DO_PROFILE) {
        profiler.start(ProfilerTag::WaitEpilogue,
                       make_profiler_meta(warp_id, cta_rank, mainloop_stage,
                                          epilogue_phase, this_bid, bid_m, bid_n));
      }
      mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
      if constexpr (DO_PROFILE)
        profiler.stop();

      for (int iter_k = 0; iter_k < num_iters; ++iter_k) {
        compute(tma_stage, tma_phase, mainloop_stage, iter_k != 0,
                this_bid, bid_m, bid_n);
        tma_stage = (tma_stage + 1) % NUM_STAGES;
        if (tma_stage == 0)
          tma_phase ^= 1;
      }

      constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
      tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr + mainloop_stage * 8, cta_mask);
      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0)
        epilogue_phase ^= 1;
    }
  } else if (warp_id >= 2) {
    int mainloop_stage = 0;
    int mainloop_phase = 0;

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      int bid_m = 0;
      int bid_n = 0;
      compute_bid(this_bid, bid_m, bid_n);

      if constexpr (DO_PROFILE) {
        if (elect_sync()) {
          profiler.start(ProfilerTag::WaitMainloop,
                         make_profiler_meta(warp_id, cta_rank, mainloop_stage,
                                            mainloop_phase, this_bid, bid_m, bid_n));
        }
      }
      mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");
      if constexpr (DO_PROFILE) {
        if (elect_sync())
          profiler.stop();
      }

      epilogue(mainloop_stage, this_bid, bid_m, bid_n);

      if (elect_sync()) {
        const int mbar_addr = (epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
        mbarrier_arrive(mbar_addr);
      }
      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0)
        mainloop_phase ^= 1;
    }
  }

  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");
  if (warp_id == 0)
    tcgen05_dealloc<CTA_GROUP>(taddr, BLOCK_N * 2);
  if constexpr (DO_PROFILE) {
    if (elect_sync())
      profiler.flush();
  }
}

template <
    int BLOCK_N,
    int BLOCK_K,
    int CTA_GROUP,
    int NUM_STAGES,
    int L2_GROUP_SIZE,
    bool DO_PROFILE,
    bool USE_L2_SWIZZLE,
    bool USE_HILBERT_SWIZZLE>
void launch_matmul_fp8_v8(
    const at::Tensor& a_fp8,
    const at::Tensor& b_nt_fp8,
    const at::Tensor& c,
    int M,
    int N,
    int K,
    int64_t* profiler_ptr,
    int num_entries) {
  auto A_tmap = make_tma_fp8_3d_128b_desc(a_fp8, M, K, kBlockM, BLOCK_K);
  auto B_tmap =
      make_tma_fp8_3d_128b_desc(b_nt_fp8, N, K, BLOCK_N / CTA_GROUP, BLOCK_K);

  constexpr int smem_size =
      NUM_STAGES * (kBlockM + BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(__nv_fp8_e4m3);
  auto kernel = matmul_fp8_v8_kernel<
      BLOCK_N, BLOCK_K, CTA_GROUP, NUM_STAGES, L2_GROUP_SIZE, DO_PROFILE,
      USE_L2_SWIZZLE, USE_HILBERT_SWIZZLE>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  kernel<<<kPersistentGrid, kThreads, smem_size, stream>>>(
      A_tmap,
      B_tmap,
      reinterpret_cast<nv_bfloat16*>(c.data_ptr()),
      M,
      N,
      K,
      profiler_ptr,
      num_entries);
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

}  // namespace matmul_fp8_v8_detail

template <
    int L2_GROUP_SIZE,
    bool USE_L2_SWIZZLE,
    bool USE_HILBERT_SWIZZLE,
    bool DO_PROFILE = false>
static at::Tensor matmul_fp8_v8_impl(
    const at::Tensor& A,
    const at::Tensor& B,
    const char* op_name,
    int64_t* profiler_ptr = nullptr,
    int num_entries = 0) {
  using namespace matmul_fp8_v8_detail;
  c10::cuda::CUDAGuard device_guard(A.device());

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, op_name, " expects 2D tensors");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), op_name, " expects CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              op_name, " expects BF16 inputs");
  TORCH_CHECK(A.device() == B.device(), op_name, " expects tensors on the same device");
  if constexpr (DO_PROFILE)
    TORCH_CHECK(profiler_ptr != nullptr && num_entries > 0, op_name, " expects profiler storage");

  constexpr int kBlockN = 256;
  constexpr int kBlockK = 128;
  constexpr int kCtaGroup = 2;
  constexpr int kNumStages = 7;

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  TORCH_CHECK(B.size(0) == k, op_name, " shape mismatch: A[M,K] @ B[K,N]");
  const int64_t n = B.size(1);
  TORCH_CHECK(m % (kBlockM * kCtaGroup) == 0 && n % kBlockN == 0 && k % kBlockK == 0,
              op_name, " requires M%256==0, N%256==0, K%128==0");

  auto a_bf16 = A.contiguous();
  auto b_nt_bf16 = B.transpose(0, 1).contiguous();
  auto a_fp8 = cast_bf16_to_fp8_e4m3_cuda(a_bf16);
  auto b_nt_fp8 = cast_bf16_to_fp8_e4m3_cuda(b_nt_bf16);

  auto C = at::empty({m, n}, A.options());
  launch_matmul_fp8_v8<
      kBlockN, kBlockK, kCtaGroup, kNumStages, L2_GROUP_SIZE, DO_PROFILE,
      USE_L2_SWIZZLE, USE_HILBERT_SWIZZLE>(
      a_fp8, b_nt_fp8, C,
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      profiler_ptr, num_entries);
  return C;
}

at::Tensor matmul_fp8_v8(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<6, false, false>(A, B, "matmul_fp8_v8");
}

at::Tensor matmul_fp8_v8_plain(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<6, false, false>(A, B, "matmul_fp8_v8_plain");
}

at::Tensor matmul_fp8_v8_g4(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<4, true, false>(A, B, "matmul_fp8_v8_g4");
}

at::Tensor matmul_fp8_v8_g5(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<5, true, false>(A, B, "matmul_fp8_v8_g5");
}

at::Tensor matmul_fp8_v8_g6(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<6, true, false>(A, B, "matmul_fp8_v8_g6");
}

at::Tensor matmul_fp8_v8_g7(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<7, true, false>(A, B, "matmul_fp8_v8_g7");
}

at::Tensor matmul_fp8_v8_g8(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<8, true, false>(A, B, "matmul_fp8_v8_g8");
}

at::Tensor matmul_fp8_v8_g10(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<10, true, false>(A, B, "matmul_fp8_v8_g10");
}

at::Tensor matmul_fp8_v8_g12(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<12, true, false>(A, B, "matmul_fp8_v8_g12");
}

at::Tensor matmul_fp8_v8_hilbert(const at::Tensor& A, const at::Tensor& B) {
  return matmul_fp8_v8_impl<6, false, true>(A, B, "matmul_fp8_v8_hilbert");
}

at::Tensor profile_matmul_fp8_v8_g6(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& profiler,
    int64_t num_entries) {
  TORCH_CHECK(profiler.is_cuda() && profiler.scalar_type() == at::kLong,
              "profile_matmul_fp8_v8_g6 expects CUDA int64 profiler tensor");
  return matmul_fp8_v8_impl<6, true, false, true>(
      A, B, "profile_matmul_fp8_v8_g6", profiler.data_ptr<int64_t>(),
      static_cast<int>(num_entries));
}

at::Tensor profile_matmul_fp8_v8_hilbert(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& profiler,
    int64_t num_entries) {
  TORCH_CHECK(profiler.is_cuda() && profiler.scalar_type() == at::kLong,
              "profile_matmul_fp8_v8_hilbert expects CUDA int64 profiler tensor");
  return matmul_fp8_v8_impl<6, false, true, true>(
      A, B, "profile_matmul_fp8_v8_hilbert", profiler.data_ptr<int64_t>(),
      static_cast<int>(num_entries));
}
