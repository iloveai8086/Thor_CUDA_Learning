#include "common.h"
#include "profiler.h"

#include "DeepGEMM/third-party/cutlass/include/cute/arch/cluster_sm90.hpp"
#include "DeepGEMM/third-party/cutlass/include/cutlass/arch/barrier.h"
#include "DeepGEMM/third-party/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "DeepGEMM/third-party/cutlass/include/cutlass/pipeline/pipeline.hpp"

#include <cuda_bf16.h>
#include <tuple>

constexpr int NUM_WARPS = 6;  // 1 warp for TMA, 1 warp for MMA, and 4 warps for epilogue
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;  // 192

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;
constexpr int NUM_CLC_STAGES_V12 = 2;

template <int CTA_GROUP>
using ClcClusterShapeV12 = cute::Shape<cute::Int<CTA_GROUP>, cute::_1, cute::_1>;

template <int CTA_GROUP>
using ClcPipelineV12 = cutlass::PipelineCLCFetchAsync<NUM_CLC_STAGES_V12, ClcClusterShapeV12<CTA_GROUP>>;

template <int CTA_GROUP>
using ClcSchedulerV12 = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClcClusterShapeV12<CTA_GROUP>, NUM_CLC_STAGES_V12>;

template <int CTA_GROUP>
using ClcResponseV12 = typename ClcSchedulerV12<CTA_GROUP>::CLCResponse;

template <int CTA_GROUP>
struct ClcSharedStorageV12 {
  alignas(16) ClcResponseV12<CTA_GROUP> clc_response[NUM_CLC_STAGES_V12];
  alignas(16) typename ClcPipelineV12<CTA_GROUP>::SharedStorage pipeline_storage;
};

template <int CTA_GROUP>
static __device__ inline void issue_clc_query_v12(
  typename ClcPipelineV12<CTA_GROUP>::PipelineState state,
  uint32_t mbarrier_addr,
  ClcResponseV12<CTA_GROUP> *clc_response_ptr
) {
  uint32_t result_addr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void *>(&clc_response_ptr[state.index()]));
  asm volatile(
    "{\n\t"
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
    "}\n"
    :
    : "r"(result_addr), "r"(mbarrier_addr));
}

__device__ inline bool decode_clc_response_v12(uint32_t result_addr, int &next_cluster_bid, int cluster_dim_x) {
  uint32_t first_ctaid_x = 0;
  uint32_t valid = 0;
  asm volatile(
    "{\n"
    ".reg .pred p1;\n\t"
    ".reg .b128 clc_result;\n\t"
    "ld.shared.b128 clc_result, [%3];\n\t"
    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"
    "selp.u32 %1, 1, 0, p1;\n\t"
    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, _, _, _}, clc_result;\n\t"
    "}\n"
    : "=r"(first_ctaid_x), "=r"(valid)
    : "r"(0), "r"(result_addr)
    : "memory");

  cutlass::arch::fence_view_async_shared();
  next_cluster_bid = static_cast<int>(first_ctaid_x) / cluster_dim_x;
  return valid == 1;
}

__device__ __forceinline__ bool is_power_of_two(int x) {
  return x > 0 && (x & (x - 1)) == 0;
}

__device__ __forceinline__ void hilbert_rot(int n, int &x, int &y, int rx, int ry) {
  if (ry == 0) {
    if (rx == 1) {
      x = n - 1 - x;
      y = n - 1 - y;
    }
    int t = x;
    x = y;
    y = t;
  }
}

__device__ __forceinline__ void hilbert_d2xy(int n, int d, int &x, int &y) {
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

template <
  int BLOCK_N,
  int BLOCK_K,
  int CTA_GROUP,
  int NUM_STAGES,
  int L2_GROUP_SIZE,
  bool DO_PROFILE,
  bool EPILOGUE_CACHE_MOD,
  bool USE_L2_SWIZZLE,
  bool USE_2D_SWIZZLE,
  bool USE_HILBERT_SWIZZLE,
  bool USE_CLC
>
__global__
__cluster_dims__(CTA_GROUP, 1, 1)
__launch_bounds__(TB_SIZE)
void matmul_v12_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M / BLOCK_M;  // 4096 / 128 = 32
  const int grid_n = N / BLOCK_N;  // 4096 / 256 = 16, num_tiles = 32 * 16 = 512
  // CTA rank in a cluster
  int cta_rank;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  Profiler profiler;
  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.init(num_entries, profiler_ptr, bid * NUM_WARPS + warp_id);
    profiler.start(ProfilerTag::Setup, make_profiler_meta(warp_id, cta_rank, -1, -1, bid));
  }

  // set up smem
  // each CTA only loads half of B
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);

  // set up mbarrier and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  2 mbars for mainloop
  //                  2 mbars for epilogue
  // TMA warp -> TMA mbar -> MMA warp -> mainloop mbar -> epilogue warps
  //          <- MMA mbar <-          <- epilogue mbar
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[NUM_STAGES * 2 + 4];
  __shared__ int tmem_addr[1];  // tmem address is 32-bit
  __shared__ __align__(16) char clc_shared_bytes[sizeof(ClcSharedStorageV12<CTA_GROUP>)];
  auto &clc_shared = *reinterpret_cast<ClcSharedStorageV12<CTA_GROUP> *>(clc_shared_bytes);
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

  if (warp_id == 0 && elect_sync()) {
    // tma_mbar[stage] = TMA full
    // mma_mbar[stage] = MMA done
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);  // both CTAs report TMA to CTA0 only
      mbarrier_init(mma_mbar_addr + i * 8, 1);          // CTA0 reports MMA to BOTH CTAs (multicast)  CTA0
    }
    // mainloop_mbar[0/1] = MMA/mainloop full
    // epilogue_mbar[0/1] = epilogue done
    for (int i = 0; i < 2; i++) {
      mbarrier_init(mainloop_mbar_addr + i * 8, 1);              // CTA0 reports mainloop to BOTH CTAs (multicast)  CTA0
      mbarrier_init(epilogue_mbar_addr + i * 8, 4 * CTA_GROUP);  // 4 epilogue warps x both CTAs report to CTA0 only
    }
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem for output (issued by both CTAs)
    // we allocate double BLOCK_N to double-buffer accumulator
    // it's unlikely that we use BLOCK_N > 256 (tmem limit is 512 columns)
    static_assert(BLOCK_N * 2 <= 512);
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    tcgen05_alloc<CTA_GROUP>(addr, BLOCK_N * 2);
  }

  if constexpr (CTA_GROUP > 1) {
    // visible to all threads in a cluster
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  }
  else {
    // visible to all threads in a threadblock
    __syncthreads();
  }
  const int taddr = tmem_addr[0];  // this will be 0
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
  constexpr int MMA_M = BLOCK_M * CTA_GROUP;  // 128 for 1SM, 256 for 2SM
  constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                            | (1U << 7U)   // atype=BF16
                            | (1U << 10U)  // btype=BF16
                            | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                            | ((uint32_t)MMA_M >> 4U << 24U)  // MMA_M
                            ;
  
  auto load = [&](int tma_stage, int mma_phase, int iter_k, int bid_tile, int bid_m, int bid_n) {
    if constexpr (DO_PROFILE) profiler.start(
      ProfilerTag::WaitMMA,
      make_profiler_meta(warp_id, cta_rank, tma_stage, mma_phase, bid_tile, bid_m, bid_n, iter_k)
    );
    mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);
    if constexpr (DO_PROFILE) profiler.stop();

    if constexpr (DO_PROFILE) profiler.start(
      ProfilerTag::IssueTMA,
      make_profiler_meta(warp_id, cta_rank, tma_stage, mma_phase, bid_tile, bid_m, bid_n, iter_k)
    );
    // both CTA ranks update tx-count of CTA0's mbar
    // https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/copy_sm100_tma.hpp#L113-L115
    const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;  // this is on CTA0
    const int A_smem = smem + tma_stage * (A_size + B_size);
    const int B_smem = A_smem + A_size;

    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N + cta_rank * (BLOCK_N / CTA_GROUP);
    const int off_k = iter_k * BLOCK_K;
    tma_3d_g2s<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
    tma_3d_g2s<CTA_GROUP>(B_smem, &B_tmap, 0, off_n, off_k / 64, mbar_addr);

    // NOTE: we are using .shared::cluster here
    // signal TMA full：
    mbarrier_arrive_expect_tx(mbar_addr, A_size + B_size);
    if constexpr (DO_PROFILE) profiler.stop();
  };

  auto compute = [&](int tma_stage, int tma_phase, int mainloop_stage, 
                    int enable_input_d, int bid_tile, int bid_m, int bid_n) {
    // wait for TMA on the local TMA mbar
    if constexpr (DO_PROFILE) profiler.start(
      ProfilerTag::WaitTMA,
      make_profiler_meta(warp_id, cta_rank, tma_stage, tma_phase, bid_tile, bid_m, bid_n, enable_input_d)
    );
    mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");  // (why) do we need this? from DeepGEMM
    if constexpr (DO_PROFILE) profiler.stop();

    if constexpr (DO_PROFILE) profiler.start(
      ProfilerTag::IssueMMA,
      make_profiler_meta(warp_id, cta_rank, tma_stage, mainloop_stage, bid_tile, bid_m, bid_n, enable_input_d)
    );
    // select TMA buffer
    const int A_smem = smem + tma_stage * (A_size + B_size);
    const int B_smem = A_smem + A_size;

    // set up shared memory descriptors for A and B
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
    // 128-byte swizzling. LBO is implied to be 1.
    auto make_desc = [](int addr) -> uint64_t {
      const int SBO = 8 * 128;
      return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };

    // select tmem buffer,V5
    const int tmem = taddr + mainloop_stage * BLOCK_N;

    // we specify the LOCAL A smem and B smem here. the tensor cores hardware will know
    // to fetch data from BOTH CTAs, assuming we use the same offset across CTAs.
    // manually unroll 1st iteration to disable accumulation
    {
      tcgen05_mma_f16<CTA_GROUP>(tmem, make_desc(A_smem), make_desc(B_smem), i_desc, enable_input_d);
      for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k2 * 32);
        tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
      }
    }
    // k1 selects the (BLOCK_M, 64) tile.
    // k2 selects the (BLOCK_M, 16) tile, whose rows are swizzled.
    for (int k1 = 1; k1 < BLOCK_K / 64; k1++)
      for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k1 * (BLOCK_N / CTA_GROUP) * 128 + k2 * 32);
        tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
      }
    // this signals to mbar on BOTH CTAs (thanks to .multicast::cluster)
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    // signal MMA done
    tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + tma_stage * 8, cta_mask);
    if constexpr (DO_PROFILE) profiler.stop();
  };

  auto epilogue = [&](int mainloop_stage, int bid_tile, int bid_m, int bid_n) {
    if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(
      ProfilerTag::Epilogue,
      make_profiler_meta(warp_id, cta_rank, mainloop_stage, -1, bid_tile, bid_m, bid_n)
    );
    // we are using warp2-warp5 for epilogue. hence, we need to remap the warp_id
    // for accessing tmem.
    const int epilogue_warp_id = warp_id % 4;
    const int epilogue_tid = epilogue_warp_id * WARP_SIZE + lane_id;

    // V12 keeps the same row-strided store pattern as V6/V7, but halves the
    // epilogue loop count by loading/storing 16 columns per thread at a time.
    constexpr int WIDTH = 16;
    for (int n = 0; n < BLOCK_N / WIDTH; n++) {
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a
      // Layout A
      // select tmem buffer
      const int row = cta_rank * 128 + epilogue_warp_id * 32;
      const int col = mainloop_stage * BLOCK_N + n * WIDTH;
      const int addr = taddr + (row << 16) + col;

      // uncoalesced writes weeee
      nv_bfloat16 *out_ptr = C_ptr + (bid_m * BLOCK_M + epilogue_tid) * N + (bid_n * BLOCK_N + n * WIDTH);
      if constexpr (EPILOGUE_CACHE_MOD)
        asm volatile(
          "{\n"
          ".reg .f32 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;\n"
          ".reg .b32 b0, b1, b2, b3, b4, b5, b6, b7;\n"
          "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
          "  {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15}, [%1];\n"
          "tcgen05.wait::ld.sync.aligned;\n"
          "cvt.rn.bf16x2.f32 b0, f1, f0;\n"
          "cvt.rn.bf16x2.f32 b1, f3, f2;\n"
          "cvt.rn.bf16x2.f32 b2, f5, f4;\n"
          "cvt.rn.bf16x2.f32 b3, f7, f6;\n"
          "cvt.rn.bf16x2.f32 b4, f9, f8;\n"
          "cvt.rn.bf16x2.f32 b5, f11, f10;\n"
          "cvt.rn.bf16x2.f32 b6, f13, f12;\n"
          "cvt.rn.bf16x2.f32 b7, f15, f14;\n"
          "st.relaxed.cta.global.L1::no_allocate.v8.b32 [%0], {b0, b1, b2, b3, b4, b5, b6, b7};\n"
          "}"
          :: "l"(out_ptr), "r"(addr)
        );
      else
        asm volatile(
          "{\n"
          ".reg .f32 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;\n"
          ".reg .b32 b0, b1, b2, b3, b4, b5, b6, b7;\n"
          "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
          "  {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15}, [%1];\n"
          "tcgen05.wait::ld.sync.aligned;\n"
          "cvt.rn.bf16x2.f32 b0, f1, f0;\n"
          "cvt.rn.bf16x2.f32 b1, f3, f2;\n"
          "cvt.rn.bf16x2.f32 b2, f5, f4;\n"
          "cvt.rn.bf16x2.f32 b3, f7, f6;\n"
          "cvt.rn.bf16x2.f32 b4, f9, f8;\n"
          "cvt.rn.bf16x2.f32 b5, f11, f10;\n"
          "cvt.rn.bf16x2.f32 b6, f13, f12;\n"
          "cvt.rn.bf16x2.f32 b7, f15, f14;\n"
          "st.global.v8.b32 [%0], {b0, b1, b2, b3, b4, b5, b6, b7};\n"
          "}"
          :: "l"(out_ptr), "r"(addr)
        );
    }
    if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();
  };

  const int num_tiles = grid_m * grid_n;
  const int num_iters = K / BLOCK_K;

  auto compute_bid = [&](int bid) -> std::tuple<int, int> {
    int bid_m, bid_n;
    if constexpr (USE_HILBERT_SWIZZLE) {
      // Hilbert scheduling in cluster-tile space. CTA_GROUP=2 means one logical
      // cluster tile expands to two CTA tiles with the same N and adjacent M.
      constexpr int GROUP_M = 2;
      const int cluster_grid_m = grid_m / GROUP_M;
      const int cluster_grid_n = grid_n;
      const int cluster_bid = bid / GROUP_M;
      const int cta_rank_in_cluster_tile = bid % GROUP_M;

      if (cluster_grid_m == cluster_grid_n && is_power_of_two(cluster_grid_m)) {
        int cluster_m, n_tile;
        hilbert_d2xy(cluster_grid_m, cluster_bid, cluster_m, n_tile);
        bid_m = cluster_m * GROUP_M + cta_rank_in_cluster_tile;
        bid_n = n_tile;
      }
      else {
        constexpr int GROUP_SIZE = L2_GROUP_SIZE;
        const int num_blocks_per_group = grid_m * GROUP_SIZE;
        const int group_idx = bid / num_blocks_per_group;
        const int first_n = group_idx * GROUP_SIZE;
        const int in_group = bid % num_blocks_per_group;
        const int num_in_group = min(GROUP_SIZE, grid_n - first_n);
        const int m_group_width = num_in_group * GROUP_M;
        const int m_group = in_group / m_group_width;
        const int rem = in_group % m_group_width;
        bid_m = m_group * GROUP_M + (rem % GROUP_M);
        bid_n = first_n + (rem / GROUP_M);
      }
    }
    else if constexpr (USE_2D_SWIZZLE) {
      // 2D rectangular swizzle in cluster-tile space.
      // CTA_GROUP=2 means the true scheduling unit is one cluster tile:
      //   cluster_m = bid_m / 2, bid_n = bid_n.
      // A group of 2 cluster rows x 5 N tiles corresponds to 4x5 CTA tiles,
      // i.e. 10 clusters / 20 CTAs, matching the current 20-SM launch wave.
      constexpr int GROUP_M = 2;
      constexpr int GROUP_CLUSTER_M = 2;
      constexpr int GROUP_N = 5;
      const int cluster_grid_m = grid_m / GROUP_M;
      const int cluster_bid = bid / GROUP_M;
      const int cta_rank_in_cluster_tile = bid % GROUP_M;

      int remaining = cluster_bid;
      int first_cluster_m = 0;
      int first_n = 0;
      int local = 0;
      int group_cluster_m = GROUP_CLUSTER_M;
      int group_n = GROUP_N;

      const int num_groups_n = (grid_n + GROUP_N - 1) / GROUP_N;
      const int num_groups_m = (cluster_grid_m + GROUP_CLUSTER_M - 1) / GROUP_CLUSTER_M;
      for (int group_n_idx = 0; group_n_idx < num_groups_n; group_n_idx++) {
        const int candidate_first_n = group_n_idx * GROUP_N;
        const int candidate_group_n = min(GROUP_N, grid_n - candidate_first_n);
        const int n_band_tiles = cluster_grid_m * candidate_group_n;
        if (remaining >= n_band_tiles) {
          remaining -= n_band_tiles;
          continue;
        }

        first_n = candidate_first_n;
        group_n = candidate_group_n;
        for (int group_m_idx = 0; group_m_idx < num_groups_m; group_m_idx++) {
          const int candidate_first_cluster_m = group_m_idx * GROUP_CLUSTER_M;
          const int candidate_group_cluster_m = min(GROUP_CLUSTER_M, cluster_grid_m - candidate_first_cluster_m);
          const int group_tiles = candidate_group_cluster_m * group_n;
          if (remaining >= group_tiles) {
            remaining -= group_tiles;
            continue;
          }
          first_cluster_m = candidate_first_cluster_m;
          group_cluster_m = candidate_group_cluster_m;
          local = remaining;
          break;
        }
        break;
      }

      const int local_cluster_m = local / group_n;
      const int local_n = local % group_n;
      bid_m = (first_cluster_m + local_cluster_m) * GROUP_M + cta_rank_in_cluster_tile;
      bid_n = first_n + local_n;
    }
    else if constexpr (USE_L2_SWIZZLE) {
      // Group tiles by nearby N coordinates so persistent CTAs reuse B tiles in L2,
      // while preserving v6's original 2-row M ordering inside each group.
      constexpr int GROUP_M = 2;
      // constexpr int GROUP_M = 2
      // bid 0 -> m=0, n=0  bid 1 -> m=1, n=0
      // N group 0: n = 0..5
      // N group 1: n = 6..11
      // N group 2: n = 12..15
      constexpr int GROUP_SIZE = L2_GROUP_SIZE;
      const int num_blocks_per_group = grid_m * GROUP_SIZE;  // 32 * 12 = 384
      // const int num_blocks_per_group = grid_m * GROUP_SIZE
      const int group_idx = bid / num_blocks_per_group;
      // const int group_idx = bid / num_blocks_per_group
      // bid=0    -> group_idx=0
      // bid=191  -> group_idx=0
      // bid=192  -> group_idx=1
      // bid=383  -> group_idx=1
      // bid=384  -> group_idx=2
      const int first_n = group_idx * GROUP_SIZE;
      // const int first_n = group_idx * GROUP_SIZE
      // group_idx=0 -> first_n=0
      // group_idx=1 -> first_n=6
      // group_idx=2 -> first_n=12
      const int in_group = bid % num_blocks_per_group;
      // bid=193
      // num_blocks_per_group=192
      // group_idx=1
      // first_n=6
      // in_group=193 % 192 = 1
      const int num_in_group = min(GROUP_SIZE, grid_n - first_n);
      // group 0: first_n=0,  num_in_group=min(6,16)=6      -> n=0..5
      // group 1: first_n=6,  num_in_group=min(6,10)=6      -> n=6..11
      // group 2: first_n=12, num_in_group=min(6,4)=4       -> n=12..15
      const int m_group_width = num_in_group * GROUP_M;
      // m=0,n=0
      // m=1,n=0
      // m=0,n=1
      // m=1,n=1
      // ...
      // in_group 0..11  -> m=0/1, n=0..5
      // in_group 12..23 -> m=2/3, n=0..5
      // in_group 24..35 -> m=4/5, n=0..5
      const int m_group = in_group / m_group_width;
      // in_group 0..11   -> m_group=0 -> m=0/1
      // in_group 12..23  -> m_group=1 -> m=2/3
      // in_group 24..35  -> m_group=2 -> m=4/5
      const int rem = in_group % m_group_width;
      // in_group=14
      // m_group_width=12
      // m_group = 14 / 12 = 1
      // rem     = 14 % 12 = 2
      bid_m = m_group * GROUP_M + (rem % GROUP_M);
      // m_group=0 -> base m=0
      // m_group=1 -> base m=2
      // m_group=2 -> base m=4
      bid_n = first_n + (rem / GROUP_M);
      // rem=0,1   -> rem / 2 = 0 -> n=first_n + 0
      // rem=2,3   -> rem / 2 = 1 -> n=first_n + 1
      // rem=4,5   -> rem / 2 = 2 -> n=first_n + 2
      // bid 0 -> m=0,n=0
      // bid 1 -> m=1,n=0
      // bid 2 -> m=0,n=1
      // bid 3 -> m=1,n=1
      // bid 0  -> group=0, in_group=0   -> m=0, n=0
      // bid 1  -> group=0, in_group=1   -> m=1, n=0
      // bid 2  -> group=0, in_group=2   -> m=0, n=1
      // bid 3  -> group=0, in_group=3   -> m=1, n=1
      // ...
      // bid 10 -> group=0, in_group=10  -> m=0, n=5
      // bid 11 -> group=0, in_group=11  -> m=1, n=5

      // bid 12 -> group=0, in_group=12  -> m=2, n=0
      // bid 13 -> group=0, in_group=13  -> m=3, n=0
      // bid 14 -> group=0, in_group=14  -> m=2, n=1
      // bid 15 -> group=0, in_group=15  -> m=3, n=1
      // C[m=0,n=3]
      // C[m=1,n=3]
      // C[m=2,n=3]
    }
    else {
      // bid must run along M-mode first so that .cta_group::2 works correctly
      constexpr int GROUP_M = 2;
      bid_m = bid / (grid_n * GROUP_M) * GROUP_M + (bid % GROUP_M);
      bid_n = (bid / GROUP_M) % grid_n;
    }
    return {bid_m, bid_n};
  };
  // bid 0 -> bid_m=0, bid_n=0
  // bid 1 -> bid_m=1, bid_n=0
  // bid 2 -> bid_m=0, bid_n=1
  // bid 3 -> bid_m=1, bid_n=1
  // this_bid = bid, bid + num_bids, bid + 2 * num_bids, ...
  
  if constexpr (USE_CLC) {
    using ThisClcPipeline = ClcPipelineV12<CTA_GROUP>;
    using ThisClcPipelineState = typename ThisClcPipeline::PipelineState;
    const bool is_producer_warp = (cta_rank == 0 && warp_id == 0);
    const bool is_clc_consumer = !(cta_rank != 0 && warp_id == 1);

    typename ThisClcPipeline::Params clc_params;
    clc_params.transaction_bytes = 16;
    clc_params.role = is_producer_warp
      ? ThisClcPipeline::ThreadCategory::ProducerConsumer
      : (is_clc_consumer ? ThisClcPipeline::ThreadCategory::Consumer : ThisClcPipeline::ThreadCategory::NonParticipant);
    clc_params.producer_blockid = 0;
    clc_params.producer_arv_count = 1;
    clc_params.consumer_arv_count = WARP_SIZE * (CTA_GROUP * NUM_WARPS - (CTA_GROUP > 1 ? (CTA_GROUP - 1) : 0));
    clc_params.initializing_warp = 0;
    ThisClcPipeline clc_pipeline(clc_shared.pipeline_storage, clc_params, ClcClusterShapeV12<CTA_GROUP>{});
    if constexpr (CTA_GROUP > 1) {
      cute::cluster_sync();
    } else {
      __syncthreads();
    }

    const int num_cluster_tiles = num_tiles / CTA_GROUP;
    const int launched_clusters = static_cast<int>(cute::cluster_grid_dims().x);
    int current_cluster_bid = static_cast<int>(cute::cluster_id_in_grid().x);
    bool has_work = current_cluster_bid < num_cluster_tiles;

    auto fetch_next_work = [&](ThisClcPipelineState &consumer_state, int current_bid) {
      int fallback_cluster_bid = current_bid + launched_clusters;
      int fallback_valid = static_cast<int>(fallback_cluster_bid < num_cluster_tiles);

      auto token = clc_pipeline.consumer_try_wait(consumer_state);
      clc_pipeline.consumer_wait(consumer_state, token);
      int next_cluster_bid = 0;
      int valid = 0;
      if (lane_id == 0) {
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&clc_shared.clc_response[consumer_state.index()]);
        valid = static_cast<int>(decode_clc_response_v12(smem_addr, next_cluster_bid, CTA_GROUP));
      }
      clc_pipeline.consumer_release(consumer_state);
      ++consumer_state;

      next_cluster_bid = __shfl_sync(0xFFFF'FFFF, next_cluster_bid, 0);
      valid = __shfl_sync(0xFFFF'FFFF, valid, 0);
      if (valid == 0) {
        next_cluster_bid = fallback_cluster_bid;
        valid = fallback_valid;
      }
      return std::make_pair(next_cluster_bid, valid != 0);
    };

    if (warp_id == 0) {
      int tma_stage = 0;
      int mma_phase = 1;
      ThisClcPipelineState consumer_state;
      auto producer_state = cutlass::make_producer_start_state<ThisClcPipeline>();

      while (has_work) {
        int effective_bid = current_cluster_bid * CTA_GROUP + cta_rank;
        auto [bid_m, bid_n] = compute_bid(effective_bid);

        if (is_producer_warp) {
          clc_pipeline.producer_acquire(producer_state);
          if (elect_sync()) {
            issue_clc_query_v12<CTA_GROUP>(producer_state, clc_pipeline.producer_get_barrier(producer_state), clc_shared.clc_response);
          }
          ++producer_state;
        }

        if (elect_sync()) {
          for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            load(tma_stage, mma_phase, iter_k, effective_bid, bid_m, bid_n);
            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0)
              mma_phase ^= 1;
          }
        }

        auto [next_cluster_bid, next_has_work] = fetch_next_work(consumer_state, current_cluster_bid);
        has_work = next_has_work;
        current_cluster_bid = next_cluster_bid;
      }

      if (is_producer_warp) {
        clc_pipeline.producer_tail(producer_state);
      }
    }
    else if (cta_rank == 0 && warp_id == 1) {
      int tma_stage = 0;
      int tma_phase = 0;
      int mainloop_stage = 0;
      int epilogue_phase = 1;
      int local_cluster_bid = current_cluster_bid;
      bool local_has_work = has_work;
      ThisClcPipelineState consumer_state;

      while (local_has_work) {
        int effective_bid = local_cluster_bid * CTA_GROUP;
        auto [bid_m, bid_n] = compute_bid(effective_bid);

        if (elect_sync()) {
          if constexpr (DO_PROFILE) profiler.start(
            ProfilerTag::WaitEpilogue,
            make_profiler_meta(warp_id, cta_rank, mainloop_stage, epilogue_phase, effective_bid, bid_m, bid_n)
          );
          mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
          if constexpr (DO_PROFILE) profiler.stop();

          for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            compute(tma_stage, tma_phase, mainloop_stage, iter_k, effective_bid, bid_m, bid_n);
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

        auto [next_bid, next_valid] = fetch_next_work(consumer_state, local_cluster_bid);
        local_has_work = next_valid;
        local_cluster_bid = next_bid;
      }
    }
    else if (warp_id >= 2) {
      int mainloop_stage = 0;
      int mainloop_phase = 0;
      int local_cluster_bid = current_cluster_bid;
      bool local_has_work = has_work;
      ThisClcPipelineState consumer_state;

      while (local_has_work) {
        int effective_bid = local_cluster_bid * CTA_GROUP + cta_rank;
        auto [bid_m, bid_n] = compute_bid(effective_bid);

        if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(
          ProfilerTag::WaitMainloop,
          make_profiler_meta(warp_id, cta_rank, mainloop_stage, mainloop_phase, effective_bid, bid_m, bid_n)
        );
        mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

        epilogue(mainloop_stage, effective_bid, bid_m, bid_n);

        if (elect_sync()) {
          const int mbar_addr = (epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
          mbarrier_arrive(mbar_addr);
        }

        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0)
          mainloop_phase ^= 1;

        auto [next_bid, next_valid] = fetch_next_work(consumer_state, local_cluster_bid);
        local_has_work = next_valid;
        local_cluster_bid = next_bid;
      }
    }
  }
  else {
    if (warp_id == 0 && elect_sync()) {
      // TMA warp
      int tma_stage = 0;
      int mma_phase = 1;  // the initial MMA phase is 0, and it is available. so we initialize it with 1.
      // num_bids = 20,num_tiles = 32 * 16 = 512
      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        auto [bid_m, bid_n] = compute_bid(this_bid);
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          load(tma_stage, mma_phase, iter_k, this_bid, bid_m, bid_n);

          // flip phase when we have cycled through all TMA buffers
          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0)
            mma_phase ^= 1;
        }
      }
    }
    else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
      // MMA warp
      int tma_stage = 0;
      int tma_phase = 0;
      int mainloop_stage = 0;
      int epilogue_phase = 1;  // the initial epilogue phase is 0, and it is available. hence we initialize it with 1.

      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        auto [bid_m, bid_n] = compute_bid(this_bid);

        // wait for epilogue to finish
        if constexpr (DO_PROFILE) profiler.start(
          ProfilerTag::WaitEpilogue,
          make_profiler_meta(warp_id, cta_rank, mainloop_stage, epilogue_phase, this_bid, bid_m, bid_n)
        );
        mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
        if constexpr (DO_PROFILE) profiler.stop();

        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          compute(tma_stage, tma_phase, mainloop_stage, iter_k, this_bid, bid_m, bid_n);

          // flip phase when we have cycled through all TMA buffers
          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0)
            tma_phase ^= 1;
        }

        // signal when tcgen05 finishes with the main loop to BOTH CTAs
        // (notice .multicast::cluster)
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr + mainloop_stage * 8, cta_mask);

        // flip phase when we have cycled through all tmem buffers
        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0)
          epilogue_phase ^= 1;
      }
    }
    else if (warp_id >= 2) {
      int mainloop_stage = 0;
      int mainloop_phase = 0;

      // epilogue warps
      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        auto [bid_m, bid_n] = compute_bid(this_bid);

        // wait for mainloop to finish
        if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(
          ProfilerTag::WaitMainloop,
          make_profiler_meta(warp_id, cta_rank, mainloop_stage, mainloop_phase, this_bid, bid_m, bid_n)
        );
        mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
        // PTX doc says we need to add this before tcgen05.ld, after tcgen05.mma
        asm volatile("tcgen05.fence::after_thread_sync;");
        if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

        epilogue(mainloop_stage, this_bid, bid_m, bid_n);

        // all epilogue warps report to CTA0 mbar
        // signal epilogue done
        if (elect_sync()) {
          const int mbar_addr = (epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
          mbarrier_arrive(mbar_addr);
        }

        // flip phase when we have cycled through all tmem buffers
        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0)
          mainloop_phase ^= 1;
      }
    }
  }

  if constexpr (CTA_GROUP > 1) {
    // this is important. otherwise the kernel may fail.
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  } else {
    __syncthreads();  // all threads finish reading data from tmem
  }

  if (warp_id == 0)  // deallocate tmem (issued by both CTAs)
    tcgen05_dealloc<CTA_GROUP>(taddr, BLOCK_N * 2);
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.flush();
}

template <
  int BLOCK_N,
  int BLOCK_K,
  int CTA_GROUP,
  int NUM_STAGES,
  int L2_GROUP_SIZE,
  bool DO_PROFILE,
  bool EPILOGUE_CACHE_MOD,
  bool USE_L2_SWIZZLE,
  bool USE_2D_SWIZZLE,
  bool USE_HILBERT_SWIZZLE,
  bool USE_CLC
>
void matmul_v12_launch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  // when using threadblock cluster, each threadblock:
  // - still loads (BLOCK_M, BLOCK_K) of A
  // - only loads (BLOCK_N / 2, BLOCK_K) of B.
  // - and still stores (BLOCK_M, BLOCK_N) of C.
  CUtensorMap A_tmap, B_tmap;
  init_tmap_3d_128B(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
  init_tmap_3d_128B(&B_tmap, B_ptr, N, K, BLOCK_N / CTA_GROUP, BLOCK_K);

  const int num_tiles = (M / BLOCK_M) * (N / BLOCK_N);
  // int grid = 148;  // B200
  int grid = USE_CLC ? num_tiles : 20;  // thor T5000 fixed persistent path uses 20 CTAs
  int size_AB = (BLOCK_M + BLOCK_N / CTA_GROUP) * BLOCK_K * NUM_STAGES;
  int smem_size = size_AB * sizeof(nv_bfloat16);

  auto this_kernel = matmul_v12_kernel<BLOCK_N, BLOCK_K, CTA_GROUP, NUM_STAGES, L2_GROUP_SIZE, DO_PROFILE, EPILOGUE_CACHE_MOD, USE_L2_SWIZZLE, USE_2D_SWIZZLE, USE_HILBERT_SWIZZLE, USE_CLC>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  if constexpr (USE_CLC) {
    check_cuda(cudaFuncSetAttribute(this_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = dim3(grid, 1, 1);
    launch_config.blockDim = dim3(TB_SIZE, 1, 1);
    launch_config.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute cluster_attr = {};
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim.x = CTA_GROUP;
    cluster_attr.val.clusterDim.y = 1;
    cluster_attr.val.clusterDim.z = 1;
    launch_config.attrs = &cluster_attr;
    launch_config.numAttrs = 1;

    check_cuda(cudaLaunchKernelEx(&launch_config, this_kernel, A_tmap, B_tmap, C_ptr, M, N, K, profiler_ptr, num_entries));
  }
  else {
    this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K, profiler_ptr, num_entries);
  }
  check_cuda(cudaGetLastError());
}

void matmul_v12(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // V6 persistent kernel + L2 N-band swizzle, with V7-style x16 epilogue and
  // ordinary st.global.v8.b32 stores. GROUP_SIZE=6 was the best recent sweep.
  matmul_v12_launch<256, 64, 2, 7, 6, false, false, true, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v12_nc(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // Same as matmul_v12, but with ordinary st.global.v8.b32 stores.
  matmul_v12_launch<256, 64, 2, 7, 6, false, false, true, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v12_l1noalloc(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // Experimental cache modifier path. It is not the default because it can be
  // slower on Thor for this epilogue.
  matmul_v12_launch<256, 64, 2, 7, 6, false, true, true, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v12_hilbert(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // Hilbert tile order from v6_2_hilbert, with the x16 epilogue.
  matmul_v12_launch<256, 64, 2, 7, 6, false, false, false, false, true, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v12_hilbert_l1noalloc(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // Hilbert tile order plus the L1::no_allocate store variant.
  matmul_v12_launch<256, 64, 2, 7, 6, false, true, false, false, true, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v12_clc_hilbert(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // V11_2-style Cluster Launch Control work fetching, combined with Hilbert
  // tile order and the V12 x16 epilogue.
  matmul_v12_launch<256, 64, 2, 7, 6, false, false, false, false, true, true>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void profile_matmul_v12(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  matmul_v12_launch<256, 64, 2, 7, 6, true, false, true, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries);
}
// python bench_flopsv2.py   --kernel matmul_v6_2_g6,matmul_v6_2_hilbert,matmul_v11_2,matmul_v12,matmul_v12_l1noalloc,matmul_v12_hilbert,matmul_v12_hilbert_l1noalloc,matmul_v12_clc_hilbert   --shape 4096,4096,4096   --warmup 10   --repeat-warmup 1   --iters 100   --repeats 20   --samples   --no-verbose-build --flush-l2-mb 256 --flush-l2-mode repeat
// tcgen05.ld...x8 -> tcgen05.ld...x16
// C++ __float22bfloat162_rn + reinterpret_cast<int4> -> inline PTX cvt.rn.bf16x2.f32 + st.global.v8.b32
// matmul_v12: v6 g6 swizzle + x16 epilogue +
// matmul_v12_hilbert: Hilbert swizzle + x16 epilogue
