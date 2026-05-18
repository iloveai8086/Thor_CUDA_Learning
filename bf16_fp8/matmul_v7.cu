#include "common.h"
#include "profiler.h"

#include <cuda_bf16.h>


// v6:
// warp 0: TMA
// warp 1: MMA
// warp 2-5: epilogue
// v7：
// warp 0-3: epilogue
// warp 4: TMA
// warp 5: MMA

// tcgen05.ld...x8
// float tmp[8]
// int4 store
// tcgen05.ld...x16
// st.global.v8.b32
// v6 epilogue loop: 256 / 8  = 32
// v7 epilogue loop: 256 / 16 = 16


// mbarrier_wait(...)
// if (elect_sync()) mbarrier_arrive(...)
// if (warp_id == 0) mbarrier_wait(...)
// bar.sync 1, 128
// mbarrier_arrive(...)
// v6: 4 * CTA_GROUP
// v7: 4 * CTA_GROUP * WARP_SIZE

// __shared__ uint64_t mbars[NUM_STAGES * 2 + 4];
// __shared__ int tmem_addr[1];



// BLOCK_M = 128
// BLOCK_N = 256
// grid_m = 4096 / 128 = 32
// grid_n = 4096 / 256 = 16
// num_tiles = 32 * 16 = 512
// v7a: grid = 20
// v7b/v7c: grid = 128
// v7b: st.global.v8.b32
// v7c: st.relaxed.cta.global.L1::no_allocate.v8.b32

constexpr int NUM_WARPS = 6;  // 1 warp for TMA, 1 warp for MMA, and 4 warps for epilogue
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 64;  // 128-byte
constexpr int MMA_K = 16;
constexpr int TMA_STORE_EPI_M = 128;
constexpr int TMA_STORE_EPI_N = 64;
constexpr int TMA_STORE_C_STAGES = 2;

__device__ inline void prefetch_tma_descriptor_v7(const void *tmap_ptr) {
  uint64_t gmem_desc = reinterpret_cast<uint64_t>(tmap_ptr);
  asm volatile("prefetch.tensormap [%0];" :: "l"(gmem_desc) : "memory");
}

__device__ inline void tma_2d_smem2gmem_v7(const void *tmap_ptr, int smem_addr, int x, int y) {
  uint64_t gmem_desc = reinterpret_cast<uint64_t>(tmap_ptr);
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
    :: "l"(gmem_desc), "r"(smem_addr), "r"(x), "r"(y)
    : "memory");
}

__device__ inline void tma_store_commit_v7() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N>
__device__ inline void tma_store_wait_v7() {
  asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

__device__ inline void fence_async_shared_v7() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
  int CTA_GROUP,
  int NUM_STAGES,
  int L2_GROUP_SIZE,
  bool EPILOGUE_CACHE_MOD,
  bool DO_PROFILE,
  bool USE_L2_SWIZZLE,
  bool USE_HILBERT_SWIZZLE,
  bool USE_TMA_STORE,
  bool USE_TMA_PREFETCH
>
__global__
__cluster_dims__(CTA_GROUP, 1, 1)
__launch_bounds__(TB_SIZE)
void matmul_v7_kernel_cutlass(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  const __grid_constant__ CUtensorMap C_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  const int tid = threadIdx.x;
  const int bid = warp_uniform(blockIdx.x);
  const int num_bids = warp_uniform(gridDim.x);

  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;
  const int profiler_warp_id = warp_id == NUM_WARPS - 2 ? 0
                             : (warp_id == NUM_WARPS - 1 ? 1 : warp_id + 2);

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;

  Profiler profiler;
  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.init(num_entries, profiler_ptr, bid * NUM_WARPS + warp_id);
  }

  // CTA rank in a cluster
  int cta_rank;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.start(
      ProfilerTag::Setup,
      make_profiler_meta(profiler_warp_id, cta_rank)
    );
  }

  if constexpr (USE_TMA_PREFETCH) {
    if (warp_id == 0 && elect_sync()) {
      prefetch_tma_descriptor_v7(&A_tmap);
      prefetch_tma_descriptor_v7(&B_tmap);
      if constexpr (USE_TMA_STORE)
        prefetch_tma_descriptor_v7(&C_tmap);
    }
  }

  // set up smem
  // each CTA only loads half of B
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int C_epi_size = TMA_STORE_EPI_M * TMA_STORE_EPI_N * sizeof(nv_bfloat16);
  const int C_smem_base = smem + (A_size + B_size) * NUM_STAGES;

  // set up mbarrier and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  2 mbars for mainloop
  //                  2 mbars for epilogue
  // TMA warp -> TMA mbar -> MMA warp -> mainloop mbar -> epilogue warps
  //          <- MMA mbar <-          <- epilogue mbar
  const int tma_mbar_addr = C_smem_base + (USE_TMA_STORE ? C_epi_size * TMA_STORE_C_STAGES : 0);
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);  // both CTAs report TMA to CTA0 only
      mbarrier_init(mma_mbar_addr + i * 8, 1);          // CTA0 reports MMA to BOTH CTAs (multicast)
    }
    for (int i = 0; i < 2; i++) {
      mbarrier_init(mainloop_mbar_addr + i * 8, 1);     // CTA0 reports mainloop to BOTH CTAs (multicast)
      mbarrier_init(epilogue_mbar_addr + i * 8, 4 * CTA_GROUP * WARP_SIZE);  // 4 epilogue warps x both CTAs report to CTA0 only
    }
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }

  if constexpr (CTA_GROUP > 1) {
    // notice .relaxed
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  }
  else {
    // visible to all threads in a threadblock
    __syncthreads();
  }
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

  const int num_tiles = grid_m * grid_n;
  const int num_iters = K / BLOCK_K;

  auto compute_bid = [&](int bid) -> std::tuple<int, int> {
    // bid must run along M-mode first so that .cta_group::2 works correctly
    constexpr int GROUP_M = 2;
    int bid_m, bid_n;
    if constexpr (USE_HILBERT_SWIZZLE) {
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
    else if constexpr (USE_L2_SWIZZLE) {
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
    else {
      bid_m = bid / (grid_n * GROUP_M) * GROUP_M + (bid % GROUP_M);
      bid_n = (bid / GROUP_M) % grid_n;
    }
    return {bid_m, bid_n};
  };

  if (warp_id == NUM_WARPS - 2) {
    // TMA warp
    if (elect_sync()) {
      int tma_stage = 0;
      int mma_phase = 1;  // the initial MMA phase is 0, and it is available. so we initialize it with 1.

      // both CTA ranks update tx-count of CTA0's mbar
      // https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/copy_sm100_tma.hpp#L113-L115
      const int tma_mbar_addr_ = tma_mbar_addr & 0xFEFFFFFF;

      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        auto [bid_m, bid_n] = compute_bid(this_bid);
        const int off_m = bid_m * BLOCK_M;
        const int off_n = bid_n * BLOCK_N + cta_rank * (BLOCK_N / CTA_GROUP);

        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          // do address calculations before mbarrier_wait()
          const int mbar_addr = tma_mbar_addr_ + tma_stage * 8;
          const int A_smem = smem + tma_stage * (A_size + B_size);
          const int B_smem = A_smem + A_size;

          // wait for MMA on the local MMA mbar
          if constexpr (DO_PROFILE) profiler.start(
            ProfilerTag::WaitMMA,
            make_profiler_meta(profiler_warp_id, cta_rank)
          );
          mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);
          if constexpr (DO_PROFILE) profiler.stop();

          if constexpr (DO_PROFILE) profiler.start(
            ProfilerTag::IssueTMA,
            make_profiler_meta(profiler_warp_id, cta_rank)
          );
          tma_3d_g2s<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, iter_k, mbar_addr);
          tma_3d_g2s<CTA_GROUP>(B_smem, &B_tmap, 0, off_n, iter_k, mbar_addr);
          mbarrier_arrive_expect_tx(mbar_addr, A_size + B_size);
          if constexpr (DO_PROFILE) profiler.stop();

          // flip phase when we have cycled through all TMA buffers
          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0)
            mma_phase ^= 1;
        }
      }
    }
  }
  else if (warp_id == NUM_WARPS - 1) {
    // MMA warp
    // allocate double-buffered tmem for output (issued by both CTAs)
    // do it here so it doesn't block TMA
    tcgen05_alloc<CTA_GROUP>(epilogue_mbar_addr + 8 * 2, BLOCK_N * 2);

    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    constexpr uint32_t MMA_M = BLOCK_M * CTA_GROUP;  // 128 for 1SM, 256 for 2SM
    constexpr uint32_t MMA_N = BLOCK_N;
    constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                              | (1U << 7U)   // atype=BF16
                              | (1U << 10U)  // btype=BF16
                              | (MMA_N >> 3U << 17U)
                              | (MMA_M >> 4U << 24U)
                              ;

    // set up shared memory descriptors for A and B
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
    // 128-byte swizzling. LBO is implied to be 1.
    constexpr uint64_t AB_desc = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

    // only CTA0 issue MMA
    if (cta_rank == 0 && elect_sync()) {
      int tma_stage = 0;
      int tma_phase = 0;
      int mainloop_stage = 0;
      int epilogue_phase = 1;  // the initial epilogue phase is 0, and it is available. hence we initialize it with 1.

      constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        // wait for epilogue to finish
        if constexpr (DO_PROFILE) {
          profiler.start(
            ProfilerTag::WaitEpilogue,
            make_profiler_meta(profiler_warp_id, cta_rank)
          );
        }
        mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
        if constexpr (DO_PROFILE) profiler.stop();

        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          // select TMA and tmem buffer
          const int A_smem = smem + tma_stage * (A_size + B_size);
          const int B_smem = A_smem + A_size;
          const int tmem = mainloop_stage * BLOCK_N;

          uint64_t a_desc = AB_desc | (A_smem >> 4);
          uint64_t b_desc = AB_desc | (B_smem >> 4);

          // wait for TMA on the local TMA mbar
          if constexpr (DO_PROFILE) profiler.start(
            ProfilerTag::WaitTMA,
            make_profiler_meta(profiler_warp_id, cta_rank)
          );
          mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
          asm volatile("tcgen05.fence::after_thread_sync;");  // (why) do we need this? from DeepGEMM
          if constexpr (DO_PROFILE) profiler.stop();

          if constexpr (DO_PROFILE) profiler.start(
            ProfilerTag::IssueMMA,
            make_profiler_meta(profiler_warp_id, cta_rank)
          );

          // manually unroll 1st iteration to disable accumulation
          tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, iter_k);
          for (int k = 1; k < BLOCK_K / MMA_K; k++) {
            a_desc += (32 >> 4);  // next 32-byte
            b_desc += (32 >> 4);
            tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
          }

          // this signals to mbar on BOTH CTAs
          tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + tma_stage * 8, cta_mask);
          if constexpr (DO_PROFILE) profiler.stop();

          // flip phase when we have cycled through all TMA buffers
          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0)
            tma_phase ^= 1;
        }

        // signal when tcgen05 finishes with the main loop to BOTH CTAs
        tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr + mainloop_stage * 8, cta_mask);

        // flip phase when we have cycled through all tmem buffers
        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0)
          epilogue_phase ^= 1;
      }
    }
  }
  else {
    int mainloop_stage = 0;
    int mainloop_phase = 0;
    int c_store_count = 0;

    // named barrier
    auto epilogue_sync = []() {
      asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(4 * WARP_SIZE) : "memory");
    };

    // epilogue warps
    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      auto [bid_m, bid_n] = compute_bid(this_bid);

      // wait for mainloop to finish
      if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(
        ProfilerTag::WaitMainloop,
        make_profiler_meta(profiler_warp_id, cta_rank)
      );
      if (warp_id == 0)
        mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      epilogue_sync();
      asm volatile("tcgen05.fence::after_thread_sync;");
      if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

      if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(
        ProfilerTag::Epilogue,
        make_profiler_meta(profiler_warp_id, cta_rank)
      );
      constexpr int WIDTH = 16;
      if constexpr (USE_TMA_STORE) {
        constexpr int NUM_SUBTILES = BLOCK_N / TMA_STORE_EPI_N;
        static_assert(TMA_STORE_EPI_M == BLOCK_M);
        static_assert(TMA_STORE_EPI_N % WIDTH == 0);

        for (int subtile = 0; subtile < NUM_SUBTILES; subtile++) {
          const int c_buf = c_store_count % TMA_STORE_C_STAGES;
          const int c_smem_buf = C_smem_base + c_buf * C_epi_size;

          if (c_store_count >= TMA_STORE_C_STAGES) {
            if (warp_id == 0 && elect_sync())
              tma_store_wait_v7<TMA_STORE_C_STAGES - 1>();
            epilogue_sync();
          }

          for (int w = 0; w < TMA_STORE_EPI_N / WIDTH; w++) {
            const int t_row = cta_rank * 128 + warp_id * 32;
            const int t_col = mainloop_stage * BLOCK_N + subtile * TMA_STORE_EPI_N + w * WIDTH;
            const int t_addr = (t_row << 16) + t_col;
            const int c_smem_thread = c_smem_buf
                                    + tid * TMA_STORE_EPI_N * (int)sizeof(nv_bfloat16)
                                    + w * WIDTH * (int)sizeof(nv_bfloat16);

            asm volatile(
              "{\n"
              ".reg .f32 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;\n"
              ".reg .b32 b0, b1, b2, b3, b4, b5, b6, b7;\n"
              "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
              "  {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15}, [%2];\n"
              "tcgen05.wait::ld.sync.aligned;\n"
              "cvt.rn.bf16x2.f32 b0, f1, f0;\n"
              "cvt.rn.bf16x2.f32 b1, f3, f2;\n"
              "cvt.rn.bf16x2.f32 b2, f5, f4;\n"
              "cvt.rn.bf16x2.f32 b3, f7, f6;\n"
              "cvt.rn.bf16x2.f32 b4, f9, f8;\n"
              "cvt.rn.bf16x2.f32 b5, f11, f10;\n"
              "cvt.rn.bf16x2.f32 b6, f13, f12;\n"
              "cvt.rn.bf16x2.f32 b7, f15, f14;\n"
              "st.shared.v4.b32 [%0], {b0, b1, b2, b3};\n"
              "st.shared.v4.b32 [%1], {b4, b5, b6, b7};\n"
              "}"
              :: "r"(c_smem_thread), "r"(c_smem_thread + 16), "r"(t_addr)
            );
          }

          fence_async_shared_v7();
          epilogue_sync();

          if (warp_id == 0 && elect_sync()) {
            const int g_col = bid_n * BLOCK_N + subtile * TMA_STORE_EPI_N;
            const int g_row = bid_m * BLOCK_M;
            tma_2d_smem2gmem_v7(&C_tmap, c_smem_buf, g_col, g_row);
            tma_store_commit_v7();
          }
          c_store_count++;
          epilogue_sync();
        }
      }
      else {
        // load 16 columns from tmem at a time -> store 32 bytes per thread to smem
        // (still strided though)
        for (int n = 0; n < BLOCK_N / WIDTH; n++) {
          // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a
          // Layout A
          // select tmem buffer
          const int t_row = cta_rank * 128 + warp_id * 32;
          const int t_col = mainloop_stage * BLOCK_N + n * WIDTH;
          const int t_addr = (t_row << 16) + t_col;

          const int g_row = bid_m * BLOCK_M + tid;
          const int g_col = bid_n * BLOCK_N + n * WIDTH;

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
              :: "l"(C_ptr + g_row * N + g_col), "r"(t_addr)
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
              :: "l"(C_ptr + g_row * N + g_col), "r"(t_addr)
            );
        }
      }
      if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

      // all epilogue warps report to CTA0 mbar
      const int mbar_addr = (epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
      mbarrier_arrive(mbar_addr);

      // flip phase when we have cycled through all tmem buffers
      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0)
        mainloop_phase ^= 1;
    }

    if constexpr (CTA_GROUP > 1) {
      // notice .relaxed
      asm volatile("barrier.cluster.arrive.relaxed.aligned;");
      asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
      epilogue_sync();
    }

    if (warp_id == 0)  // deallocate tmem (issued by both CTAs)
      tcgen05_dealloc<CTA_GROUP>(0, BLOCK_N * 2);
  }

  if constexpr (USE_TMA_STORE) {
    if (warp_id == 0 && elect_sync())
      tma_store_wait_v7<0>();
  }

  if constexpr (DO_PROFILE) if (elect_sync()) profiler.flush();
}

template <
  int BLOCK_N,
  int CTA_GROUP,
  int NUM_BLOCKS,
  int L2_GROUP_SIZE,
  bool EPILOGUE_CACHE_MOD,
  bool DO_PROFILE,
  bool USE_L2_SWIZZLE,
  bool USE_HILBERT_SWIZZLE,
  bool USE_TMA_STORE,
  bool USE_TMA_PREFETCH
>
void matmul_v7_launch(
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
  CUtensorMap A_tmap, B_tmap, C_tmap{};
  init_tmap_3d_128B(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
  init_tmap_3d_128B(&B_tmap, B_ptr, N, K, BLOCK_N / CTA_GROUP, BLOCK_K);
  if constexpr (USE_TMA_STORE) {
    init_tmap_2d_simple(
      &C_tmap, C_ptr, M, N, TMA_STORE_EPI_M, TMA_STORE_EPI_N,
      CU_TENSOR_MAP_SWIZZLE_NONE);
  }

  int grid = std::min(NUM_BLOCKS, (M / BLOCK_M) * (N / BLOCK_N));

  constexpr int AB_size = (BLOCK_M + BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int dynamic_size = AB_size + 2 * 8;  // TMA+MMA mbar for each stage
  constexpr int C_epi_size = USE_TMA_STORE ? TMA_STORE_C_STAGES * TMA_STORE_EPI_M * TMA_STORE_EPI_N * sizeof(nv_bfloat16) : 0;
  constexpr int static_size = C_epi_size + 4 * 8 + 4;  // 2 mainloop+epilogue mbar, and tmem address

  constexpr int sm100_size = 227 * 1024;
  constexpr int NUM_STAGES = (sm100_size - static_size) / dynamic_size;
  constexpr int smem_size = NUM_STAGES * dynamic_size + static_size;

  auto this_kernel = matmul_v7_kernel_cutlass<
    BLOCK_N, CTA_GROUP, NUM_STAGES, L2_GROUP_SIZE,
    EPILOGUE_CACHE_MOD, DO_PROFILE, USE_L2_SWIZZLE, USE_HILBERT_SWIZZLE, USE_TMA_STORE, USE_TMA_PREFETCH
  >;
  cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_tmap, C_ptr, M, N, K, profiler_ptr, num_entries);
  check_cuda(cudaGetLastError());
}

void matmul_v7a(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // matmul_v7_launch<256, 2, 148, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
  matmul_v7_launch<256, 2, 20, 6, false, false, false, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7b(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 128, 6, false, false, false, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7c(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 128, 6, true, false, false, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7a_g6(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 20, 6, false, false, true, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7b_g6(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 128, 6, false, false, true, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7c_g6(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 128, 6, true, false, true, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7a_hilbert(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 20, 6, false, false, false, true, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7a_hilbert_l1noalloc(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 20, 6, true, false, false, true, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7a_hilbert_l1noalloc_tma_store(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 20, 6, true, false, false, true, true, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7a_hilbert_l1noalloc_tma_store_prefetch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 20, 6, true, false, false, true, true, true>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7b_hilbert(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 128, 6, false, false, false, true, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void matmul_v7c_hilbert(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v7_launch<256, 2, 128, 6, true, false, false, true, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0);
}

void profile_matmul_v7a(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  matmul_v7_launch<256, 2, 20, 6, false, true, false, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries);
}

void profile_matmul_v7b(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  matmul_v7_launch<256, 2, 128, 6, false, true, false, false, false, false>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries);
}

void profile_matmul_v7a_hilbert_l1noalloc_tma_store_prefetch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  matmul_v7_launch<256, 2, 20, 6, true, true, false, true, true, true>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries);
}
// profile timeline
/*
source ~/cutedsl44.sh && python bench_flops.py \
  --profile \
  --kernel matmul_v7a_hilbert_l1noalloc_tma_store_prefetch \
  --shape 4096,4096,4096 \
  --profile-output profile_v7a_hilbert_l1noalloc_tma_store_prefetch.json.gz \
  --no-verbose-build
*/