#include "common.h"

#include "DeepGEMM/third-party/cutlass/include/cute/arch/cluster_sm90.hpp"
#include "DeepGEMM/third-party/cutlass/include/cutlass/arch/barrier.h"
#include "DeepGEMM/third-party/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "DeepGEMM/third-party/cutlass/include/cutlass/pipeline/pipeline.hpp"

#include <cuda_bf16.h>
#include <tuple>

constexpr int NUM_WARPS = 6;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;
constexpr int NUM_CLC_STAGES = 1;

__device__ inline void named_barrier_sync(int barrier_id, int num_threads) {
  asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads) : "memory");
}

template <int CTA_GROUP>
using ClcClusterShape = cute::Shape<cute::Int<CTA_GROUP>, cute::_1, cute::_1>;

template <int CTA_GROUP>
using ClcPipeline = cutlass::PipelineCLCFetchAsync<NUM_CLC_STAGES, ClcClusterShape<CTA_GROUP>>;

template <int CTA_GROUP>
using ClcScheduler = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClcClusterShape<CTA_GROUP>, NUM_CLC_STAGES>;

template <int CTA_GROUP>
using ClcResponse = typename ClcScheduler<CTA_GROUP>::CLCResponse;

template <int CTA_GROUP>
struct ClcSharedStorage {
  alignas(16) ClcResponse<CTA_GROUP> clc_response[NUM_CLC_STAGES];
  alignas(16) typename ClcPipeline<CTA_GROUP>::SharedStorage pipeline_storage;
};

template <int CTA_GROUP>
static __device__ inline void issue_clc_query(
  typename ClcPipeline<CTA_GROUP>::PipelineState state,
  uint32_t mbarrier_addr,
  ClcResponse<CTA_GROUP> *clc_response_ptr
) {
  uint32_t result_addr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void*>(&clc_response_ptr[state.index()]));
  asm volatile(
    "{\n\t"
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
    "}\n"
    :
    : "r"(result_addr), "r"(mbarrier_addr));
}

__device__ inline bool decode_clc_response(uint32_t result_addr, int &next_cluster_bid, int cluster_dim_x) {
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

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, bool CTA_LOCAL_WORK_BROADCAST>
__global__
__cluster_dims__(CTA_GROUP, 1, 1)
__launch_bounds__(TB_SIZE)
void matmul_v10_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int num_tiles = grid_m * grid_n;

  int cta_rank;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int NUM_STAGES = 7;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[NUM_STAGES * 2 + 4];
  __shared__ int tmem_addr[1];
  __shared__ int next_cluster_bid_shared;
  __shared__ int next_has_work_shared;
  __shared__ __align__(16) char clc_shared_bytes[sizeof(ClcSharedStorage<CTA_GROUP>)];
  auto &clc_shared = *reinterpret_cast<ClcSharedStorage<CTA_GROUP> *>(clc_shared_bytes);

  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

  using ThisClcPipeline = ClcPipeline<CTA_GROUP>;
  using ThisClcPipelineState = typename ThisClcPipeline::PipelineState;
  constexpr int CTA_WORK_SYNC_BARRIER = 7;

  const bool is_producer_warp = (cta_rank == 0 && warp_id == 0);
  const bool is_consumer_warp = (warp_id == 0) || (warp_id >= 2) || (cta_rank == 0 && warp_id == 1);
  const bool uses_clc_consumer = CTA_LOCAL_WORK_BROADCAST ? (warp_id == 0) : is_consumer_warp;
  const int cta_active_threads = WARP_SIZE * (cta_rank == 0 ? NUM_WARPS : (NUM_WARPS - 1));

  typename ThisClcPipeline::Params clc_params;
  clc_params.transaction_bytes = 16;
  clc_params.role = is_producer_warp
    ? ThisClcPipeline::ThreadCategory::ProducerConsumer
    : (uses_clc_consumer ? ThisClcPipeline::ThreadCategory::Consumer : ThisClcPipeline::ThreadCategory::NonParticipant);
  clc_params.producer_blockid = 0;
  clc_params.producer_arv_count = 1;
  clc_params.consumer_arv_count = CTA_LOCAL_WORK_BROADCAST ? (WARP_SIZE * CTA_GROUP) : (WARP_SIZE * (CTA_GROUP + 1 + 4 * CTA_GROUP));
  clc_params.initializing_warp = 0;
  ThisClcPipeline clc_pipeline(clc_shared.pipeline_storage, clc_params, ClcClusterShape<CTA_GROUP>{});

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }
    for (int i = 0; i < 2; i++) {
      mbarrier_init(mainloop_mbar_addr + i * 8, 1);
      mbarrier_init(epilogue_mbar_addr + i * 8, 4 * CTA_GROUP);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  else if (warp_id == 1) {
    static_assert(BLOCK_N * 2 <= 512);
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    tcgen05_alloc<CTA_GROUP>(addr, BLOCK_N * 2);
  }

  if constexpr (CTA_GROUP > 1) {
    cute::cluster_sync();
  } else {
    __syncthreads();
  }

  const int taddr = tmem_addr[0];

  constexpr int MMA_M = BLOCK_M * CTA_GROUP;
  constexpr uint32_t i_desc = (1U << 4U)
                            | (1U << 7U)
                            | (1U << 10U)
                            | ((uint32_t)BLOCK_N >> 3U << 17U)
                            | ((uint32_t)MMA_M >> 4U << 24U);

  auto make_desc = [](int addr) -> uint64_t {
    const int sbo = 8 * 128;
    return desc_encode(addr) | (desc_encode(sbo) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
  };

  auto load = [&](int tma_stage, int mma_phase, int iter_k, int bid_m, int bid_n) {
    mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);

    const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;
    const int A_smem = smem + tma_stage * (A_size + B_size);
    const int B_smem = A_smem + A_size;

    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N + cta_rank * (BLOCK_N / CTA_GROUP);
    const int off_k = iter_k * BLOCK_K;

    tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
    tma_3d_gmem2smem<CTA_GROUP>(B_smem, &B_tmap, 0, off_n, off_k / 64, mbar_addr);
    mbarrier_arrive_expect_tx(mbar_addr, A_size + B_size);
  };

  auto compute = [&](int tma_stage, int tma_phase, int mainloop_stage, int enable_input_d) {
    mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int A_smem = smem + tma_stage * (A_size + B_size);
    const int B_smem = A_smem + A_size;
    const int tmem = taddr + mainloop_stage * BLOCK_N;

    tcgen05_mma_f16<CTA_GROUP>(tmem, make_desc(A_smem), make_desc(B_smem), i_desc, enable_input_d);
    for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
      uint64_t a_desc = make_desc(A_smem + k2 * 32);
      uint64_t b_desc = make_desc(B_smem + k2 * 32);
      tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
    }

    for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
      for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k1 * (BLOCK_N / CTA_GROUP) * 128 + k2 * 32);
        tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
      }
    }

    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + tma_stage * 8, cta_mask);
  };

  auto epilogue = [&](int mainloop_stage, int bid_m, int bid_n) {
    const int epilogue_warp_id = warp_id % 4;
    const int epilogue_tid = epilogue_warp_id * WARP_SIZE + lane_id;

    for (int n = 0; n < BLOCK_N / 8; n++) {
      float tmp[8];
      const int row = cta_rank * 128 + epilogue_warp_id * 32;
      const int col = mainloop_stage * BLOCK_N + n * 8;
      const int addr = taddr + (row << 16) + col;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                  : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                    "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                  : "r"(addr));
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      nv_bfloat162 out[4];
      for (int i = 0; i < 4; i++) {
        out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
      }

      nv_bfloat16 *out_ptr = C_ptr + (bid_m * BLOCK_M + epilogue_tid) * N + (bid_n * BLOCK_N + n * 8);
      reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
    }
  };

  auto compute_bid = [&](int bid) -> std::tuple<int, int> {
    int bid_m = 0;
    int bid_n = 0;
    constexpr int GROUP_M = 2;
    constexpr int GROUP_SIZE = 8;
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
    return {bid_m, bid_n};
  };

  const int num_iters = K / BLOCK_K;
  const int launched_clusters = static_cast<int>(cute::cluster_grid_dims().x);
  int current_cluster_bid = static_cast<int>(cute::cluster_id_in_grid().x);
  bool has_work = current_cluster_bid < num_tiles;

  auto fetch_next_work = [&](ThisClcPipelineState &consumer_state, int current_cluster_bid, bool current_has_work) {
    int fallback_cluster_bid = current_cluster_bid + launched_clusters;
    int fallback_valid = static_cast<int>(fallback_cluster_bid < num_tiles);
    int next_cluster_bid = fallback_cluster_bid;
    int valid = 0;

    auto token = clc_pipeline.consumer_try_wait(consumer_state);
    clc_pipeline.consumer_wait(consumer_state, token);
    if (lane_id == 0) {
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&clc_shared.clc_response[consumer_state.index()]);
      valid = static_cast<int>(decode_clc_response(smem_addr, next_cluster_bid, CTA_GROUP));
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

  auto sync_cta_work_state = [&]() {
    if constexpr (CTA_LOCAL_WORK_BROADCAST) {
      if (warp_id == 0 || warp_id >= 2 || (cta_rank == 0 && warp_id == 1)) {
        named_barrier_sync(CTA_WORK_SYNC_BARRIER, cta_active_threads);
      }
    }
  };

  if (warp_id == 0) {
    int tma_stage = 0;
    int mma_phase = 1;
    ThisClcPipelineState consumer_state;
    auto producer_state = cutlass::make_producer_start_state<ThisClcPipeline>();

    while (has_work) {
      auto [bid_m, bid_n] = compute_bid(current_cluster_bid);

      if (is_producer_warp) {
        clc_pipeline.producer_acquire(producer_state);
        if (elect_sync()) {
          issue_clc_query<CTA_GROUP>(producer_state, clc_pipeline.producer_get_barrier(producer_state), clc_shared.clc_response);
        }
        ++producer_state;
      }

      if (elect_sync()) {
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          load(tma_stage, mma_phase, iter_k, bid_m, bid_n);
          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0) {
            mma_phase ^= 1;
          }
        }
      }

      if constexpr (CTA_LOCAL_WORK_BROADCAST) {
        auto [next_cluster_bid, next_has_work] = fetch_next_work(consumer_state, current_cluster_bid, has_work);
        if (elect_sync()) {
          next_cluster_bid_shared = next_cluster_bid;
          next_has_work_shared = static_cast<int>(next_has_work);
        }
        sync_cta_work_state();
        has_work = next_has_work_shared != 0;
        current_cluster_bid = next_cluster_bid_shared;
      } else {
        auto [next_cluster_bid, next_has_work] = fetch_next_work(consumer_state, current_cluster_bid, has_work);
        has_work = next_has_work;
        current_cluster_bid = next_cluster_bid;
      }
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
      if (elect_sync()) {
        mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          compute(tma_stage, tma_phase, mainloop_stage, iter_k > 0);
          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0) {
            tma_phase ^= 1;
          }
        }

        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr + mainloop_stage * 8, cta_mask);
        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0) {
          epilogue_phase ^= 1;
        }
      }

      if constexpr (CTA_LOCAL_WORK_BROADCAST) {
        sync_cta_work_state();
        local_has_work = next_has_work_shared != 0;
        local_cluster_bid = next_cluster_bid_shared;
      } else {
        auto [next_cluster_bid, next_has_work] = fetch_next_work(consumer_state, local_cluster_bid, local_has_work);
        local_has_work = next_has_work;
        local_cluster_bid = next_cluster_bid;
      }
    }
  }
  else if (warp_id >= 2) {
    int mainloop_stage = 0;
    int mainloop_phase = 0;
    int local_cluster_bid = current_cluster_bid;
    bool local_has_work = has_work;
    ThisClcPipelineState consumer_state;

    while (local_has_work) {
      mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      auto [bid_m, bid_n] = compute_bid(local_cluster_bid);
      epilogue(mainloop_stage, bid_m, bid_n);

      if (elect_sync()) {
        const int mbar_addr = (epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
        mbarrier_arrive(mbar_addr);
      }

      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0) {
        mainloop_phase ^= 1;
      }

      if constexpr (CTA_LOCAL_WORK_BROADCAST) {
        sync_cta_work_state();
        local_has_work = next_has_work_shared != 0;
        local_cluster_bid = next_cluster_bid_shared;
      } else {
        auto [next_cluster_bid, next_has_work] = fetch_next_work(consumer_state, local_cluster_bid, local_has_work);
        local_has_work = next_has_work;
        local_cluster_bid = next_cluster_bid;
      }
    }
  }

  if constexpr (CTA_GROUP > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  } else {
    __syncthreads();
  }

  if (warp_id == 0) {
    tcgen05_dealloc<CTA_GROUP>(taddr, BLOCK_N * 2);
  }
}

namespace {

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, bool CTA_LOCAL_WORK_BROADCAST>
void matmul_v10_launch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  CUtensorMap A_tmap, B_tmap;

  auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr, uint64_t global_height, uint32_t shared_height) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank] = {64, global_height, static_cast<uint64_t>(K) / 64};
    uint64_t globalStrides[rank - 1] = {static_cast<uint64_t>(K) * sizeof(nv_bfloat16), 128};
    uint32_t boxDim[rank] = {64, shared_height, static_cast<uint32_t>(BLOCK_K) / 64};
    uint32_t elementStrides[rank] = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      (void *)ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    check_cu(err);
  };

  init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
  init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N / CTA_GROUP);

  const int num_tiles = (M / BLOCK_M) * (N / BLOCK_N);
  const int grid = num_tiles * CTA_GROUP;
  const int size_AB = (BLOCK_M + BLOCK_N / CTA_GROUP) * BLOCK_K * 7;
  const int smem_size = size_AB * sizeof(nv_bfloat16);

  auto this_kernel = matmul_v10_kernel<BLOCK_N, BLOCK_K, CTA_GROUP, CTA_LOCAL_WORK_BROADCAST>;
  if (smem_size > 48'000) {
    check_cuda(cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
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

  check_cuda(cudaLaunchKernelEx(&launch_config, this_kernel, A_tmap, B_tmap, C_ptr, M, N, K));
  check_cuda(cudaGetLastError());
}

}  // namespace

void matmul_v10(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v10_launch<256, 64, 2, false>(A_ptr, B_ptr, C_ptr, M, N, K);
}

void matmul_v10_2(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v10_launch<256, 64, 2, true>(A_ptr, B_ptr, C_ptr, M, N, K);
}