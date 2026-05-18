// v9: Based on DeepGEMM SM100 BF16 GEMM architecture
// No CuTE library, plain CUDA + inline PTX, following v1-v7 style
// Key features:
//   - SM100 UMMA descriptors (version=1) with SBO/LBO
//   - 3 warp roles: TMA load (warp 0), MMA (warp 1 on rank 0), epilogue (warps 2-5)
//   - Stage merging: if NUM_STAGES >= 8, merge to reduce umma_arrive overhead
//   - TMA store epilogue with swizzle
//   - Persistent scheduler with L2 swizzle grouping

#include "common.h"

#include <cuda_bf16.h>

// ---- Config constants ----
constexpr int BLOCK_M = 128;
constexpr int BLOCK_K_BASE = 64;  // base K before merging
constexpr int MMA_K = 16;        // UMMA K-stride for BF16

// Layout D (TMEM): 128 rows per wave
constexpr int LAYOUT_AD_M = 128;

// Swizzle mode for A/B: 128 bytes
constexpr int SWIZZLE_AB = 128;
// Swizzle mode for C/D: 128 bytes for BF16 output
constexpr int SWIZZLE_CD = 128;

// Store atom dimensions
constexpr int STORE_BLOCK_M = 128;  // min(BLOCK_M, LAYOUT_AD_M)

// TMA store double-buffered stages
constexpr int NUM_TMA_STORE_STAGES = 2;

// ---- Thread config ----
constexpr int NUM_NON_EPILOGUE_WARPS = 4;  // warps 0-3 (128 threads)
constexpr int NUM_EPILOGUE_WARPS = 4;       // warps 4-7 (128 threads)
constexpr int NUM_WARPS = NUM_NON_EPILOGUE_WARPS + NUM_EPILOGUE_WARPS;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;  // 256

// ---- Helper: build SM100 UMMA shared memory descriptor ----
// SmemDescriptor layout (64 bits):
//   [0:14)   start_address >> 4
//   [16:30)  leading_byte_offset >> 4
//   [32:46)  stride_byte_offset >> 4
//   [46:48)  version (1 for SM100)
//   [49:52)  base_offset
//   [52:53)  lbo_mode
//   [61:64)  layout_type: 0=NONE, 1=128B_BASE32B, 2=128B, 4=64B, 6=32B

__device__ inline
uint64_t make_smem_desc(int smem_addr, uint32_t sbo, uint32_t lbo) {
  // layout_type for 128B swizzle = 2
  constexpr uint64_t layout_type = 2;  // SWIZZLE_128B
  constexpr uint64_t version = 1;      // SM100

  uint64_t desc = 0;
  desc |= (uint64_t)((smem_addr >> 4) & 0x3FFF);          // start_address [0:14)
  desc |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;          // leading_byte_offset [16:30)
  desc |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;          // stride_byte_offset [32:46)
  desc |= version << 46;                                    // version [46:48)
  // base_offset = 0 [49:52)
  // lbo_mode = 0 [52:53)
  desc |= layout_type << 61;                                // layout_type [61:64)
  return desc;
}

// For K-major inputs with 128B swizzle:
// Atom: 8 rows x 128 bytes (= 64 BF16 = BLOCK_K)
// SBO = stride between atoms on MN axis = (128/16)*BLOCK_K*sizeof(BF16) = 8*64*2 = 1024
// LBO = stride between atoms on K axis = 0 (only 1 atom on K for BLOCK_K=64)

__device__ inline
uint64_t make_k_major_desc(int smem_base, int stage_offset) {
  int addr = smem_base + stage_offset;
  constexpr uint32_t sbo = (128 / 16) * BLOCK_K_BASE * sizeof(nv_bfloat16);  // 1024
  constexpr uint32_t lbo = 0;
  return make_smem_desc(addr, sbo, lbo);
}

// Advance desc_lo for K iterations within a merged stage block
// For K-major: stride on K axis is 1 element (contiguous), advance by MMA_K * sizeof(BF16)
// desc_lo += (MMA_K * sizeof(BF16)) >> 4
__device__ inline
uint32_t advance_desc_lo_k_major(uint32_t base_lo, uint32_t k_idx) {
  // For K-major with swizzle-128B: stride_k = 1 element
  return base_lo + (k_idx * MMA_K * sizeof(nv_bfloat16)) / 16;
}

// ---- Named barrier helpers ----
__device__ inline
void named_barrier_sync(int barrier_id, int num_threads) {
  asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads) : "memory");
}

// ---- TMA store helpers ----
__device__ inline
void tma_store_2d(const void *tmap_ptr, int smem_addr, int x, int y) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
    :: "l"(tmap_ptr), "r"(smem_addr), "r"(x), "r"(y)
    : "memory");
}

__device__ inline
void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int COUNT>
__device__ inline
void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(COUNT) : "memory");
}

__device__ inline
void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;");
}

// ---- TMEM load helpers ----
// Load 4 x 32-bit from TMEM (float: 4 elements per bank group)
__device__ inline
void tmem_load_32dp32b_x4(uint32_t tmem_addr, uint32_t &v0, uint32_t &v1, uint32_t &v2, uint32_t &v3) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
    : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(tmem_addr));
}

// Load 8 x 32-bit from TMEM (float: 8 elements per bank group, for BF16 epilogue)
__device__ inline
void tmem_load_32dp32b_x8(uint32_t tmem_addr,
    uint32_t &v0, uint32_t &v1, uint32_t &v2, uint32_t &v3,
    uint32_t &v4, uint32_t &v5, uint32_t &v6, uint32_t &v7) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
    : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3),
      "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7) : "r"(tmem_addr));
}

__device__ inline
void tmem_load_fence() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

// ---- Shared memory store helpers ----
__device__ inline
void st_shared_v4_u32(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
    :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

// ---- BF16 conversion ----
__device__ inline
uint32_t cvt_f32x2_to_bf16x2(uint32_t a, uint32_t b) {
  uint32_t result;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(__uint_as_float(b)), "f"(__uint_as_float(a)));
  return result;
}

// ---- tcgen05 fences ----
__device__ inline
void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ inline
void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// ---- Main kernel ----
template <int BLOCK_N, int NUM_STAGES_BASE, int NUM_SMs>
__global__
__launch_bounds__(TB_SIZE, 1)
void matmul_v9_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  const __grid_constant__ CUtensorMap CD_tmap,
  int M, int N, int K
) {
  // Stage merging: if NUM_STAGES_BASE >= 8, merge stages so we have at least 8
  constexpr int NUM_MIN_STAGES = 8;
  constexpr int NUM_STAGES_PER_MERGE = (NUM_STAGES_BASE >= 8) ? (NUM_STAGES_BASE / NUM_MIN_STAGES) : 1;
  constexpr int BLOCK_K = BLOCK_K_BASE * NUM_STAGES_PER_MERGE;
  constexpr int NUM_STAGES = NUM_STAGES_BASE / NUM_STAGES_PER_MERGE;

  // Store block N for swizzle atoms: SWIZZLE_CD / sizeof(float) = 32 floats = 64 BF16 elems
  // But since we store BF16: STORE_BLOCK_N = SWIZZLE_CD / sizeof(BF16) = 64
  constexpr int STORE_BLOCK_N = SWIZZLE_CD / sizeof(nv_bfloat16);  // 64

  // Epilogue stages: how many TMEM accumulator double-buffers
  constexpr int NUM_EPILOGUE_STAGES = (2 * BLOCK_N > 512) ? 1 : 2;

  // TMEM columns
  constexpr int NUM_ACCUM_TMEM_COLS = NUM_EPILOGUE_STAGES * BLOCK_N;
  constexpr int NUM_TMEM_COLS =
    (NUM_ACCUM_TMEM_COLS <= 32) ? 32 :
    (NUM_ACCUM_TMEM_COLS <= 64) ? 64 :
    (NUM_ACCUM_TMEM_COLS <= 128) ? 128 :
    (NUM_ACCUM_TMEM_COLS <= 256) ? 256 : 512;

  // Number of epilogue store threads = STORE_BLOCK_M = 128
  constexpr int NUM_UMMA_STORE_THREADS = STORE_BLOCK_M;  // 128

  // Shared memory layout:
  //   [0, SMEM_CD_SIZE): CD store buffer (double-buffered)
  //   [SMEM_CD_SIZE, ... ): A stages
  //   [after A, ...): B stages
  //   [after B, ...): barriers
  //   [after barriers, ...): tmem_addr
  constexpr int SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * SWIZZLE_CD;  // 128 * 128 = 16384
  constexpr int SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * NUM_TMA_STORE_STAGES;
  constexpr int SMEM_A_PER_STAGE = BLOCK_M * BLOCK_K_BASE * sizeof(nv_bfloat16);
  constexpr int SMEM_B_PER_STAGE = BLOCK_N * BLOCK_K_BASE * sizeof(nv_bfloat16);
  // For merged stages, A/B occupy NUM_STAGES_BASE stages worth of space
  // but conceptually grouped into NUM_STAGES groups of NUM_STAGES_PER_MERGE each
  // The total is the same: NUM_STAGES * (SMEM_A_PER_STAGE * NUM_STAGES_PER_MERGE)

  // Barrier layout: full[NUM_STAGES] + empty[NUM_STAGES] + tmem_full[NUM_EPILOGUE_STAGES] + tmem_empty[NUM_EPILOGUE_STAGES]
  constexpr int NUM_BARRIERS = NUM_STAGES * 2 + NUM_EPILOGUE_STAGES * 2;
  constexpr int BARRIER_SIZE = 8;  // uint64_t mbarrier

  extern __shared__ __align__(1024) char smem_buf[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_buf));

  // Pointers
  auto cd_smem_addr = [&](int stage) { return smem + stage * SMEM_CD_SIZE_PER_STAGE; };
  auto a_smem_addr = [&](int stage) { return smem + SMEM_CD_SIZE + stage * SMEM_A_PER_STAGE * NUM_STAGES_PER_MERGE; };
  auto b_smem_addr = [&](int stage) { return smem + SMEM_CD_SIZE + NUM_STAGES * SMEM_A_PER_STAGE * NUM_STAGES_PER_MERGE + stage * SMEM_B_PER_STAGE * NUM_STAGES_PER_MERGE; };
  const int barrier_base = smem + SMEM_CD_SIZE + NUM_STAGES * (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE) * NUM_STAGES_PER_MERGE;

  auto full_barrier = [&](int i) { return barrier_base + i * BARRIER_SIZE; };
  auto empty_barrier = [&](int i) { return barrier_base + (NUM_STAGES + i) * BARRIER_SIZE; };
  auto tmem_full_barrier = [&](int i) { return barrier_base + (NUM_STAGES * 2 + i) * BARRIER_SIZE; };
  auto tmem_empty_barrier = [&](int i) { return barrier_base + (NUM_STAGES * 2 + NUM_EPILOGUE_STAGES + i) * BARRIER_SIZE; };
  const int tmem_ptr_addr = barrier_base + NUM_BARRIERS * BARRIER_SIZE;

  const int tid = threadIdx.x;
  const int bid = warp_uniform(blockIdx.x);
  const int num_bids = warp_uniform(gridDim.x);
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int num_tiles = grid_m * grid_n;
  const int num_k_iters = K / BLOCK_K;

  // L2 swizzle grouping: we group in 1D blocks along N
  // Select group size from {8,16} to minimize L2 footprint
  auto compute_bid = [&](int block_idx) -> std::tuple<int, int> {
    constexpr int GROUP_SIZE = 8;  // DeepGEMM picks from {8,16}, 8 is usually good for BLOCK_N=256
    const int num_blocks_per_group = grid_m * GROUP_SIZE;
    const int group_idx = block_idx / num_blocks_per_group;
    const int first_n = group_idx * GROUP_SIZE;
    const int in_group = block_idx % num_blocks_per_group;
    const int num_in_group = min(GROUP_SIZE, grid_n - first_n);
    int m_idx = in_group / num_in_group;
    int n_idx = first_n + in_group % num_in_group;
    return {m_idx, n_idx};
  };

  // Initialize barriers (warp 1) and allocate TMEM (warp 2)
  if (warp_id == 1 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(full_barrier(i), 1);   // TMA warp arrives
      mbarrier_init(empty_barrier(i), 1);  // MMA warp arrives via tcgen05.commit
    }
    for (int i = 0; i < NUM_EPILOGUE_STAGES; i++) {
      mbarrier_init(tmem_full_barrier(i), 1);                    // MMA warp arrives via tcgen05.commit
      mbarrier_init(tmem_empty_barrier(i), NUM_UMMA_STORE_THREADS); // epilogue threads arrive
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  else if (warp_id == 2) {
    // Allocate TMEM
    tcgen05_alloc<1>(tmem_ptr_addr, NUM_TMEM_COLS);
  }
  __syncthreads();

  // Pipeline state
  int stage_idx = 0;
  int phase = 0;

  auto advance_pipeline = [&]() {
    stage_idx = (stage_idx + 1) % NUM_STAGES;
    if (stage_idx == 0) phase ^= 1;
  };

  // Instruction descriptor for UMMA
  // BF16 A/B, FP32 accumulate, K-major A and B
  // c_format=1 (FP32), a_format=1 (BF16), b_format=1 (BF16), a_major=0 (K), b_major=0 (K)
  constexpr uint32_t MMA_M = BLOCK_M;  // 128
  constexpr uint32_t MMA_N = BLOCK_N;
  constexpr uint32_t i_desc = (1U << 4U)                // c_format = FP32 (bit 4-5)
                            | (1U << 7U)                // a_format = BF16 (bit 7-9)
                            | (1U << 10U)               // b_format = BF16 (bit 10-12)
                            | (MMA_N >> 3U << 17U)      // n_dim (bit 17-22)
                            | (MMA_M >> 4U << 24U)      // m_dim (bit 24-28)
                            ;

  // Dispatch warps
  if (warp_id == 0 && elect_sync()) {
    // ============ TMA LOAD WARP ============
    int iter = -1;
    stage_idx = 0; phase = 0;

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      iter++;
      auto [bid_m, bid_n] = compute_bid(this_bid);
      int off_m = bid_m * BLOCK_M;
      int off_n = bid_n * BLOCK_N;

      for (int k_block = 0; k_block < num_k_iters; k_block++) {
        // Wait for MMA to release this stage
        mbarrier_wait(empty_barrier(stage_idx), phase ^ 1);

        // Issue TMA loads for A and B
        // A is K-major: (K, M) with 3D tmap (64, M, K/64)
        // B is K-major: (K, N) with 3D tmap (64, N, K/64)
        int a_smem = a_smem_addr(stage_idx);
        int b_smem = b_smem_addr(stage_idx);

        // For merged stages, we issue NUM_STAGES_PER_MERGE consecutive loads
        for (int merge = 0; merge < NUM_STAGES_PER_MERGE; merge++) {
          int k_idx = k_block * NUM_STAGES_PER_MERGE + merge;
          tma_3d_gmem2smem(a_smem + merge * SMEM_A_PER_STAGE, &A_tmap, 0, off_m, k_idx, full_barrier(stage_idx));
          tma_3d_gmem2smem(b_smem + merge * SMEM_B_PER_STAGE, &B_tmap, 0, off_n, k_idx, full_barrier(stage_idx));
        }

        // Arrive with expected TX bytes
        constexpr int tx_bytes = (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE) * NUM_STAGES_PER_MERGE;
        mbarrier_arrive_expect_tx(full_barrier(stage_idx), tx_bytes);

        advance_pipeline();
      }
    }
  }
  else if (warp_id == 1) {
    // ============ MMA WARP (all 32 lanes participate for __shfl_sync) ============
    int iter = -1;
    stage_idx = 0; phase = 0;

    // Build base UMMA descriptors for stage 0
    uint64_t a_desc_base = make_k_major_desc(a_smem_addr(0), 0);
    uint64_t b_desc_base = make_k_major_desc(b_smem_addr(0), 0);
    uint32_t a_desc_base_lo = (uint32_t)(a_desc_base);
    uint32_t b_desc_base_lo = (uint32_t)(b_desc_base);
    uint32_t a_desc_hi = (uint32_t)(a_desc_base >> 32);
    uint32_t b_desc_hi = (uint32_t)(b_desc_base >> 32);

    // Pre-compute per-stage desc_lo offsets: lane i holds stage i's offset
    uint32_t a_stage_lo = (lane_id < NUM_STAGES) ?
      a_desc_base_lo + lane_id * (SMEM_A_PER_STAGE * NUM_STAGES_PER_MERGE / 16) : 0u;
    uint32_t b_stage_lo = (lane_id < NUM_STAGES) ?
      b_desc_base_lo + lane_id * (SMEM_B_PER_STAGE * NUM_STAGES_PER_MERGE / 16) : 0u;

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      iter++;
      int accum_stage = iter % NUM_EPILOGUE_STAGES;
      int accum_phase = (iter / NUM_EPILOGUE_STAGES) & 1;

      // Wait for epilogue to release this TMEM accumulator
      mbarrier_wait(tmem_empty_barrier(accum_stage), accum_phase ^ 1);
      tcgen05_fence_after_thread_sync();

      for (int k_block = 0; k_block < num_k_iters; k_block++) {
        // Wait for TMA to fill this stage
        mbarrier_wait(full_barrier(stage_idx), phase);
        tcgen05_fence_after_thread_sync();

        // All lanes participate in shfl_sync to broadcast per-stage desc_lo
        uint32_t a_lo = __shfl_sync(0xffffffff, a_stage_lo, stage_idx);
        uint32_t b_lo = __shfl_sync(0xffffffff, b_stage_lo, stage_idx);

        // Only elected thread issues MMA and commits
        if (elect_sync()) {
          for (int k = 0; k < BLOCK_K / MMA_K; k++) {
            int atom_k = k * MMA_K / BLOCK_K_BASE;
            int k_in_atom = k * MMA_K % BLOCK_K_BASE;

            uint32_t a_lo_k = a_lo + (uint32_t)((atom_k * BLOCK_M * BLOCK_K_BASE + k_in_atom) * (int)sizeof(nv_bfloat16)) / 16;
            uint32_t b_lo_k = b_lo + (uint32_t)((atom_k * BLOCK_N * BLOCK_K_BASE + k_in_atom) * (int)sizeof(nv_bfloat16)) / 16;

            uint64_t a_desc = ((uint64_t)a_desc_hi << 32) | a_lo_k;
            uint64_t b_desc = ((uint64_t)b_desc_hi << 32) | b_lo_k;

            int tmem_offset = accum_stage * BLOCK_N;
            int enable_d = (k_block > 0 || k > 0) ? 1 : 0;
            tcgen05_mma_f16<1>(tmem_offset, a_desc, b_desc, i_desc, enable_d);
          }

          // Signal empty barrier (MMA consumed this stage)
          tcgen05_commit<1>(empty_barrier(stage_idx));

          // On last K block, also signal tmem_full (accumulator ready for epilogue)
          if (k_block == num_k_iters - 1)
            tcgen05_commit<1>(tmem_full_barrier(accum_stage));
        }

        advance_pipeline();
      }
    }

    // Wait for last epilogue to finish before we deallocate barriers
    if (iter >= 0) {
      int last_accum_phase = (iter / NUM_EPILOGUE_STAGES) & 1;
      mbarrier_wait(tmem_empty_barrier(iter % NUM_EPILOGUE_STAGES), last_accum_phase);
    }
  }
  else if (warp_id >= NUM_NON_EPILOGUE_WARPS) {
    // ============ EPILOGUE WARPS (128 threads) ============
    const int epilogue_warp_idx = warp_id - NUM_NON_EPILOGUE_WARPS;
    int iter = -1;
    int tma_store_stage = 0;

    // Bank group: 16 bytes for BF16 = 8 BF16 elements
    constexpr int BANK_GROUP_BYTES = 16;
    constexpr int ELEMS_PER_BANK_GROUP = BANK_GROUP_BYTES / sizeof(nv_bfloat16);  // 8

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      iter++;
      auto [bid_m, bid_n] = compute_bid(this_bid);
      int accum_stage = iter % NUM_EPILOGUE_STAGES;
      int accum_phase = (iter / NUM_EPILOGUE_STAGES) & 1;

      // Wait for MMA to finish accumulating
      mbarrier_wait(tmem_full_barrier(accum_stage), accum_phase);
      tcgen05_fence_after_thread_sync();

      // Iterate over N in STORE_BLOCK_N chunks
      constexpr int NUM_STORES = BLOCK_N / STORE_BLOCK_N;
      for (int s = 0; s < NUM_STORES; s++) {
        // Wait for TMA store pipeline
        if (epilogue_warp_idx == 0)
          tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
        named_barrier_sync(1, NUM_UMMA_STORE_THREADS);

        int m_idx = bid_m * BLOCK_M;
        int n_idx = bid_n * BLOCK_N + s * STORE_BLOCK_N;

        // Load from TMEM, convert to BF16, write to SMEM with swizzle
        // For BF16 output: 8 elems per bank group (16 bytes)
        // STSM atom: STORE_BLOCK_M rows x (SWIZZLE_CD / BANK_GROUP_BYTES) = 128 x 8 bank groups
        // We have 128 threads (4 warps of 32), each lane handles STORE_BLOCK_N / ELEMS_PER_BANK_GROUP = 8 bank groups
        // across lane_id * 8 bank group columns
        for (int i = 0; i < STORE_BLOCK_N / ELEMS_PER_BANK_GROUP; i++) {
          // Swizzle computation
          // bank_group_index = i + lane_id * (SWIZZLE_CD / BANK_GROUP_BYTES)
          //                  = i + lane_id * 8
          // Shortcut: SWIZZLE_CD / BANK_GROUP_BYTES == 8
          int row = i / 8 + lane_id;  // simplified since SWIZZLE_CD/BANK_GROUP_BYTES == 8
          int col = i;
          col ^= row % (SWIZZLE_CD / 16);  // swizzle pattern

          // TMEM address
          uint32_t tmem_addr = accum_stage * BLOCK_N +   // accumulator offset
                               s * STORE_BLOCK_N + i * ELEMS_PER_BANK_GROUP;

          // SMEM destination with swizzle
          uint8_t* smem_base = (uint8_t*)(smem_buf + tma_store_stage * SMEM_CD_SIZE_PER_STAGE);
          uint8_t* smem_dst = smem_base +
                              epilogue_warp_idx * 32 * SWIZZLE_CD +
                              row * (BANK_GROUP_BYTES * 8) + col * BANK_GROUP_BYTES;

          // Load 8 floats from TMEM, convert to BF16, store as 4 x uint32 (8 BF16)
          uint32_t v[8];
          tmem_load_32dp32b_x8(tmem_addr,
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
          tmem_load_fence();

          uint32_t bf16_0 = cvt_f32x2_to_bf16x2(v[0], v[1]);
          uint32_t bf16_1 = cvt_f32x2_to_bf16x2(v[2], v[3]);
          uint32_t bf16_2 = cvt_f32x2_to_bf16x2(v[4], v[5]);
          uint32_t bf16_3 = cvt_f32x2_to_bf16x2(v[6], v[7]);

          st_shared_v4_u32(smem_dst, bf16_0, bf16_1, bf16_2, bf16_3);
        }

        // Signal tmem_empty on last store of last wave
        if (s == NUM_STORES - 1) {
          tcgen05_fence_before_thread_sync();
          // Each epilogue thread arrives at tmem_empty barrier
          mbarrier_arrive((tmem_empty_barrier(accum_stage)));
        }
        __syncwarp();

        // TMA store: sync all epilogue threads, then 1 thread issues TMA store
        tma_store_fence();
        named_barrier_sync(1, NUM_UMMA_STORE_THREADS);
        if (epilogue_warp_idx == 0 && elect_sync()) {
          tma_store_2d(&CD_tmap, cd_smem_addr(tma_store_stage), n_idx, m_idx);
          tma_store_commit();
        }

        tma_store_stage = (tma_store_stage + 1) % NUM_TMA_STORE_STAGES;
      }
    }
  }

  // Deallocate TMEM
  __syncthreads();
  if (warp_id == 0)
    tcgen05_dealloc<1>(0, NUM_TMEM_COLS);
}

// ---- Host-side launcher ----
template <int BLOCK_N, int NUM_SMs>
void matmul_v9_launch(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  CUtensorMap A_tmap, B_tmap, CD_tmap;

  // A is K-major: 3D tensormap (64, M, K/64) : (K*2, 128, 1)
  auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr, uint64_t global_height, uint32_t shared_height) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};  // in bytes
    uint32_t boxDim[rank]          = {64, shared_height, 1};  // 1 k-slice per TMA
    uint32_t elementStrides[rank]  = {1, 1, 1};

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
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
  };

  init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
  init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N);

  // CD tensor map: BF16, 2D (N, M) with swizzle-128B
  // smem box: (STORE_BLOCK_N, STORE_BLOCK_M) where STORE_BLOCK_N = SWIZZLE_CD / sizeof(BF16) = 64
  {
    constexpr uint32_t rank = 2;
    constexpr int STORE_BLOCK_N_CD = SWIZZLE_CD / sizeof(nv_bfloat16);  // 64
    uint64_t globalDim[rank]       = {(uint64_t)N, (uint64_t)M};
    uint64_t globalStrides[rank-1] = {(uint64_t)N * sizeof(nv_bfloat16)};  // row stride in bytes
    uint32_t boxDim[rank]          = {(uint32_t)STORE_BLOCK_N_CD, STORE_BLOCK_M};
    uint32_t elementStrides[rank]  = {1, 1};

    auto err = cuTensorMapEncodeTiled(
      &CD_tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      (void *)C_ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
  }

  int grid = std::min(NUM_SMs, (M / BLOCK_M) * (N / BLOCK_N));

  // Compute shared memory size
  constexpr int SMEM_CD_SIZE = STORE_BLOCK_M * SWIZZLE_CD * NUM_TMA_STORE_STAGES;
  constexpr int AB_DYNAMIC = BLOCK_M * BLOCK_K_BASE * sizeof(nv_bfloat16);  // A per base stage
  constexpr int BB_DYNAMIC = BLOCK_N * BLOCK_K_BASE * sizeof(nv_bfloat16);  // B per base stage
  // With stage merging
  constexpr int NUM_STAGES_PER_MERGE_ = 1;  // will be adjusted at kernel level
  // We need to compute based on MAX stages to allocate enough
  constexpr int smem_capacity = 232448;
  // Calculate max stages that fit
  constexpr int per_base_stage = AB_DYNAMIC + BB_DYNAMIC + 2 * 8;  // A+B + full+empty barrier
  constexpr int fixed_overhead = SMEM_CD_SIZE + 2 * 2 * 8 + 4;    // CD + tmem barriers + tmem ptr
  constexpr int NUM_STAGES_BASE = (smem_capacity - fixed_overhead) / per_base_stage;
  constexpr int smem_size = NUM_STAGES_BASE * per_base_stage + fixed_overhead;

  auto kernel = matmul_v9_kernel<BLOCK_N, NUM_STAGES_BASE, NUM_SMs>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, CD_tmap, M, N, K);
  check_cuda(cudaGetLastError());
}

void matmul_v9(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  matmul_v9_launch<256, 20>(A_ptr, B_ptr, C_ptr, M, N, K);
}
