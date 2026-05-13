#include "common.h"

#include <algorithm>
#include <cuda_bf16.h>

// matmul_v8: Persistent GEMM matching CuTeDSL dense_gemm_persistent.py features:
//   --mma_tiler_mn 128,256 --cluster_shape_mn 1,4 --use_tma_store
// Features:
//   - cluster_shape = (1,4): 4 CTAs clustered along N dimension
//   - CTA_GROUP=1 (no 2CTA MMA), each CTA runs MMA independently
//   - A multicast: all 4 CTAs share same A tile, one CTA loads and
//     multicasts to the other 3 (cluster along N shares same M rows)
//   - TMA store epilogue: TMEM -> RMEM -> SMEM -> GMEM via TMA
//   - Persistent tile scheduling with warp specialization
//   - Epilogue sub-tiling: 128x64 sub-tiles of C (4 per full 128x256 tile)

namespace {

constexpr int NUM_EPILOGUE_WARPS = 4;
constexpr int MMA_WARP_ID = 4;
constexpr int TMA_WARP_ID = 5;
constexpr int NUM_WARPS = 6;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 64;

// Epilogue sub-tile: matches CuTeDSL compute_epilogue_tile_shape
// for cta_tile=(128,256), use_2cta=false, c_dtype=bf16
constexpr int EPI_M = 128;
constexpr int EPI_N = 64;

// TMA multicast load: load from global to shared and multicast to all CTAs
__device__ inline
void tma_3d_gmem2smem_multicast(int dst, const void *tmap_ptr, int x, int y, int z,
                                int mbar_addr, uint16_t mcast_mask) {
	asm volatile(
		"cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
		"[%0], [%1, {%2, %3, %4}], [%5], %6;"
		:: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
		   "r"(mbar_addr), "h"(mcast_mask)
		: "memory");
}

// TMA store: shared memory to global memory (2D)
__device__ inline
void tma_2d_smem2gmem(const void *tmap_ptr, int smem_addr, int x, int y) {
	uint64_t gmem_desc = reinterpret_cast<uint64_t>(tmap_ptr);
	asm volatile(
		"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
		:: "l"(gmem_desc), "r"(smem_addr), "r"(x), "r"(y)
		: "memory");
}

__device__ inline
void tma_store_commit() {
	asm volatile("cp.async.bulk.commit_group;");
}

template <int N>
__device__ inline
void tma_store_wait() {
	asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

__device__ inline
void fence_async_shared() {
	asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Local mbarrier operations using shared::cta (not shared::cluster)
// to correctly target the LOCAL CTA's mbarrier in cluster mode.
__device__ inline
void mbarrier_arrive_expect_tx_local(int mbar_addr, int size) {
	asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
	            :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline
void mbarrier_arrive_local(int mbar_addr) {
	asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

// Non-blocking try_wait: returns 1 if mbar phase completed, 0 if not yet.
// Used for peek-ahead in the MMA pipeline to overlap wait with computation.
__device__ inline
int mbarrier_try_wait_local(int mbar_addr, int phase) {
	int result;
	asm volatile(
		"{\n\t"
		".reg .pred P1;\n\t"
		"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n\t"
		"selp.b32 %0, 1, 0, P1;\n\t"
		"}"
		: "=r"(result) : "r"(mbar_addr), "r"(phase));
	return result;
}

// TMA load using .shared::cta (for local CTA's B load in cluster mode).
// The common.h tma_3d_gmem2smem uses .shared::cluster which in cluster
// mode (without CTA rank bits) targets CTA 0. This version correctly
// targets the issuing CTA's shared memory.
__device__ inline
void tma_3d_gmem2smem_cta(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
	asm volatile(
		"cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
		"[%0], [%1, {%2, %3, %4}], [%5];"
		:: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
		: "memory");
}

// tcgen05.commit with multicast: arrives at all CTAs specified by mask
__device__ inline
void tcgen05_commit_multicast(int mbar_addr, uint16_t cta_mask) {
	asm volatile(
		"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
		:: "r"(mbar_addr), "h"(cta_mask)
		: "memory");
}

// tcgen05.commit only supports .shared::cluster, not .shared::cta.
// Use mapa to convert local CTA mbar address to cluster address.
__device__ inline
int mapa_cluster(int cta_addr, int cta_rank) {
	int cluster_addr;
	asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
	            : "=r"(cluster_addr) : "r"(cta_addr), "r"(cta_rank));
	return cluster_addr;
}

template <int BLOCK_N, int CLUSTER_N, int NUM_STAGES, int NUM_C_STAGES>
__global__
__launch_bounds__(TB_SIZE)
void matmul_v8_kernel_persistent(
	const __grid_constant__ CUtensorMap A_tmap,
	const __grid_constant__ CUtensorMap B_tmap,
	const __grid_constant__ CUtensorMap C_tmap,
	int M, int N, int K
) {
	const int tid = threadIdx.x;
	const int cta_id = warp_uniform(blockIdx.x);
	const int num_ctas = warp_uniform(gridDim.x);
	const int warp_id = warp_uniform(tid / WARP_SIZE);

	const int grid_m = M / BLOCK_M;
	const int grid_n = N / BLOCK_N;
	const int num_iters = K / BLOCK_K;

	int cta_rank;
	asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

	extern __shared__ __align__(1024) char smem_ptr[];
	const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

	// Shared memory layout:
	// [A_size + B_size] * NUM_STAGES  -- A/B pipeline buffers
	// [C_epi_size] * NUM_C_STAGES     -- C epilogue sub-tile buffers
	// tma_mbar[NUM_STAGES]            -- TMA load full barriers
	// mma_mbar[NUM_STAGES]            -- MMA empty barriers (signals TMA warp)
	// mainloop_mbar[2]                -- mainloop done barriers (acc double buffer)
	// epilogue_mbar[2]                -- epilogue done barriers (acc double buffer)
	constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
	constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);
	constexpr int C_epi_size = EPI_M * EPI_N * sizeof(nv_bfloat16);

	const int C_smem_base = smem + (A_size + B_size) * NUM_STAGES;
	const int tma_mbar_addr = C_smem_base + C_epi_size * NUM_C_STAGES;
	const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
	const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
	const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

	// Initialize barriers
	if (warp_id == 0 && elect_sync()) {
		for (int i = 0; i < NUM_STAGES; i++) {
			mbarrier_init(tma_mbar_addr + i * 8, 1);
			mbarrier_init(mma_mbar_addr + i * 8, CLUSTER_N);  // All CTAs multicast-commit
		}
		for (int i = 0; i < 2; i++) {
			mbarrier_init(mainloop_mbar_addr + i * 8, 1);
			mbarrier_init(epilogue_mbar_addr + i * 8, NUM_EPILOGUE_WARPS * WARP_SIZE);
		}
		asm volatile("fence.mbarrier_init.release.cluster;");
	}

	// Cluster sync after barrier init
	asm volatile("barrier.cluster.arrive.relaxed.aligned;");
	asm volatile("barrier.cluster.wait.acquire.aligned;");

	// A multicast mask: all CTAs in cluster receive A
	uint16_t a_mcast_mask = (1 << CLUSTER_N) - 1;

	// Cluster-aware tile scheduling: all CTAs in a cluster share same M tile
	const int cluster_id = cta_id / CLUSTER_N;
	const int num_clusters_active = num_ctas / CLUSTER_N;
	const int grid_n_clusters = grid_n / CLUSTER_N;
	const int num_work_units = grid_m * grid_n_clusters;

	auto compute_tile = [&](int work_id) -> std::tuple<int, int> {
		int tile_m = work_id / grid_n_clusters;
		int tile_n = (work_id % grid_n_clusters) * CLUSTER_N + cta_rank;
		return {tile_m, tile_n};
	};

	// ========================= TMA WARP =========================
	if (warp_id == TMA_WARP_ID) {
		if (elect_sync()) {
			int tma_stage = 0;
			int mma_phase = 1;

			for (int work = cluster_id; work < num_work_units; work += num_clusters_active) {
				auto [tile_m, tile_n] = compute_tile(work);
				const int off_m = tile_m * BLOCK_M;
				const int off_n = tile_n * BLOCK_N;

				for (int iter_k = 0; iter_k < num_iters; iter_k++) {
					const int mbar_addr = tma_mbar_addr + tma_stage * 8;
					const int A_smem_addr = smem + tma_stage * (A_size + B_size);
					const int B_smem_addr = A_smem_addr + A_size;

					// Wait for MMA to finish consuming this stage
					mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);

					// Set expected bytes BEFORE issuing loads (matches CuTeDSL)
					mbarrier_arrive_expect_tx_local(mbar_addr, A_size + B_size);

					// Load A with multicast (only rank 0 issues the load;
					// hardware delivers A to all CTAs and updates their mbarriers)
					if (cta_rank == 0) {
						tma_3d_gmem2smem_multicast(
							A_smem_addr, &A_tmap, 0, off_m, iter_k,
							mbar_addr, a_mcast_mask);
					}
					// Each CTA loads its own B tile (.shared::cta for local CTA)
					tma_3d_gmem2smem_cta(B_smem_addr, &B_tmap, 0, off_n, iter_k, mbar_addr);

					tma_stage = (tma_stage + 1) % NUM_STAGES;
					if (tma_stage == 0) {
						mma_phase ^= 1;
					}
				}
			}
		}
	}
	// ========================= MMA WARP =========================
	else if (warp_id == MMA_WARP_ID) {
		// Allocate tensor memory (CTA_GROUP=1)
		tcgen05_alloc<1>(epilogue_mbar_addr + 8 * 2, BLOCK_N * 2);

		// MMA instruction descriptor
		constexpr uint32_t MMA_M = BLOCK_M;
		constexpr uint32_t MMA_N = BLOCK_N;
		constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
		                          | (1U << 7U)    // atype=BF16
		                          | (1U << 10U)   // btype=BF16
		                          | (MMA_N >> 3U << 17U)
		                          | (MMA_M >> 4U << 24U);

		// Shared memory descriptors: 128-byte swizzling
		constexpr uint64_t AB_desc = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

		if (elect_sync()) {
			int tma_stage = 0;
			int tma_phase = 0;
			int acc_stage = 0;
			int epilogue_phase = 1;

			for (int work = cluster_id; work < num_work_units; work += num_clusters_active) {
				// Wait for epilogue to finish with this acc buffer
				mbarrier_wait(epilogue_mbar_addr + acc_stage * 8, epilogue_phase);

				for (int iter_k = 0; iter_k < num_iters; iter_k++) {
					const int A_smem_addr = smem + tma_stage * (A_size + B_size);
					const int B_smem_addr = A_smem_addr + A_size;
					const int tmem = acc_stage * BLOCK_N;

					uint64_t a_desc = AB_desc | (A_smem_addr >> 4);
					uint64_t b_desc = AB_desc | (B_smem_addr >> 4);

					// Wait for TMA load to complete
					mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
					asm volatile("tcgen05.fence::after_thread_sync;");

					// Issue MMA (CTA_GROUP=1)
					tcgen05_mma_f16<1>(tmem, a_desc, b_desc, i_desc, iter_k);
					#pragma unroll
					for (int k = 1; k < BLOCK_K / 16; k++) {
						a_desc += (32 >> 4);
						b_desc += (32 >> 4);
						tcgen05_mma_f16<1>(tmem, a_desc, b_desc, i_desc, 1);
					}

					// Signal ALL CTAs that MMA consumed this TMA stage
					tcgen05_commit_multicast(mma_mbar_addr + tma_stage * 8, a_mcast_mask);

					tma_stage = (tma_stage + 1) % NUM_STAGES;
					if (tma_stage == 0) {
						tma_phase ^= 1;
					}
				}

				// Signal mainloop done for this acc buffer (mapa for correct cluster address)
				tcgen05_commit<1>(mapa_cluster(mainloop_mbar_addr + acc_stage * 8, cta_rank));

				acc_stage = (acc_stage + 1) % 2;
				if (acc_stage == 0) {
					epilogue_phase ^= 1;
				}
			}
		}
	}
	// ========================= EPILOGUE WARPS (0-3): TMA STORE =========================
	else {
		int acc_stage = 0;
		int mainloop_phase = 0;
		int c_store_count = 0;

		auto epilogue_sync = []() {
			asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(NUM_EPILOGUE_WARPS * WARP_SIZE) : "memory");
		};

		constexpr int NUM_SUBTILES = BLOCK_N / EPI_N;  // 256/64 = 4

		for (int work = cluster_id; work < num_work_units; work += num_clusters_active) {
			auto [tile_m, tile_n] = compute_tile(work);

			// Wait for mainloop (MMA) to finish
			if (warp_id == 0) {
				mbarrier_wait(mainloop_mbar_addr + acc_stage * 8, mainloop_phase);
			}
			epilogue_sync();
			asm volatile("tcgen05.fence::after_thread_sync;");

			// Process epilogue in sub-tiles of EPI_M x EPI_N
			for (int subtile = 0; subtile < NUM_SUBTILES; subtile++) {
				const int c_buf = c_store_count % NUM_C_STAGES;
				const int c_smem_buf = C_smem_base + c_buf * C_epi_size;

				// Wait for previous TMA store using this buffer to complete
				// BEFORE overwriting SMEM (race condition if done after!)
				if (c_store_count >= NUM_C_STAGES) {
					if (warp_id == 0 && elect_sync()) {
						tma_store_wait<NUM_C_STAGES - 1>();
					}
					epilogue_sync();
				}

				// Load from TMEM and store to SMEM
				constexpr int WIDTH = 16;
				for (int w = 0; w < EPI_N / WIDTH; w++) {
					const int t_col = acc_stage * BLOCK_N + subtile * EPI_N + w * WIDTH;
					const int t_addr = (warp_id * 32 << 16) + t_col;

					const int c_smem_thread = c_smem_buf
						+ tid * EPI_N * (int)sizeof(nv_bfloat16)
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

				// Fence SMEM writes visible to TMA engine
				fence_async_shared();
				epilogue_sync();

				// TMA store this sub-tile from SMEM to GMEM
				if (warp_id == 0 && elect_sync()) {
					const int g_col = tile_n * BLOCK_N + subtile * EPI_N;
					const int g_row = tile_m * BLOCK_M;
					tma_2d_smem2gmem(&C_tmap, c_smem_buf, g_col, g_row);
					tma_store_commit();
				}
				c_store_count++;
				epilogue_sync();
			}

			// Signal epilogue done for this acc buffer
			mbarrier_arrive_local(epilogue_mbar_addr + acc_stage * 8);

			acc_stage = (acc_stage + 1) % 2;
			if (acc_stage == 0) {
				mainloop_phase ^= 1;
			}
		}

		// Wait for all TMA stores to complete
		if (warp_id == 0 && elect_sync()) {
			tma_store_wait<0>();
		}
	}

	// Cluster barrier to ensure all CTAs' multicast commits have completed
	asm volatile("barrier.cluster.arrive.relaxed.aligned;");
	asm volatile("barrier.cluster.wait.acquire.aligned;");

	// All warp groups sync before TMEM dealloc
	__syncthreads();
	if (warp_id == 0) {
		tcgen05_dealloc<1>(0, BLOCK_N * 2);
	}
}

inline int get_sm_count() {
	int device = 0;
	check_cuda(cudaGetDevice(&device));
	int sm_count = 0;
	check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
	return sm_count;
}

template <int BLOCK_N, int CLUSTER_N>
void matmul_v8_launch(
	const nv_bfloat16 *A_ptr,
	const nv_bfloat16 *B_ptr,
	nv_bfloat16 *C_ptr,
	int M, int N, int K
) {
	CUtensorMap A_tmap, B_tmap, C_tmap;

	// A/B tensor maps: 3D with 128-byte swizzle
	auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr,
	                        uint64_t global_height, uint32_t shared_height) {
		constexpr uint32_t rank = 3;
		uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
		uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};
		uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BLOCK_K / 64};
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
			CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
			CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
		);
		check_cu(err);
	};

	// C tensor map: 2D, no swizzle, for TMA store
	// C is row-major: M rows x N cols of bf16
	// TMA sub-tile: EPI_M rows x EPI_N cols
	{
		constexpr uint32_t rank = 2;
		uint64_t globalDim[rank]       = {(uint64_t)N, (uint64_t)M};
		uint64_t globalStrides[rank-1] = {(uint64_t)N * sizeof(nv_bfloat16)};
		uint32_t boxDim[rank]          = {(uint32_t)EPI_N, (uint32_t)EPI_M};
		uint32_t elementStrides[rank]  = {1, 1};

		auto err = cuTensorMapEncodeTiled(
			&C_tmap,
			CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
			rank,
			(void *)C_ptr,
			globalDim,
			globalStrides,
			boxDim,
			elementStrides,
			CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
			CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
			CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
			CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
		);
		check_cu(err);
	}

	init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
	init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N);

	const int num_work_units = (M / BLOCK_M) * (N / BLOCK_N / CLUSTER_N);
	const int sm_count = get_sm_count();
	const int max_active_clusters = sm_count / CLUSTER_N;
	const int num_clusters = std::min(max_active_clusters, num_work_units);
	const int grid = num_clusters * CLUSTER_N;

	// Compute SMEM size
	constexpr int AB_per_stage = (BLOCK_M + BLOCK_N) * BLOCK_K * (int)sizeof(nv_bfloat16);
	constexpr int C_epi_subtile = EPI_M * EPI_N * (int)sizeof(nv_bfloat16);  // 16KB
	constexpr int NUM_C_STAGES = 2;
	constexpr int sm100_smem = 227 * 1024;
	constexpr int mbar_overhead = 1024;  // generous margin for all barriers
	constexpr int available = sm100_smem - NUM_C_STAGES * C_epi_subtile - mbar_overhead;
	constexpr int NUM_STAGES = available / AB_per_stage;
	constexpr int smem_size = NUM_STAGES * AB_per_stage
	                        + NUM_C_STAGES * C_epi_subtile
	                        + mbar_overhead;

	auto this_kernel = matmul_v8_kernel_persistent<BLOCK_N, CLUSTER_N, NUM_STAGES, NUM_C_STAGES>;
	check_cuda(cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
	check_cuda(cudaFuncSetAttribute(this_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

	// Use cudaLaunchKernelEx for proper cluster launch
	cudaLaunchConfig_t launch_config = {};
	launch_config.gridDim = dim3(grid, 1, 1);
	launch_config.blockDim = dim3(TB_SIZE, 1, 1);
	launch_config.dynamicSmemBytes = smem_size;

	cudaLaunchAttribute cluster_attr = {};
	cluster_attr.id = cudaLaunchAttributeClusterDimension;
	cluster_attr.val.clusterDim.x = CLUSTER_N;
	cluster_attr.val.clusterDim.y = 1;
	cluster_attr.val.clusterDim.z = 1;
	launch_config.attrs = &cluster_attr;
	launch_config.numAttrs = 1;

	check_cuda(cudaLaunchKernelEx(&launch_config, this_kernel, A_tmap, B_tmap, C_tmap, M, N, K));
	check_cuda(cudaGetLastError());
}

}  // namespace

void matmul_v8(
	const nv_bfloat16 *A_ptr,
	const nv_bfloat16 *B_ptr,
	nv_bfloat16 *C_ptr,
	int M, int N, int K
) {
	matmul_v8_launch<256, 1>(A_ptr, B_ptr, C_ptr, M, N, K);
}
