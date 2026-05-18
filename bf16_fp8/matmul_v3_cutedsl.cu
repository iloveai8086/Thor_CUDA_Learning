#include "common.h"

#include <cuda_bf16.h>

namespace {

constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;
constexpr int TMEM_COLS = 512;
constexpr int SUBTILE_CNT = 4;

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__
__launch_bounds__(TB_SIZE)
void vectorized_kernel(
	const __grid_constant__ CUtensorMap A_tmap,
	const __grid_constant__ CUtensorMap B_tmap,
	nv_bfloat16 *C_ptr,
	int M, int N, int K
) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	const int warp_id = tid / WARP_SIZE;
	const int grid_n = N / BLOCK_N;
	const int bid_m = bid / grid_n;
	const int bid_n = bid % grid_n;

	const int off_m = bid_m * BLOCK_M;
	const int off_n = bid_n * BLOCK_N;

	extern __shared__ __align__(1024) char smem_ptr[];
	const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
	constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
	constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);
	constexpr int STAGE_SIZE = A_size + B_size;
	constexpr int PREFETCH_STAGES = NUM_STAGES - 2;

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ uint64_t tma_mbars[NUM_STAGES];
	__shared__ uint64_t mma_mbars[1];
	__shared__ int tmem_addr[1];
	const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
	const int mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));

	if (warp_id == 0 && elect_sync()) {
		for (int i = 0; i < NUM_STAGES; i++)
			mbarrier_init(tma_mbar_addr + i * 8, 1);
		mbarrier_init(mma_mbar_addr, 1);
		asm volatile("fence.mbarrier_init.release.cluster;");
	}
	else if (warp_id == 1) {
		const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
		tcgen05_alloc(addr, TMEM_COLS);
	}

	__syncthreads();
	const int taddr = tmem_addr[0];

	int tma_phase = 0;

	constexpr uint32_t i_desc = (1U << 4U)
														| (1U << 7U)
														| (1U << 10U)
														| ((uint32_t)BLOCK_N >> 3U << 17U)
														| ((uint32_t)BLOCK_M >> 4U << 24U);

	auto load = [&](int iter_k) {
		if (warp_id == 0 && elect_sync()) {
			const int stage_id = iter_k % NUM_STAGES;
			const int mbar_addr = tma_mbar_addr + stage_id * 8;
			const int A_smem = smem + stage_id * STAGE_SIZE;
			const int B_smem = A_smem + A_size;
			const int off_k = iter_k * BLOCK_K;

			tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
			tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 64, mbar_addr);
			mbarrier_arrive_expect_tx(mbar_addr, STAGE_SIZE);
		}
	};

	auto compute = [&](int iter_k) {
		const int stage_id = iter_k % NUM_STAGES;
		const int mbar_addr = tma_mbar_addr + stage_id * 8;
		mbarrier_wait(mbar_addr, tma_phase);
		asm volatile("tcgen05.fence::after_thread_sync;");

		const int A_smem = smem + stage_id * STAGE_SIZE;
		const int B_smem = A_smem + A_size;

		if (stage_id == NUM_STAGES - 1)
			tma_phase ^= 1;

		if (warp_id == 0 && elect_sync()) {
			auto make_desc = [](int addr) -> uint64_t {
				const int SBO = 8 * 128;
				return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
			};

			tcgen05_mma_f16(taddr, make_desc(A_smem), make_desc(B_smem), i_desc, iter_k);
			for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
				uint64_t a_desc = make_desc(A_smem + k2 * 32);
				uint64_t b_desc = make_desc(B_smem + k2 * 32);
				tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
			}

			for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
				for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
					uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
					uint64_t b_desc = make_desc(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
					tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
				}
			}
		}
	};

	const int num_iters = K / BLOCK_K;
	const int initial_prefetch = num_iters < PREFETCH_STAGES ? num_iters : PREFETCH_STAGES;
	const int drain_start = num_iters > PREFETCH_STAGES ? num_iters - PREFETCH_STAGES : 0;

	#pragma unroll
	for (int i = 0; i < initial_prefetch; i++)
		load(i);

	#pragma unroll
	for (int iter_k = 0; iter_k + PREFETCH_STAGES < num_iters; iter_k++) {
		load(iter_k + PREFETCH_STAGES);
		compute(iter_k);
	}

	for (int iter_k = drain_start; iter_k < num_iters; iter_k++) {
		compute(iter_k);
	}

	if (warp_id == 0 && elect_sync())
		tcgen05_commit(mma_mbar_addr);
	mbarrier_wait(mma_mbar_addr, 0);

	asm volatile("tcgen05.fence::after_thread_sync;");

	constexpr int SUBTILE_N = BLOCK_N / SUBTILE_CNT;
	constexpr int EPILOGUE_WIDTH = 16;
	const int g_row = off_m + tid;
	#pragma unroll
	for (int subtile = 0; subtile < SUBTILE_CNT; subtile++) {
		#pragma unroll
		for (int n = 0; n < SUBTILE_N / EPILOGUE_WIDTH; n++) {
			const int col = subtile * SUBTILE_N + n * EPILOGUE_WIDTH;
			const int addr = taddr + ((warp_id * 32) << 16) + col;
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
				:: "l"(C_ptr + g_row * N + off_n + col), "r"(addr)
			);
		}
	}

	__syncthreads();
	if (warp_id == 0)
		tcgen05_dealloc(taddr, TMEM_COLS);
}

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES>
void matmul_v3_cutedsl_launch(
	const nv_bfloat16 *A_ptr,
	const nv_bfloat16 *B_ptr,
				nv_bfloat16 *C_ptr,
	int M, int N, int K
) {
	TORCH_CHECK(M % BLOCK_M == 0, "matmul_v3_cutedsl requires M to be a multiple of ", BLOCK_M);
	TORCH_CHECK(N % BLOCK_N == 0, "matmul_v3_cutedsl requires N to be a multiple of ", BLOCK_N);
	TORCH_CHECK(K % BLOCK_K == 0, "matmul_v3_cutedsl requires K to be a multiple of ", BLOCK_K);
	TORCH_CHECK(reinterpret_cast<uintptr_t>(A_ptr) % 32 == 0, "matmul_v3_cutedsl requires A to be 32-byte aligned");
	TORCH_CHECK(reinterpret_cast<uintptr_t>(B_ptr) % 32 == 0, "matmul_v3_cutedsl requires B to be 32-byte aligned");
	TORCH_CHECK(reinterpret_cast<uintptr_t>(C_ptr) % 32 == 0, "matmul_v3_cutedsl requires C to be 32-byte aligned");

	CUtensorMap A_tmap, B_tmap;
	auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr, uint64_t global_height, uint32_t shared_height) {
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

	init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
	init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N);

	const int grid = (M / BLOCK_M) * (N / BLOCK_N);
	const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * NUM_STAGES * sizeof(nv_bfloat16);

	auto this_kernel = vectorized_kernel<BLOCK_N, BLOCK_K, NUM_STAGES>;
	if (smem_size > 48'000)
		check_cuda(cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K);
	check_cuda(cudaGetLastError());
}

}  // namespace

void matmul_v3_cutedsl(
	const nv_bfloat16 *A_ptr,
	const nv_bfloat16 *B_ptr,
				nv_bfloat16 *C_ptr,
	int M, int N, int K
) {
	matmul_v3_cutedsl_launch<256, 64, 4>(A_ptr, B_ptr, C_ptr, M, N, K);
}
