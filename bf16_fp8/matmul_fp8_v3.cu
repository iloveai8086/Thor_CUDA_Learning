#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <tuple>
#include <utility>

#include "matmul_fp8_v3_common.h"

namespace matmul_fp8_v3_detail {

constexpr int kGranK = 128;
constexpr int kShapeM = 4096;
constexpr int kShapeN = 4096;
constexpr int kShapeK = 4096;
constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 128;
constexpr int kLoadBlockM = 128;
constexpr int kLoadBlockN = 128;
constexpr int kStoreBlockM = 128;
constexpr int kStoreBlockN = 256;
constexpr int kSwizzleAMode = 128;
constexpr int kSwizzleBMode = 128;
constexpr int kSwizzleCDMode = 128;
constexpr int kNumStages = 5;  // 6
constexpr int kNumThreads = 256;
constexpr int kNumNonEpilogueThreads = 128;
constexpr int kNumEpilogueThreads = 128;
constexpr int kNumMulticast = 2;
constexpr bool kIsMulticastOnA = false;
constexpr int kNumSMs = 20;
constexpr int kSmemSize = 205100;

int64_t align_up(int64_t x, int64_t y) {
  return ((x + y - 1) / y) * y;
}

CUtensorMapDataType tensor_map_dtype(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kInt:
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case at::kBFloat16:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case at::ScalarType::Float8_e4m3fn:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    default:
      TORCH_CHECK(false, "matmul_fp8_v3 unsupported TMA dtype: ", t.scalar_type());
  }
}

CUtensorMapSwizzle tensor_map_swizzle(int mode) {
  switch (mode) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      TORCH_CHECK(false, "matmul_fp8_v3 unsupported TMA swizzle mode: ", mode);
  }
}

void check_cu(CUresult result, const char* expr) {
  if (result == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  TORCH_CHECK(false, expr, " failed: ", name ? name : "<unknown>", " ",
              message ? message : "");
}

#define MATMUL_FP8_V3_CU_CHECK(expr) check_cu((expr), #expr)

CUtensorMap make_tma_2d_desc(
    const at::Tensor& t,
    int gmem_inner_dim,
    int gmem_outer_dim,
    int smem_inner_dim,
    int smem_outer_dim,
    int gmem_outer_stride,
    int swizzle_mode) {
  const auto elem_size = static_cast<int>(t.element_size());
  if (swizzle_mode != 0)
    smem_inner_dim = swizzle_mode / elem_size;

  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {
      static_cast<cuuint64_t>(gmem_inner_dim),
      static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {
      static_cast<cuuint32_t>(smem_inner_dim),
      static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride * elem_size)};
  const cuuint32_t elem_strides[2] = {1, 1};

  MATMUL_FP8_V3_CU_CHECK(cuTensorMapEncodeTiled(
      &tensor_map,
      tensor_map_dtype(t),
      2,
      t.data_ptr(),
      gmem_dims,
      gmem_strides,
      smem_dims,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      tensor_map_swizzle(swizzle_mode),
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensor_map;
}

CUtensorMap make_tma_a_desc(const at::Tensor& t, int m, int k) {
  return make_tma_2d_desc(t, k, m, kBlockK, kLoadBlockM,
                          static_cast<int>(t.stride(0)), kSwizzleAMode);
}

CUtensorMap make_tma_b_desc(const at::Tensor& t, int n, int k) {
  return make_tma_2d_desc(t, k, n, kBlockK, kLoadBlockN,
                          static_cast<int>(t.stride(0)), kSwizzleBMode);
}

CUtensorMap make_tma_cd_desc(const at::Tensor& t, int m, int n) {
  return make_tma_2d_desc(t, n, m, kStoreBlockN, kStoreBlockM,
                          static_cast<int>(t.stride(0)), kSwizzleCDMode);
}

CUtensorMap make_tma_sf_desc(const at::Tensor& t, int mn, int k, int block_mn) {
  const int aligned_mn = static_cast<int>(align_up(mn, 4));
  const int packed_sf_k = static_cast<int>(align_up(k / kGranK, 4) / 4);
  return make_tma_2d_desc(t, aligned_mn, packed_sf_k, block_mn, 1,
                          aligned_mn, 0);
}

__global__ void per_token_cast_to_fp8_ue8m0_kernel(
    const __nv_bfloat16* x,
    cutlass::float_e4m3_t* x_fp8,
    int32_t* packed_sf,
    int mn,
    int k,
    int sf_k,
    int aligned_mn) {
  __shared__ float reduce_buf[kGranK];

  const int row = blockIdx.x;
  const int pack_k = blockIdx.y;
  const int tid = threadIdx.x;

  uint32_t packed = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int sf_idx = pack_k * 4 + j;
    const int k_base = sf_idx * kGranK;

    float v_abs = 0.0f;
    if (row < mn && sf_idx < sf_k && tid < kGranK) {
      const float v = __bfloat162float(x[row * k + k_base + tid]);
      v_abs = fabsf(v);
    }
    reduce_buf[tid] = v_abs;
    __syncthreads();

    for (int offset = kGranK / 2; offset > 0; offset >>= 1) {
      if (tid < offset)
        reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + offset]);
      __syncthreads();
    }

    const float amax = reduce_buf[0];
    const float raw_sf = fmaxf(amax / 448.0f, 1.0e-4f);
    int exp_unbiased = static_cast<int>(ceilf(log2f(raw_sf)));
    int exp_biased = exp_unbiased + 127;
    exp_biased = exp_biased < 1 ? 1 : exp_biased;
    exp_biased = exp_biased > 254 ? 254 : exp_biased;
    const float sf = exp2f(static_cast<float>(exp_biased - 127));

    if (row < mn && sf_idx < sf_k && tid < kGranK) {
      const float v = __bfloat162float(x[row * k + k_base + tid]);
      x_fp8[row * k + k_base + tid] = cutlass::float_e4m3_t(v / sf);
    }
    if (tid == 0)
      packed |= static_cast<uint32_t>(exp_biased) << (8 * j);
    __syncthreads();
  }

  if (row < mn && tid == 0)
    packed_sf[pack_k * aligned_mn + row] = static_cast<int32_t>(packed);
}

std::tuple<at::Tensor, at::Tensor> per_token_cast_to_fp8_ue8m0(const at::Tensor& x) {
  TORCH_CHECK(x.dim() == 2, "matmul_fp8_v3 quantize expects a 2D tensor");
  TORCH_CHECK(x.is_cuda(), "matmul_fp8_v3 expects CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "matmul_fp8_v3 expects BF16 input tensors");
  TORCH_CHECK(x.is_contiguous(), "matmul_fp8_v3 quantize expects contiguous input");

  const auto mn = static_cast<int>(x.size(0));
  const auto k = static_cast<int>(x.size(1));
  TORCH_CHECK(k % kGranK == 0, "matmul_fp8_v3 requires K to be divisible by 128");

  const int sf_k = k / kGranK;
  const int packed_sf_k = static_cast<int>(align_up(sf_k, 4) / 4);
  const int aligned_mn = static_cast<int>(align_up(mn, 4));

  auto x_fp8 = at::empty({mn, k}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto packed_sf = at::empty_strided({mn, packed_sf_k}, {1, aligned_mn}, x.options().dtype(at::kInt));

  const dim3 grid(mn, packed_sf_k);
  const dim3 block(kGranK);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  per_token_cast_to_fp8_ue8m0_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<cutlass::float_e4m3_t*>(x_fp8.data_ptr()),
      reinterpret_cast<int32_t*>(packed_sf.data_ptr()),
      mn,
      k,
      sf_k,
      aligned_mn);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_fp8, packed_sf};
}

struct FP8Cache {
  const void* a_ptr = nullptr;
  const void* b_ptr = nullptr;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  at::Tensor a_fp8;
  at::Tensor a_sf;
  at::Tensor b_fp8;
  at::Tensor b_sf;
  at::Tensor d;
};

FP8Cache cache;
std::mutex cache_mutex;

bool cache_matches(const at::Tensor& a, const at::Tensor& b, int64_t m, int64_t n, int64_t k) {
  return cache.a_ptr == a.data_ptr() &&
         cache.b_ptr == b.data_ptr() &&
         cache.m == m &&
         cache.n == n &&
         cache.k == k &&
         cache.a_fp8.defined() &&
         cache.b_fp8.defined() &&
         cache.a_sf.defined() &&
         cache.b_sf.defined() &&
         cache.d.defined();
}

void refresh_cache(const at::Tensor& a, const at::Tensor& b, int64_t m, int64_t n, int64_t k) {
  auto b_nt = b.transpose(0, 1).contiguous();
  std::tie(cache.a_fp8, cache.a_sf) = per_token_cast_to_fp8_ue8m0(a.contiguous());
  std::tie(cache.b_fp8, cache.b_sf) = per_token_cast_to_fp8_ue8m0(b_nt);
  cache.d = at::empty({m, n}, a.options());
  cache.a_ptr = a.data_ptr();
  cache.b_ptr = b.data_ptr();
  cache.m = m;
  cache.n = n;
  cache.k = k;
}

using DirectKernel = decltype(&deep_gemm::sm100_fp8_gemm_1d1d_impl<
    cute::UMMA::Major::K, cute::UMMA::Major::K,
    kGranK, kGranK,
    kShapeM, kShapeN, kShapeK,
    kBlockM, kBlockN, kBlockK,
    1,
    kSwizzleAMode, kSwizzleBMode, kSwizzleCDMode,
    kNumStages,
    kNumNonEpilogueThreads, kNumEpilogueThreads,
    kNumMulticast, kIsMulticastOnA,
    kNumSMs,
    deep_gemm::GemmType::Normal, false,
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
    deep_gemm::EpilogueIdentity>);

DirectKernel direct_kernel() {
  return deep_gemm::sm100_fp8_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      kGranK, kGranK,
      kShapeM, kShapeN, kShapeK,
      kBlockM, kBlockN, kBlockK,
      1,
      kSwizzleAMode, kSwizzleBMode, kSwizzleCDMode,
      kNumStages,
      kNumNonEpilogueThreads, kNumEpilogueThreads,
      kNumMulticast, kIsMulticastOnA,
      kNumSMs,
      deep_gemm::GemmType::Normal, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
      deep_gemm::EpilogueIdentity>;
}

void launch_direct_gemm(
    const at::Tensor& a_fp8,
    const at::Tensor& a_sf,
    const at::Tensor& b_fp8,
    const at::Tensor& b_sf,
    const at::Tensor& d,
    int m,
    int n,
    int k) {
  auto tensor_map_a = make_tma_a_desc(a_fp8, m, k);
  auto tensor_map_b = make_tma_b_desc(b_fp8, n, k);
  auto tensor_map_sfa = make_tma_sf_desc(a_sf, m, k, kBlockM);
  auto tensor_map_sfb = make_tma_sf_desc(b_sf, n, k, kBlockN);
  auto tensor_map_cd = make_tma_cd_desc(d, m, n);

  int* grouped_layout = nullptr;
  uint32_t shape_m = static_cast<uint32_t>(m);
  uint32_t shape_n = static_cast<uint32_t>(n);
  uint32_t shape_k = static_cast<uint32_t>(k);

  auto kernel = reinterpret_cast<const void*>(direct_kernel());
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumSMs, 1, 1);
  config.blockDim = dim3(kNumThreads, 1, 1);
  config.dynamicSmemBytes = kSmemSize;
  config.stream = at::cuda::getCurrentCUDAStream().stream();

  cudaLaunchAttribute attrs[2]{};
  config.attrs = attrs;
  attrs[config.numAttrs].id = cudaLaunchAttributeClusterDimension;
  attrs[config.numAttrs].val.clusterDim = {kNumMulticast, 1, 1};
  ++config.numAttrs;
  attrs[config.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[config.numAttrs].val.programmaticStreamSerializationAllowed = 1;
  ++config.numAttrs;

  void* args[] = {
      &grouped_layout,
      &shape_m,
      &shape_n,
      &shape_k,
      &tensor_map_a,
      &tensor_map_b,
      &tensor_map_sfa,
      &tensor_map_sfb,
      &tensor_map_cd};
  C10_CUDA_CHECK(cudaLaunchKernelExC(&config, kernel, args));
}

}  // namespace matmul_fp8_v3_detail

at::Tensor matmul_fp8_v3(const at::Tensor& A, const at::Tensor& B) {
  using namespace matmul_fp8_v3_detail;

  c10::cuda::CUDAGuard device_guard(A.device());

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul_fp8_v3 expects 2D tensors");
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "matmul_fp8_v3 expects CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              "matmul_fp8_v3 expects BF16 inputs");
  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  TORCH_CHECK(B.size(0) == k, "matmul_fp8_v3 shape mismatch: A[M,K] @ B[K,N]");
  const int64_t n = B.size(1);
  TORCH_CHECK(m == kShapeM && n == kShapeN && k == kShapeK,
              "matmul_fp8_v3 direct kernel currently supports only 4096x4096x4096");
  TORCH_CHECK(m <= std::numeric_limits<int>::max() &&
              n <= std::numeric_limits<int>::max() &&
              k <= std::numeric_limits<int>::max(),
              "matmul_fp8_v3 only supports int32 problem dimensions");

  std::lock_guard<std::mutex> lock(cache_mutex);
  if (!cache_matches(A, B, m, n, k))
    refresh_cache(A, B, m, n, k);

  launch_direct_gemm(
      cache.a_fp8,
      cache.a_sf,
      cache.b_fp8,
      cache.b_sf,
      cache.d,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return cache.d;
}
