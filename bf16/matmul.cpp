#include <torch/library.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>

typedef void MatmulFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);

MatmulFn matmul_v0;
MatmulFn matmul_v1a;
MatmulFn matmul_v1b;
MatmulFn matmul_v2a;
MatmulFn matmul_v2b;
MatmulFn matmul_v3;
MatmulFn matmul_v3_2;
MatmulFn matmul_v3_cutedsl;
MatmulFn matmul_v4;
MatmulFn matmul_v4_2;
MatmulFn matmul_v4_3;
MatmulFn matmul_v5;
MatmulFn matmul_v5_2;
MatmulFn matmul_v6;
MatmulFn matmul_v6_2;
MatmulFn matmul_v6_2_g4;
MatmulFn matmul_v6_2_g5;
MatmulFn matmul_v6_2_g6;
MatmulFn matmul_v6_2_g7;
MatmulFn matmul_v6_2_g8;
MatmulFn matmul_v6_2_g10;
MatmulFn matmul_v6_2_g12;
MatmulFn matmul_v6_2_hilbert;
MatmulFn matmul_v6_4;
MatmulFn matmul_v7a;
MatmulFn matmul_v7b;
MatmulFn matmul_v7c;
MatmulFn matmul_v7a_g6;
MatmulFn matmul_v7b_g6;
MatmulFn matmul_v7c_g6;
MatmulFn matmul_v7a_hilbert;
MatmulFn matmul_v7a_hilbert_l1noalloc;
MatmulFn matmul_v7a_hilbert_l1noalloc_tma_store;
MatmulFn matmul_v7a_hilbert_l1noalloc_tma_store_prefetch;
MatmulFn matmul_v7b_hilbert;
MatmulFn matmul_v7c_hilbert;
MatmulFn matmul_v8;
MatmulFn matmul_v9;
MatmulFn matmul_v10;
MatmulFn matmul_v10_2;
MatmulFn matmul_v11;
MatmulFn matmul_v11_2;
MatmulFn matmul_v12;
MatmulFn matmul_v12_nc;
MatmulFn matmul_v12_l1noalloc;
MatmulFn matmul_v12_hilbert;
MatmulFn matmul_v12_hilbert_l1noalloc;
MatmulFn matmul_v12_clc_hilbert;

template <MatmulFn matmul_fn>
at::Tensor matmul(const at::Tensor& A, const at::Tensor& B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto C = at::empty({M, N}, A.options());
  //auto C = at::zeros({M, N}, A.options());  // for correctness check, use this
  matmul_fn(
    reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
    M, N, K
  );
  return C;
}

typedef void ProfileMatmulFn(
  const nv_bfloat16 *A,
  const nv_bfloat16 *B,
        nv_bfloat16 *C,
  int M, int N, int K,
  int64_t *profiler,
  int num_entries
);

ProfileMatmulFn profile_matmul_v5;
ProfileMatmulFn profile_matmul_v5_2;
ProfileMatmulFn profile_matmul_v6;
ProfileMatmulFn profile_matmul_v6_2;
ProfileMatmulFn profile_matmul_v6_4;
ProfileMatmulFn profile_matmul_v7a;
ProfileMatmulFn profile_matmul_v7b;
ProfileMatmulFn profile_matmul_v7a_hilbert_l1noalloc_tma_store_prefetch;
ProfileMatmulFn profile_matmul_v12;

template <ProfileMatmulFn profile_matmul_fn>
at::Tensor profile_matmul(
  const at::Tensor& A,
  const at::Tensor& B,
        at::Tensor& profiler,
  int64_t num_entries
) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto C = at::empty({M, N}, A.options());
  //auto C = at::zeros({M, N}, A.options());  // for correctness check, use this
  profile_matmul_fn(
    reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
    M, N, K,
    profiler.data_ptr<int64_t>(),
    num_entries
  );
  return C;
}

TORCH_LIBRARY(my_matmul, m) {
  m.def("matmul_v0(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v0>);
  m.def("matmul_v1a(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v1a>);
  m.def("matmul_v1b(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v1b>);
  m.def("matmul_v2a(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v2a>);
  m.def("matmul_v2b(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v2b>);
  m.def("matmul_v3(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v3>);
  m.def("matmul_v3_2(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v3_2>);
  m.def("matmul_v3_cutedsl(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v3_cutedsl>);
  m.def("matmul_v4(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v4>);
  m.def("matmul_v4_2(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v4_2>);
  m.def("matmul_v4_3(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v4_3>);
  m.def("matmul_v5(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v5>);
  m.def("matmul_v5_2(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v5_2>);
  m.def("matmul_v6(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6>);
  m.def("matmul_v6_2(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2>);
  m.def("matmul_v6_2_g4(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_g4>);
  m.def("matmul_v6_2_g5(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_g5>);
  m.def("matmul_v6_2_g6(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_g6>);
  m.def("matmul_v6_2_g7(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_g7>);
  m.def("matmul_v6_2_g8(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_g8>);
  m.def("matmul_v6_2_g10(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_g10>);
  m.def("matmul_v6_2_g12(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_g12>);
  m.def("matmul_v6_2_hilbert(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_2_hilbert>);
  m.def("matmul_v6_4(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v6_4>);
  m.def("matmul_v7a(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7a>);
  m.def("matmul_v7b(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7b>);
  m.def("matmul_v7c(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7c>);
  m.def("matmul_v7a_g6(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7a_g6>);
  m.def("matmul_v7b_g6(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7b_g6>);
  m.def("matmul_v7c_g6(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7c_g6>);
  m.def("matmul_v7a_hilbert(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7a_hilbert>);
  m.def("matmul_v7a_hilbert_l1noalloc(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7a_hilbert_l1noalloc>);
  m.def("matmul_v7a_hilbert_l1noalloc_tma_store(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7a_hilbert_l1noalloc_tma_store>);
  m.def("matmul_v7a_hilbert_l1noalloc_tma_store_prefetch(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7a_hilbert_l1noalloc_tma_store_prefetch>);
  m.def("matmul_v7b_hilbert(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7b_hilbert>);
  m.def("matmul_v7c_hilbert(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7c_hilbert>);
  m.def("matmul_v8(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v8>);
  m.def("matmul_v9(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v9>);
  m.def("matmul_v10(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v10>);
  m.def("matmul_v10_2(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v10_2>);
  m.def("matmul_v11(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v11>);
  m.def("matmul_v11_2(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v11_2>);
  m.def("matmul_v12(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v12>);
  m.def("matmul_v12_nc(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v12_nc>);
  m.def("matmul_v12_l1noalloc(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v12_l1noalloc>);
  m.def("matmul_v12_hilbert(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v12_hilbert>);
  m.def("matmul_v12_hilbert_l1noalloc(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v12_hilbert_l1noalloc>);
  m.def("matmul_v12_clc_hilbert(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v12_clc_hilbert>);

  m.def("profile_matmul_v5(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v5>);
  m.def("profile_matmul_v5_2(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v5_2>);
  m.def("profile_matmul_v6(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v6>);
    m.def("profile_matmul_v6_2(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
      &profile_matmul<profile_matmul_v6_2>);
  m.def("profile_matmul_v6_4(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v6_4>);
  m.def("profile_matmul_v7a(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v7a>);
  m.def("profile_matmul_v7b(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v7b>);
  m.def("profile_matmul_v7a_hilbert_l1noalloc_tma_store_prefetch(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v7a_hilbert_l1noalloc_tma_store_prefetch>);
  m.def("profile_matmul_v12(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor",
        &profile_matmul<profile_matmul_v12>);
}
