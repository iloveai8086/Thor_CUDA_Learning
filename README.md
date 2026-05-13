# Thor CUDA Learning

这是一个 CUDA GEMM 学习和实验仓库，主要围绕 NVIDIA Blackwell/Thor 平台上的矩阵乘法 kernel 做迭代优化。

当前重点是 BF16 GEMM：从基础 tiled matmul 开始，逐步加入 TMA、cluster、CTA swizzle、Hilbert 调度、L1 no-alloc、TMA store、prefetch、tensor memory pipeline 等优化，并和 `torch.mm`、CuTe/CUTLASS、DeepGEMM、cuBLAS/cuBLASLt 做对照。

完整 Nsight / profiling 原始记录见：[bf16/profile.md](bf16/profile.md)。

## 排名

以下结果来自当前记录中的 `4096 x 4096 x 4096` GEMM profiling，按 median latency 从低到高排列。详细原始输出在 [bf16/profile.md](bf16/profile.md)。

| Rank | Kernel | Median | Avg | Min |
| ---: | --- | ---: | ---: | ---: |
| 1 | `nvfp4_cublas` | 0.232 ms | 0.251 ms | 0.222 ms |
| 2 | `fp8_cublas` | 0.476 ms | 0.480 ms | 0.454 ms |
| 3 | `deepgemm_fp8` | 0.592 ms | 0.621 ms | 0.531 ms |
| 4 | `int8_cublas` | 0.616 ms | 0.623 ms | 0.566 ms |
| 5 | `matmul_v7a_hilbert_l1noalloc_tma_store_prefetch` | 0.991 ms | 0.999 ms | 0.810 ms |
| 6 | `matmul_v7a_hilbert_l1noalloc_tma_store` | 1.000 ms | 1.109 ms | 0.824 ms |
| 7 | `matmul_v12_hilbert_l1noalloc` | 1.005 ms | 1.102 ms | 0.825 ms |
| 8 | `matmul_v7a_hilbert_l1noalloc` | 1.006 ms | 1.075 ms | 0.821 ms |
| 9 | `matmul_v12_clc_hilbert` | 1.017 ms | 1.065 ms | 0.829 ms |
| 10 | `torch.mm_bf16` | 1.026 ms | 1.111 ms | 0.829 ms |
| 11 | `matmul_v12_hilbert` | 1.027 ms | 1.117 ms | 0.819 ms |
| 12 | `matmul_v7a_hilbert` | 1.038 ms | 1.087 ms | 0.845 ms |
| 13 | `matmul_v11_2` | 1.051 ms | 1.089 ms | 0.879 ms |
| 14 | `matmul_v6_2_hilbert` | 1.055 ms | 1.103 ms | 0.838 ms |
| 15 | `matmul_v7a_g6` | 1.071 ms | 1.115 ms | 0.926 ms |
| 16 | `matmul_v6_2_g6` | 1.098 ms | 1.181 ms | 0.916 ms |
| 17 | `matmul_v12_l1noalloc` | 1.113 ms | 1.106 ms | 0.898 ms |
| 18 | `matmul_v12` | 1.115 ms | 1.141 ms | 0.902 ms |
| 19 | `matmul_v9` | 1.201 ms | 1.245 ms | 1.156 ms |
| 20 | `cute_fp16_gemm_4_aot` | 1.211 ms | 1.322 ms | 0.982 ms |
| 21 | `matmul_v6` | 1.454 ms | 1.454 ms | 1.266 ms |
| 22 | `matmul_v7b` | 1.461 ms | 1.495 ms | 1.395 ms |
| 23 | `matmul_v7a` | 1.462 ms | 1.459 ms | 1.211 ms |
| 24 | `matmul_v7c` | 1.527 ms | 1.538 ms | 1.440 ms |
| 25 | `matmul_v8` | 1.552 ms | 1.574 ms | 1.478 ms |
| 26 | `matmul_v4_2` | 2.447 ms | 2.496 ms | 2.411 ms |
| 27 | `matmul_v4` | 2.453 ms | 2.504 ms | 2.407 ms |
| 28 | `matmul_v5` | 2.698 ms | 2.774 ms | 2.524 ms |
| 29 | `matmul_v5_2` | 2.754 ms | 2.796 ms | 2.486 ms |
| 30 | `matmul_v3_2` | 2.824 ms | 2.905 ms | 2.668 ms |
| 31 | `matmul_v3` | 2.854 ms | 2.940 ms | 2.674 ms |
| 32 | `matmul_v2b` | 2.924 ms | 2.949 ms | 2.618 ms |
| 33 | `matmul_v2a` | 3.073 ms | 3.097 ms | 2.773 ms |
| 34 | `matmul_v3_cutedsl` | 3.437 ms | 3.510 ms | 2.989 ms |
| 35 | `matmul_v1b` | 6.976 ms | 7.037 ms | 6.388 ms |
| 36 | `matmul_v1a` | 8.368 ms | 8.467 ms | 7.543 ms |

## 目录

```text
bf16/
  matmul.cpp        # PyTorch custom op registration
  matmul_v*.cu      # BF16 GEMM kernel iterations
  bench_flops.py    # correctness + benchmark driver
  profiler.h        # device-side profiler
  profile_utils.py  # profile trace helpers
  profile.md        # full profiling notes
  DeepGEMM/         # DeepGEMM reference

cublas/
  fp8_cublas.cu
  i8_cublas.cu
  nvfp4_cublaslt.cu
```

## 运行

```shell
cd bf16
python bench_flops.py --shape 4096,4096,4096 --kernel matmul_v7a_hilbert_l1noalloc_tma_store_prefetch --no-verbose-build
```

对比多个 kernel：

```shell
python bench_flops.py --kernel matmul_v6_2_hilbert,matmul_v7a_hilbert_l1noalloc_tma_store_prefetch,matmul_v12_clc_hilbert --no-verbose-build
```

## 备注

- **建议使用nsight逐kernel评估性能**

## TODO
* 更多性能优化
* FP8
* MXFP8
* NVFP4
* CuTeDSL
* ...
