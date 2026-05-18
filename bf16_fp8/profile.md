# BF16
```shell
nsys profile --stats=true --force-overwrite=true   -o dump   python bench_flopsv2_single_kernel.py     --kernel torch.mm_bf16     --shape 4096,4096,4096     --warmup 30     --repeat-warmup 0     --iters 1     --repeats 100     --no-tegrastats     --no-verbose-build
```
## matmul_v1a
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)    Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     95.9    1,100,763,232        130  8,467,409.5  8,368,368.0  7,542,688  10,261,056    553,152.3  void matmul_v1_kernel<(int)256, (int)256, (bool)0>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *,…
```
## matmul_v1b
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     95.1      914,763,616        130  7,036,643.2  6,976,000.0  6,387,936  8,425,280    390,703.5  void matmul_v1_kernel<(int)256, (int)256, (bool)1>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *,…
```
## matmul_v2a
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     89.7      402,649,184        130  3,097,301.4  3,073,424.0  2,773,472  3,521,792    145,417.1  void matmul_v2_kernel<(int)256, (int)256, (bool)0>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *,…
```
## matmul_v2b
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     89.2      383,421,152        130  2,949,393.5  2,924,128.0  2,618,176  3,942,016    181,881.9  void matmul_v2_kernel<(int)256, (int)256, (bool)1>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *,…
```
## matmul_v3
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     89.1      382,196,416        130  2,939,972.4  2,853,840.0  2,673,664  3,508,736    196,508.8  void matmul_v3_kernel<(int)256, (int)128, (int)2>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *, …
```
## matmul_v3_2
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     89.0      377,643,552        130  2,904,950.4  2,823,872.0  2,667,872  3,695,168    198,010.2  void matmul_v3_2_kernel<(int)256, (int)128>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *, int, i…
```
## matmul_v3_cutedsl
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     90.7      456,314,208        130  3,510,109.3  3,437,168.0  2,988,896  4,475,392    304,566.6  void <unnamed>::vectorized_kernel<(int)256, (int)64, (int)4>(CUtensorMap_st, CUtensorMap_st, __nv_b…
```
## matmul_v4
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     87.4      325,484,960        130  2,503,730.5  2,452,672.0  2,407,296  3,836,384    147,325.3  void matmul_v4_kernel<(int)128, (int)64, (int)3>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *, i…
```
## matmul_v4_2
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     87.5      324,499,840        130  2,496,152.6  2,446,688.0  2,411,264  3,219,680    114,213.0  void matmul_v4_2_kernel<(int)128, (int)64, (int)3>(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *,…
```
## cute_fp16_gemm_4_aot
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     78.6      173,129,856        131  1,321,602.0  1,211,456.0    982,240  2,265,024    373,843.3  kernel_cutlass_kernel_TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____MMAAtom_ThrID21_ShapeMNK2562…
```
## matmul_v5
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     88.6      360,577,632        130  2,773,674.1  2,698,144.0  2,523,680  3,590,272    189,868.0  void matmul_v5_kernel<(int)256, (int)64, (int)2, (int)7, (bool)0>(CUtensorMap_st, CUtensorMap_st, _…
```
## matmul_v5_2
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     88.6      363,498,144        130  2,796,139.6  2,753,744.0  2,485,824  3,390,592    194,261.3  void matmul_v5_2_kernel<(int)256, (int)64, (int)2, (int)7, (bool)0>(CUtensorMap_st, CUtensorMap_st,…
```
## matmul_v6
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     80.2      189,000,608        130  1,453,850.8  1,454,240.0  1,266,272  1,707,008     95,666.3  void matmul_v6_kernel<(int)256, (int)64, (int)2, (int)7, (int)8, (bool)0, (bool)0, (bool)0, (bool)0…
```
## matmul_v6_2_g6
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     76.7      153,503,744        130  1,180,798.0  1,098,176.0    916,224  2,438,976    358,062.5  void matmul_v6_kernel<(int)256, (int)64, (int)2, (int)7, (int)6, (bool)0, (bool)1, (bool)0, (bool)0…
```

## matmul_v6_2_hilbert
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.5      143,438,816        130  1,103,375.5  1,054,880.0    838,336  2,398,592    276,265.6  void matmul_v6_kernel<(int)256, (int)64, (int)2, (int)7, (int)6, (bool)0, (bool)0, (bool)0, (bool)1…
```
## matmul_v11_2
```shell
Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.2      141,593,952        130  1,089,184.2  1,050,576.0    878,752  2,162,880    195,155.7  void <unnamed>::matmul_v11_2_kernel<(int)256, (int)64, (int)2>(CUtensorMap_st, CUtensorMap_st, __nv…
```
## matmul_v12
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     76.0      148,379,072        130  1,141,377.5  1,114,592.0    901,824  2,255,904    281,320.2  void matmul_v12_kernel<(int)256, (int)64, (int)2, (int)7, (int)6, (bool)0, (bool)0, (bool)1, (bool)…
```
## matmul_v12_l1noalloc
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.5      143,810,208        130  1,106,232.4  1,112,656.0    898,112  2,258,112    231,542.1  void matmul_v12_kernel<(int)256, (int)64, (int)2, (int)7, (int)6, (bool)0, (bool)1, (bool)1, (bool)…
```
## matmul_v12_hilbert
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.7      145,225,344        130  1,117,118.0  1,026,800.0    818,752  2,224,192    338,113.3  void matmul_v12_kernel<(int)256, (int)64, (int)2, (int)7, (int)6, (bool)0, (bool)0, (bool)0, (bool)…
```
## matmul_v12_hilbert_l1noalloc
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.2      143,283,104        130  1,102,177.7  1,004,640.0    824,544  2,235,968    312,647.1  void matmul_v12_kernel<(int)256, (int)64, (int)2, (int)7, (int)6, (bool)0, (bool)1, (bool)0, (bool)…
```
## matmul_v12_clc_hilbert
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     74.8      138,444,704        130  1,064,959.3  1,017,136.0    828,768  2,240,224    263,956.8  void matmul_v12_kernel<(int)256, (int)64, (int)2, (int)7, (int)6, (bool)0, (bool)0, (bool)0, (bool)…
```
## torch.mm_bf16
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.5      144,458,336        130  1,111,218.0  1,025,712.0    829,088  2,315,104    330,428.1  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT
```
## matmul_v7a
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     80.3      189,725,184        130  1,459,424.5  1,461,824.0  1,211,296  1,717,632     94,627.6  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)7, (bool)0, (bool)0>(CUtensorMap_st, CUtensorM…
```
## matmul_v7b
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     80.5      194,352,064        130  1,495,015.9  1,461,488.0  1,394,880  2,694,048    146,387.9  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)7, (bool)0, (bool)0>(CUtensorMap_st, CUtensorM…
```
## matmul_v7c
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     81.0      199,973,760        130  1,538,259.7  1,526,592.0  1,440,096  1,710,368     47,343.0  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)7, (bool)1, (bool)0>(CUtensorMap_st, CUtensorM…
```
## matmul_v7a_g6
```
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.6      144,916,992        130  1,114,746.1  1,070,848.0    926,400  2,285,728    234,799.8  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)7, (int)6, (bool)0, (bool)0, (bool)1, (bool)0>…
```
## matmul_v7a_hilbert
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     75.0      141,274,880        130  1,086,729.8  1,038,016.0    844,512  2,219,072    277,010.4  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)7, (int)6, (bool)0, (bool)0, (bool)0, (bool)1>…
```
## matmul_v7a_hilbert_l1noalloc
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     74.8      139,722,976        130  1,074,792.1  1,006,400.0    820,544  4,379,712    379,158.0  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)7, (int)6, (bool)1, (bool)0, (bool)0, (bool)1>…
```
## matmul_v7a_hilbert_l1noalloc_tma_store
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     74.6      144,175,008        130  1,109,038.5    999,616.0    823,680  2,222,304    339,736.3  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)6, (int)6, (bool)1, (bool)0, (bool)0, (bool)1,…
```
## matmul_v7a_hilbert_l1noalloc_tma_store_prefetch
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     73.7      129,826,880        130    998,668.3    991,392.0    810,272  1,146,784     81,182.4  void matmul_v7_kernel_cutlass<(int)256, (int)2, (int)6, (int)6, (bool)1, (bool)0, (bool)0, (bool)1,…
```
## matmul_v8
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     81.5      204,628,224        130  1,574,063.3  1,551,888.0  1,478,400  1,856,256     69,307.2  void <unnamed>::matmul_v8_kernel_persistent<(int)256, (int)1, (int)4, (int)2>(CUtensorMap_st, CUten…
```
## matmul_v9
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     77.6      161,880,000        130  1,245,230.8  1,200,768.0  1,156,288  2,200,864    163,242.7  void matmul_v9_kernel<(int)256, (int)4, (int)20>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, in…
```
## deepgemm_fp8
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     60.3       81,336,544        131    620,889.6    591,584.0    530,848  1,259,392     95,833.4  void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)…
```
## fp8_cublas
```shell
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                     Name                    
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------------------------
    100.0      105,992,992        221  479,606.3  475,840.0   453,952   540,768     13,585.5  nvjet_qqtst_128x256_128x6_2x1_2cta_v_bz_TNT
```
## int8 cublas
```shell
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
    100.0       74,742,816        120  622,856.8  615,760.0   566,016   720,832     25,789.9  cutlass3x_sm100_tensorop_i256x256x32gemm_s8_s8_s32_s32_s32_256x256x128_0_tnn_align16_2sm_bias_s32_r…
```
## nvfp4_cublas
```shell
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
    100.0       55,414,624        221  250,744.9  232,000.0   222,144  1,240,928    110,589.5  cutlass3x_sm100_bstensorop_s256x256x64gemm_block_scaled_ue4m3xf4_ue4m3xf4_f32_bf16_bf16_256x256x256…
```
**按 Median 排名**
```text
 1. nvfp4_cublas                                      0.2320 ms
 2. fp8_cublas                                        0.4758 ms
 3. deepgemm_fp8                                      0.5916 ms
 4. int8 cublas                                       0.6158 ms
 5. matmul_v7a_hilbert_l1noalloc_tma_store_prefetch   0.9914 ms
 6. matmul_v7a_hilbert_l1noalloc_tma_store            0.9996 ms
 7. matmul_v12_hilbert_l1noalloc                      1.0046 ms
 8. matmul_v7a_hilbert_l1noalloc                      1.0064 ms
 9. matmul_v12_clc_hilbert                            1.0171 ms
10. torch.mm_bf16                                     1.0257 ms
11. matmul_v12_hilbert                                1.0268 ms
12. matmul_v7a_hilbert                                1.0380 ms
13. matmul_v11_2                                      1.0506 ms
14. matmul_v6_2_hilbert                               1.0549 ms
15. matmul_v7a_g6                                     1.0708 ms
16. matmul_v6_2_g6                                    1.0982 ms
17. matmul_v12_l1noalloc                              1.1127 ms
18. matmul_v12                                        1.1146 ms
19. matmul_v9                                         1.2008 ms
20. cute_fp16_gemm_4_aot                              1.2115 ms
21. matmul_v6                                         1.4542 ms
22. matmul_v7b                                        1.4615 ms
23. matmul_v7a                                        1.4618 ms
24. matmul_v7c                                        1.5266 ms
25. matmul_v8                                         1.5519 ms
26. matmul_v4_2                                       2.4467 ms
27. matmul_v4                                         2.4527 ms
28. matmul_v5                                         2.6981 ms
29. matmul_v5_2                                       2.7537 ms
30. matmul_v3_2                                       2.8239 ms
31. matmul_v3                                         2.8538 ms
32. matmul_v2b                                        2.9241 ms
33. matmul_v2a                                        3.0734 ms
34. matmul_v3_cutedsl                                 3.4372 ms
35. matmul_v1b                                        6.9760 ms
36. matmul_v1a                                        8.3684 ms
```

**按 Avg 排名**
```text
 1. nvfp4_cublas                                      0.2507 ms
 2. fp8_cublas                                        0.4796 ms
 3. deepgemm_fp8                                      0.6209 ms
 4. int8 cublas                                       0.6229 ms
 5. matmul_v7a_hilbert_l1noalloc_tma_store_prefetch   0.9987 ms
 6. matmul_v12_clc_hilbert                            1.0650 ms
 7. matmul_v7a_hilbert_l1noalloc                      1.0748 ms
 8. matmul_v7a_hilbert                                1.0867 ms
 9. matmul_v11_2                                      1.0892 ms
10. matmul_v12_hilbert_l1noalloc                      1.1022 ms
11. matmul_v6_2_hilbert                               1.1034 ms
12. matmul_v12_l1noalloc                              1.1062 ms
13. matmul_v7a_hilbert_l1noalloc_tma_store            1.1090 ms
14. torch.mm_bf16                                     1.1112 ms
15. matmul_v7a_g6                                     1.1147 ms
16. matmul_v12_hilbert                                1.1171 ms
17. matmul_v12                                        1.1414 ms
18. matmul_v6_2_g6                                    1.1808 ms
19. matmul_v9                                         1.2452 ms
20. cute_fp16_gemm_4_aot                              1.3216 ms
21. matmul_v6                                         1.4539 ms
22. matmul_v7a                                        1.4594 ms
23. matmul_v7b                                        1.4950 ms
24. matmul_v7c                                        1.5383 ms
25. matmul_v8                                         1.5741 ms
26. matmul_v4_2                                       2.4962 ms
27. matmul_v4                                         2.5037 ms
28. matmul_v5                                         2.7737 ms
29. matmul_v5_2                                       2.7961 ms
30. matmul_v3_2                                       2.9050 ms
31. matmul_v3                                         2.9400 ms
32. matmul_v2b                                        2.9494 ms
33. matmul_v2a                                        3.0973 ms
34. matmul_v3_cutedsl                                 3.5101 ms
35. matmul_v1b                                        7.0366 ms
36. matmul_v1a                                        8.4674 ms
```

# FP8
## matmul_fp8_v4_nocache
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     45.4      158,809,376        130  1,221,610.6  1,064,544.0  1,053,920  1,769,376    242,952.8  matmul_fp8_v4_detail::matmul_fp8_v4_kernel(CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 *, int, in…
     40.3      140,959,872        260    542,153.4    530,784.0    517,376    976,800     62,611.2  matmul_fp8_v4_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)
```
## matmul_fp8_v5
```shell
     49.7      138,155,008        260    531,365.4    531,296.0    515,424    569,056      5,791.3  matmul_fp8_v5_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     33.3       92,539,840        130    711,844.9    708,176.0    703,776    780,608     13,450.8  void matmul_fp8_v5_detail::matmul_fp8_v5_kernel<(int)256, (int)128, (int)2>(CUtensorMap_st, CUtenso…
```
## matmul_fp8_v6(256,128,4)
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     52.1      138,068,448        260    531,032.5    530,752.0    514,432    560,864      6,738.5  matmul_fp8_v6_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     29.8       78,920,448        130    607,080.4    603,200.0    596,864    690,112     13,836.2  void matmul_fp8_v6_detail::matmul_fp8_v6_kernel<(int)256, (int)128, (int)4>(CUtensorMap_st, CUtenso…
```
## matmul_fp8_v7_n256_k128_c2_s7
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     55.1      137,925,952        260    530,484.4    530,816.0    516,352    567,872      6,164.9  matmul_fp8_v7_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     26.1       65,455,968        130    503,507.4    500,640.0    498,656    582,208     10,231.7  void matmul_fp8_v7_detail::matmul_fp8_v7_kernel<(int)256, (int)128, (int)2, (int)7>(CUtensorMap_st,…
```
## matmul_fp8_v8_plain
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     58.4      138,205,632        260    531,560.1    530,976.0    515,744    568,544      6,087.0  matmul_fp8_v8_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     21.5       50,758,592        130    390,450.7    388,816.0    372,928    499,424     10,922.6  void matmul_fp8_v8_detail::matmul_fp8_v8_kernel<(int)256, (int)128, (int)2, (int)7, (int)6, (bool)0…
```
## matmul_fp8_v8_g6
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     54.8      138,964,000        260    534,476.9    527,392.0    516,544  1,301,248     58,494.6  matmul_fp8_v8_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     26.7       67,737,888        130    521,060.7    513,376.0    494,880  1,414,720     80,780.1  void matmul_fp8_v8_detail::matmul_fp8_v8_kernel<(int)256, (int)128, (int)2, (int)7, (int)6, (bool)0…
```
## matmul_fp8_v8_g7
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     54.0      137,157,152        260    527,527.5    526,128.0    517,056    558,976      6,160.5  matmul_fp8_v8_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     27.2       69,064,256        130    531,263.5    527,216.0    515,072    614,528     14,450.8  void matmul_fp8_v8_detail::matmul_fp8_v8_kernel<(int)256, (int)128, (int)2, (int)7, (int)7, (bool)0…
```
## matmul_fp8_v8_g8
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     55.7      137,555,840        260    529,060.9    526,816.0    516,672    565,824      6,413.1  matmul_fp8_v8_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     24.9       61,616,128        130    473,970.2    477,376.0    452,672    523,296     14,416.3  void matmul_fp8_v8_detail::matmul_fp8_v8_kernel<(int)256, (int)128, (int)2, (int)7, (int)8, (bool)0…
```
## matmul_fp8_v8_g10
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     56.4      138,016,416        260    530,832.4    528,528.0    516,288    889,408     23,179.1  matmul_fp8_v8_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     24.4       59,632,416        130    458,710.9    455,552.0    442,272    531,680     14,355.8  void matmul_fp8_v8_detail::matmul_fp8_v8_kernel<(int)256, (int)128, (int)2, (int)7, (int)10, (bool)…
```
## matmul_fp8_v8_g12
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     56.2      137,520,512        260    528,925.0    526,304.0    517,120    568,448      7,307.0  matmul_fp8_v8_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     24.0       58,742,048        130    451,861.9    449,040.0    435,616    512,128     12,802.4  void matmul_fp8_v8_detail::matmul_fp8_v8_kernel<(int)256, (int)128, (int)2, (int)7, (int)12, (bool)…
```
## matmul_fp8_v8_hilbert
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     56.4      137,769,952        260    529,884.4    528,480.0    516,192    556,864      5,469.6  matmul_fp8_v8_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     24.1       58,972,960        130    453,638.2    453,232.0    439,008    500,672      8,696.4  void matmul_fp8_v8_detail::matmul_fp8_v8_kernel<(int)256, (int)128, (int)2, (int)7, (int)6, (bool)0…
```
## matmul_fp8_v9_l1noalloc_tma_store_prefetch
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     59.4      138,242,880        260    531,703.4    531,696.0    514,304    569,696      7,868.6  matmul_fp8_v9_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     20.1       46,903,200        130    360,793.8    359,280.0    357,344    452,416      8,735.8  void matmul_fp8_v9_detail::matmul_fp8_v9_kernel<(int)256, (int)128, (int)2, (int)6, (bool)1, (bool)…
```
## matmul_fp8_v9_l1noalloc
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     58.3      137,638,144        260    529,377.5    527,616.0    517,376    560,384      5,825.7  matmul_fp8_v9_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     21.5       50,684,384        130    389,879.9    388,272.0    376,288    469,632     11,205.7  void matmul_fp8_v9_detail::matmul_fp8_v9_kernel<(int)256, (int)128, (int)2, (int)7, (bool)1, (bool)…
```
## matmul_fp8_v9_l1noalloc_tma_store
```shell
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     59.5      138,209,152        260    531,573.7    532,368.0    519,392    558,624      6,234.7  matmul_fp8_v9_detail::cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16 *, __nv_fp8_e4m3 *, long)    
     20.2       46,987,104        130    361,439.3    360,080.0    358,272    430,880      7,192.8  void matmul_fp8_v9_detail::matmul_fp8_v9_kernel<(int)256, (int)128, (int)2, (int)6, (bool)1, (bool)…
```
## matmul_fp8_v3
```shell
[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     62.4       80,217,152        130    617,055.0    594,992.0    538,016  1,274,432    103,340.7  void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)…
```