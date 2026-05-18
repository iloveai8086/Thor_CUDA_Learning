import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.cpp_extension

from profile_utils import build_trace_events, format_profile_summary, summarize_profile


CURRENT_DIR = Path(__file__).parent
CUTLASS_INCLUDE_DIR = CURRENT_DIR / "DeepGEMM" / "third-party" / "cutlass" / "include"
DEFAULT_KERNELS = [
    # "matmul_v0",
    "matmul_v1a",
    "matmul_v1b",
    "matmul_v2a",
    "matmul_v11_2",
    "matmul_v2b",
    "matmul_v3",
    "matmul_v3_2",
    "matmul_v3_cutedsl",
    "matmul_v4",
    "matmul_v4_2",
    "matmul_v5",
    "matmul_v5_2",
    "matmul_v6",
    "matmul_v6_2",
    "matmul_v6_2_g5",
    "matmul_v6_2_g6",
    "matmul_v6_2_g7",
    "matmul_v6_2_g8",
    "matmul_v6_2_g10",
    "matmul_v6_2_g12",
    "matmul_v6_2_hilbert",
    "matmul_v6_4",
    "matmul_v7a",
    "matmul_v7b",
    "matmul_v7c",
    "matmul_v7a_g6",
    "matmul_v7b_g6",
    "matmul_v7c_g6",
    "matmul_v7a_hilbert",
    "matmul_v7a_hilbert_l1noalloc",
    "matmul_v7a_hilbert_l1noalloc_tma_store",
    "matmul_v7a_hilbert_l1noalloc_tma_store_prefetch",
    "matmul_v7b_hilbert",
    "matmul_v7c_hilbert",
    "matmul_v8",
    "matmul_v9",
    "matmul_v10",
    "matmul_v10_2",
    "matmul_v11",
    "matmul_v12",
    "matmul_v12_nc",
    "matmul_v12_l1noalloc",
    "matmul_v12_hilbert",
    "matmul_v12_hilbert_l1noalloc",
    "matmul_v12_clc_hilbert",
    # "cute_fp16_gemm_1",
    "cute_fp16_gemm_4_aot",
    "deepgemm_fp8",
]

REF_MODE = "torch_fp32_to_bf16"
PYTHON_KERNELS = {"cute_fp16_gemm_1", "cute_fp16_gemm_1_aot", "cute_fp16_gemm_4_aot", "deepgemm_fp8"}
AOT_ARTIFACTS_DIR = CURRENT_DIR / "artifacts" / "bench_flops_aot"
KERNEL_INPUT_DTYPES = {
    "cute_fp16_gemm_4_aot": torch.float16,
}
PROFILE_KERNEL_CONFIG = {
    "matmul_v5": {"entry": "profile_matmul_v5", "num_rows": 10_000, "num_entries": 1_000},
    "matmul_v6": {"entry": "profile_matmul_v6", "num_rows": 200 * 6, "num_entries": 100_000},
    "matmul_v6_2": {"entry": "profile_matmul_v6_2", "num_rows": 200 * 6, "num_entries": 100_000},
    "matmul_v6_4": {"entry": "profile_matmul_v6_4", "num_rows": 200 * 6, "num_entries": 100_000},
    "matmul_v7a": {"entry": "profile_matmul_v7a", "num_rows": 200 * 6, "num_entries": 100_000},
    "matmul_v7b": {"entry": "profile_matmul_v7b", "num_rows": 200 * 6, "num_entries": 100_000},
    "matmul_v7a_hilbert_l1noalloc_tma_store_prefetch": {
        "entry": "profile_matmul_v7a_hilbert_l1noalloc_tma_store_prefetch",
        "num_rows": 200 * 6,
        "num_entries": 100_000,
    },
    "matmul_v12": {"entry": "profile_matmul_v12", "num_rows": 200 * 6, "num_entries": 100_000},
}


def get_module(verbose: bool = True):
    sources = [str(CURRENT_DIR / "matmul.cpp")]
    sources += [str(p) for p in sorted(CURRENT_DIR.glob("matmul_v*.cu"))]

    torch.utils.cpp_extension.load(
        name="module_local_sm110a",
        sources=sources,
        extra_include_paths=[str(CUTLASS_INCLUDE_DIR)],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_110a,code=sm_110a",
        ],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
        is_python_module=False,
    )
    return torch.ops.my_matmul


def make_deepgemm_fp8_benchmark(a_bf16: torch.Tensor, b_bf16: torch.Tensor):
    """Prepare DeepGEMM FP8 GEMM: A[M,K] @ B[K,N] -> D[M,N] in BF16."""
    # Add DeepGEMM to path if needed
    dg_path = str(CURRENT_DIR / "DeepGEMM")
    if dg_path not in sys.path:
        sys.path.insert(0, dg_path)

    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8

    m, k = a_bf16.shape
    n = b_bf16.shape[1]  # b_bf16 is [K, N]

    # DeepGEMM fp8_gemm_nt expects B as [N, K] (K-major, row-major)
    b_nt = b_bf16.T.contiguous()  # [N, K]

    # Quantize A [M, K] and B [N, K] to FP8 with per-token (1D) scaling, gran_k=128
    a_fp8, a_sf = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=128)
    b_fp8, b_sf = per_token_cast_to_fp8(b_nt, use_ue8m0=True, gran_k=128)

    d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    # Warm up / JIT compile
    deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d, recipe=(1, 1, 128))
    torch.cuda.synchronize()

    def run(_a: torch.Tensor, _b: torch.Tensor) -> torch.Tensor:
        _ = _a, _b
        deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d, recipe=(1, 1, 128))
        return d

    return run


def get_python_kernels():
    from fp16_gemm_1 import make_benchmark_matmul, matmul as cute_fp16_gemm_1
    from fp16_gemm_4_aot import make_benchmark_matmul_aot as make_benchmark_matmul_4_aot

    return {
        "cute_fp16_gemm_1": {
            "call": cute_fp16_gemm_1,
            "prepare": make_benchmark_matmul,
        },
        "cute_fp16_gemm_1_aot": {
            "prepare": make_benchmark_matmul_aot,
        },
        "cute_fp16_gemm_4_aot": {
            "prepare": make_benchmark_matmul_4_aot,
        },
        "deepgemm_fp8": {
            "prepare": make_deepgemm_fp8_benchmark,
        },
    }


def _aot_file_stem(kernel: str, m: int, n: int, k: int) -> str:
    return f"{kernel}_m{m}_n{n}_k{k}"


def _load_aot_metadata(metadata_path: Path):
    metadata = json.loads(metadata_path.read_text())
    return {
        "function_prefix": metadata["function_prefix"],
        "object_file": Path(metadata["object_file"]),
        "cubin_file": Path(metadata["cubin_file"]),
    }


def make_benchmark_matmul_aot(a: torch.Tensor, b: torch.Tensor):
    from cutlass.cute.runtime import load_module

    from fp16_gemm_1 import export_aot, prepare_cute_arguments

    m, k = a.shape
    n = b.shape[1]
    kernel_name = "cute_fp16_gemm_1_aot"
    file_stem = _aot_file_stem(kernel_name, m, n, k)
    output_dir = AOT_ARTIFACTS_DIR / file_stem
    metadata_path = output_dir / f"{file_stem}.json"

    if not metadata_path.exists():
        export_aot(
            (m, n, k),
            output_dir=str(output_dir),
            file_name=file_stem,
            function_prefix=file_stem,
        )

    metadata = _load_aot_metadata(metadata_path)
    object_file = metadata["object_file"]
    cubin_file = metadata["cubin_file"]
    function_prefix = metadata["function_prefix"]

    if not object_file.exists():
        raise FileNotFoundError(f"AOT object file not found: {object_file}")
    if not cubin_file.exists():
        raise FileNotFoundError(f"AOT cubin file not found: {cubin_file}")

    module = load_module(str(object_file))
    fn = getattr(module, function_prefix)
    a_tensor, b_tensor, c_mn, c_tensor = prepare_cute_arguments(a, b)

    fn(a_tensor, b_tensor, c_tensor)
    torch.cuda.synchronize()

    def run(_a: torch.Tensor, _b: torch.Tensor) -> torch.Tensor:
        _ = _a, _b
        fn(a_tensor, b_tensor, c_tensor)
        return c_mn

    return run


def parse_shape(shape: str):
    m, n, k = map(int, shape.split(","))
    return m, n, k


def ref_mode_for_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "torch_fp32_to_bf16"
    if dtype == torch.float16:
        return "torch_fp32_to_fp16"
    raise ValueError(f"Unsupported reference dtype: {dtype}")


def input_dtype_for_kernel(name: str) -> torch.dtype:
    return KERNEL_INPUT_DTYPES.get(name, torch.bfloat16)


def gemm_flops(m: int, n: int, k: int) -> float:
    # Standard GEMM benchmark convention: 2 * M * N * K
    return float(2 * m * n * k)


def error_stats(out: torch.Tensor, ref: torch.Tensor, rel_mask_eps: float = 1e-2):
    out_f = out.float()
    ref_f = ref.float()
    diff = (out_f - ref_f).abs()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rmse = torch.sqrt(((out_f - ref_f) ** 2).mean()).item()

    denom = ref_f.abs().clamp_min(1e-6)
    rel = diff / denom
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    mask = ref_f.abs() > rel_mask_eps
    if mask.any():
        max_rel_masked = rel[mask].max().item()
        mean_rel_masked = rel[mask].mean().item()
    else:
        max_rel_masked = float("nan")
        mean_rel_masked = float("nan")

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "max_rel_masked": max_rel_masked,
        "mean_rel_masked": mean_rel_masked,
    }


def get_or_create_ref_bin(
    a: torch.Tensor,
    b: torch.Tensor,
    m: int,
    n: int,
    k: int,
    seed: int,
    ref_mode: str,
    ref_dtype: torch.dtype,
):
    ref_path = CURRENT_DIR / f"ref_m{m}_n{n}_k{k}_seed{seed}_{ref_mode}.bin"

    if ref_path.exists():
        print(f"[REF] loading from {ref_path}")
        arr_u16 = np.fromfile(ref_path, dtype=np.uint16)
        expected = m * n
        if arr_u16.size != expected:
            raise RuntimeError(
                f"ref size mismatch: got {arr_u16.size}, expected {expected}. "
                f"Delete {ref_path} and rerun."
            )
        ref_cpu = (
            torch.from_numpy(arr_u16.view(np.int16))
            .view(ref_dtype)
            .reshape(m, n)
            .contiguous()
        )
        return ref_cpu.cuda(non_blocking=True), ref_path

    print(f"[REF] not found, generating once via torch.mm(a.float(), b.float()).to({ref_dtype}) and saving to {ref_path}")
    ref = torch.mm(a.float(), b.float()).to(dtype=ref_dtype).contiguous()
    ref_cpu = ref.cpu()

    ref_u16 = ref_cpu.view(torch.int16).numpy().view(np.uint16)
    ref_u16.tofile(ref_path)
    return ref, ref_path


def benchmark_cuda_fn(fn, a, b, warmup: int, iters: int):
    # warmup
    for _ in range(warmup):
        _ = fn(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1e3 / iters
    return out, {
        "avg_ms": avg_ms,
        "min_ms": float("nan"),
        "max_ms": float("nan"),
    }


def verify_and_benchmark(
    name: str,
    fn,
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor,
    atol: float,
    rtol: float,
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
):
    out, timing = benchmark_cuda_fn(fn, a, b, warmup, iters)
    stats = error_stats(out, ref)

    # fix dtype mismatch: always compare in float32
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)

    total_flops = gemm_flops(m, n, k)
    avg_s = timing["avg_ms"] / 1e3
    flops_per_s = total_flops / avg_s
    tflops = flops_per_s / 1e12
    gflop = total_flops / 1e9

    return {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "avg_ms": timing["avg_ms"],
        "min_ms": timing["min_ms"],
        "max_ms": timing["max_ms"],
        "gflop": gflop,
        "flops": flops_per_s,
        "tflops": tflops,
        **stats,
    }


def print_result_line(r):
    print(
        f"[{r['status']}] {r['name']:<22} "
        f"avg={r['avg_ms']:.3f} ms "
        f"TFLOPS={r['tflops']:.3f} "
        f"max_abs={r['max_abs']:.6f} "
        f"mean_abs={r['mean_abs']:.6e} "
        f"rmse={r['rmse']:.6e} "
        f"max_rel={r['max_rel']:.6f} "
        f"max_rel(|ref|>1e-2)={r['max_rel_masked']:.6f}"
    )


def make_error_result(name: str, status: str):
    return {
        "name": name,
        "status": status,
        "avg_ms": float("nan"),
        "min_ms": float("nan"),
        "max_ms": float("nan"),
        "gflop": float("nan"),
        "flops": float("nan"),
        "tflops": float("nan"),
        "max_abs": float("nan"),
        "mean_abs": float("nan"),
        "rmse": float("nan"),
        "max_rel": float("nan"),
        "mean_rel": float("nan"),
        "max_rel_masked": float("nan"),
        "mean_rel_masked": float("nan"),
    }


def resolve_kernel(kernel: str, cpp_kernels, python_kernels, a: torch.Tensor, b: torch.Tensor):
    if cpp_kernels is not None and hasattr(cpp_kernels, kernel):
        return getattr(cpp_kernels, kernel)

    python_kernel = python_kernels.get(kernel)
    if python_kernel is None:
        return None
    if isinstance(python_kernel, dict) and "prepare" in python_kernel:
        return python_kernel["prepare"](a, b)
    return python_kernel


def run_profile_kernel(cpp_kernels, kernel: str, a: torch.Tensor, b: torch.Tensor, warmup: int, output_path: Path):
    profile_config = PROFILE_KERNEL_CONFIG.get(kernel)
    if profile_config is None:
        supported = ", ".join(sorted(PROFILE_KERNEL_CONFIG))
        raise ValueError(f"Kernel {kernel} does not support custom profiling. Supported kernels: {supported}")

    entry_name = profile_config["entry"]
    if cpp_kernels is None or not hasattr(cpp_kernels, entry_name):
        raise ValueError(f"Profile entry {entry_name} is not exported by the current extension build")

    profile_fn = getattr(cpp_kernels, entry_name)
    profiler = torch.zeros(
        profile_config["num_rows"],
        1 + profile_config["num_entries"] * 4,
        dtype=torch.int64,
        device="cuda",
    )

    for _ in range(warmup):
        profile_fn(a, b, profiler, profile_config["num_entries"])
    torch.cuda.synchronize()

    profiler.zero_()
    profile_fn(a, b, profiler, profile_config["num_entries"])
    torch.cuda.synchronize()

    profile_data = profiler.tolist()
    events, decoded_events = build_trace_events(profile_data)

    if not events:
        raise RuntimeError(f"Profiler for {kernel} produced no events")

    print(format_profile_summary(summarize_profile(decoded_events)))

    trace = {"traceEvents": events}
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            json.dump(trace, handle)
    else:
        output_path.write_text(json.dumps(trace), encoding="utf-8")

    return len(events), output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fair benchmark for torch.mm and matmul_v0...v8 on sm110a."
    )
    parser.add_argument("--shape", default="4096,4096,4096", help="M,N,K")
    parser.add_argument("--kernel", default="", help="Benchmark one or more kernels. Use commas for multiple, e.g. matmul_v6_2_g6,matmul_v6_2_g8")
    parser.add_argument("--profile", action="store_true", help="Run custom in-kernel profiler and save a trace json/json.gz")
    parser.add_argument("--profile-output", default="trace.json.gz", help="Output path for --profile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--no-verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(args.seed)
    m, n, k = parse_shape(args.shape)

    print(f"shape: M={m}, N={n}, K={k}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"tolerance: atol={args.atol}, rtol={args.rtol}")
    print(f"benchmark: warmup={args.warmup}, iters={args.iters}")
    print(f"FLOPs formula: 2 * M * N * K = {gemm_flops(m, n, k):.0f}")
    print(f"GFLOP per GEMM: {gemm_flops(m, n, k) / 1e9:.3f}")
    print()

    if args.profile:
        if not args.kernel:
            raise SystemExit("--profile requires --kernel, e.g. --kernel matmul_v5")
        if args.kernel in PYTHON_KERNELS:
            raise SystemExit(f"--profile is only supported for C++ kernels with custom profiler hooks, got {args.kernel}")

        cpp_kernels = get_module(verbose=not args.no_verbose_build)
        dtype = input_dtype_for_kernel(args.kernel)
        scale = k ** -0.5
        a = torch.randn(m, k, device="cuda").mul(scale).to(dtype)
        b = torch.randn(n, k, device="cuda").mul(scale).T.to(dtype)
        event_count, output_path = run_profile_kernel(
            cpp_kernels=cpp_kernels,
            kernel=args.kernel,
            a=a,
            b=b,
            warmup=args.warmup,
            output_path=Path(args.profile_output),
        )
        print(f"[PROFILE] {args.kernel} wrote {event_count} events to {output_path}")
        return

    kernels = [name.strip() for name in args.kernel.split(",") if name.strip()] if args.kernel else DEFAULT_KERNELS
    need_cpp_kernels = any(kernel not in PYTHON_KERNELS for kernel in kernels)
    cpp_kernels = get_module(verbose=not args.no_verbose_build) if need_cpp_kernels else None
    python_kernels = get_python_kernels() if any(kernel in PYTHON_KERNELS for kernel in kernels) else {}

    scale = k ** -0.5
    base_a = torch.randn(m, k, device="cuda").mul(scale)
    base_b = torch.randn(n, k, device="cuda").mul(scale).T

    required_dtypes = {torch.bfloat16}
    required_dtypes.update(input_dtype_for_kernel(kernel) for kernel in kernels)
    inputs_by_dtype = {
        dtype: (base_a.to(dtype=dtype), base_b.to(dtype=dtype)) for dtype in required_dtypes
    }
    refs_by_dtype = {}
    for dtype in required_dtypes:
        a_dtype, b_dtype = inputs_by_dtype[dtype]
        ref_mode = ref_mode_for_dtype(dtype)
        ref, ref_path = get_or_create_ref_bin(
            a_dtype,
            b_dtype,
            m,
            n,
            k,
            args.seed,
            ref_mode,
            dtype,
        )
        refs_by_dtype[dtype] = ref
        print(f"[REF] ready ({dtype}): {ref_path}")
    print()

    results = []

    torch_variants = [
        ("torch.mm_bf16", torch.bfloat16, lambda x, y: torch.mm(x, y)),
        # ("torch.mm_fp32", lambda x, y: torch.mm(x.float(), y.float())),
        # ("torch.mm_fp32_to_bf16", lambda x, y: torch.mm(x.float(), y.float()).bfloat16()),
    ]
    if torch.float16 in required_dtypes:
        torch_variants.append(("torch.mm_fp16", torch.float16, lambda x, y: torch.mm(x, y)))

    print("=== Torch baselines ===")
    for name, dtype, fn in torch_variants:
        try:
            a, b = inputs_by_dtype[dtype]
            ref = refs_by_dtype[dtype]
            r = verify_and_benchmark(
                name=name,
                fn=fn,
                a=a,
                b=b,
                ref=ref,
                atol=args.atol,
                rtol=args.rtol,
                m=m,
                n=n,
                k=k,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(r)
            print_result_line(r)
        except Exception as exc:
            print(f"[ERROR] {name:<22} {type(exc).__name__}: {exc}")
            results.append(
                {
                    "name": name,
                    "status": "ERROR",
                    "avg_ms": float("nan"),
                    "min_ms": float("nan"),
                    "max_ms": float("nan"),
                    "gflop": float("nan"),
                    "flops": float("nan"),
                    "tflops": float("nan"),
                    "max_abs": float("nan"),
                    "mean_abs": float("nan"),
                    "rmse": float("nan"),
                    "max_rel": float("nan"),
                    "mean_rel": float("nan"),
                    "max_rel_masked": float("nan"),
                    "mean_rel_masked": float("nan"),
                }
            )

    print()
    print("=== Custom kernels ===")
    for kernel in kernels:
        dtype = input_dtype_for_kernel(kernel)
        a, b = inputs_by_dtype[dtype]
        ref = refs_by_dtype[dtype]
        fn = resolve_kernel(kernel, cpp_kernels, python_kernels, a, b)
        if fn is None:
            print(f"[MISSING] {kernel}")
            results.append(make_error_result(kernel, "MISSING"))
            continue

        try:
            r = verify_and_benchmark(
                name=kernel,
                fn=fn,
                a=a,
                b=b,
                ref=ref,
                atol=args.atol,
                rtol=args.rtol,
                m=m,
                n=n,
                k=k,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(r)
            print_result_line(r)
        except Exception as exc:
            print(f"[ERROR] {kernel:<22} {type(exc).__name__}: {exc}")
            results.append(make_error_result(kernel, "ERROR"))

    print("\n=== Summary (original order) ===")
    kernel_width = max([32] + [len(r["name"]) for r in results])
    print(
        f"{'Kernel':<{kernel_width}} {'Status':<8} {'Avg(ms)':>10} {'TFLOPS':>10} "
        f"{'MaxAbs':>12} {'MeanAbs':>12} {'RMSE':>12} {'MaxRel':>12} {'MaxRel>|1e-2|':>14}"
    )
    print("-" * (kernel_width + 99))
    for r in results:
        print(
            f"{r['name']:<{kernel_width}} {r['status']:<8} "
            f"{r['avg_ms']:>10.3f} {r['tflops']:>10.3f} "
            f"{r['max_abs']:>12.6f} "
            f"{r['mean_abs']:>12.6e} "
            f"{r['rmse']:>12.6e} "
            f"{r['max_rel']:>12.6f} "
            f"{r['max_rel_masked']:>14.6f}"
        )

    print("\n=== Summary (sorted by TFLOPS desc) ===")
    valid_results = [r for r in results if r["status"] not in ("ERROR", "MISSING") and not np.isnan(r["tflops"])]
    valid_results = sorted(valid_results, key=lambda x: x["tflops"], reverse=True)

    print(
        f"{'Rank':<6} {'Kernel':<{kernel_width}} {'Status':<8} {'Avg(ms)':>10} {'TFLOPS':>10} "
        f"{'MaxAbs':>12} {'RMSE':>12}"
    )
    print("-" * (kernel_width + 74))
    for i, r in enumerate(valid_results, 1):
        print(
            f"{i:<6} {r['name']:<{kernel_width}} {r['status']:<8} "
            f"{r['avg_ms']:>10.3f} {r['tflops']:>10.3f} "
            f"{r['max_abs']:>12.6f} {r['rmse']:>12.6e}"
        )

    num_pass = sum(1 for r in results if r["status"] == "PASS")
    num_fail = sum(1 for r in results if r["status"] == "FAIL")
    num_err = sum(1 for r in results if r["status"] == "ERROR")
    num_missing = sum(1 for r in results if r["status"] == "MISSING")

    print("\n=== Totals ===")
    print(f"PASS={num_pass}, FAIL={num_fail}, ERROR={num_err}, MISSING={num_missing}")

    if num_fail > 0 or num_err > 0:
        raise SystemExit(1)

# python bench_flops.py --kernel matmul_v5 --profile --profile-output v5_trace.json.gz
if __name__ == "__main__":
    main()
