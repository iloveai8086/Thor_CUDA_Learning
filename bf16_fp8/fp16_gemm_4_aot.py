import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack, load_module


CURRENT_DIR = Path(__file__).parent
EXTERNAL_FP16_GEMM_4 = Path(
    "/home/lxw/Project/cutlass44/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py"
)
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "artifacts" / "fp16_gemm_4_aot"


@lru_cache(maxsize=1)
def load_external_fp16_gemm_4():
    if not EXTERNAL_FP16_GEMM_4.exists():
        raise FileNotFoundError(f"External fp16_gemm_4.py not found: {EXTERNAL_FP16_GEMM_4}")

    spec = importlib.util.spec_from_file_location(
        "cutlass44_external_fp16_gemm_4",
        EXTERNAL_FP16_GEMM_4,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {EXTERNAL_FP16_GEMM_4}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_input_dtype() -> torch.dtype:
    mod = load_external_fp16_gemm_4()
    return cutlass_torch.dtype(mod.io_dtype)


def _prepare_matmul_tensors(a: torch.Tensor, b: torch.Tensor):
    mod = load_external_fp16_gemm_4()
    expected_dtype = cutlass_torch.dtype(mod.io_dtype)

    if not a.is_cuda or not b.is_cuda:
        raise ValueError("fp16_gemm_4_aot requires CUDA tensors")
    if a.dtype != expected_dtype or b.dtype != expected_dtype:
        raise ValueError(f"fp16_gemm_4_aot requires {expected_dtype} inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("fp16_gemm_4_aot expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"fp16_gemm_4_aot shape mismatch: A is {tuple(a.shape)}, B is {tuple(b.shape)}"
        )

    m, k = a.shape
    _, n = b.shape
    tile_m, tile_n = mod.mma_tiler_mnk[:2]
    if m % tile_m != 0 or n % tile_n != 0:
        raise ValueError(
            f"fp16_gemm_4_aot requires M/N divisible by {(tile_m, tile_n)}, got {(m, n)}"
        )

    a_mk = a.contiguous()
    b_nk = b.transpose(0, 1).contiguous()
    c_mn = torch.empty((m, n), device=a.device, dtype=a.dtype)

    a_tensor = (
        from_dlpack(a_mk, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    b_tensor = (
        from_dlpack(b_nk, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    c_tensor = (
        from_dlpack(c_mn, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=n)
    )

    return a_mk, b_nk, a_tensor, b_tensor, c_mn, c_tensor


def prepare_cute_arguments(a: torch.Tensor, b: torch.Tensor):
    _, _, a_tensor, b_tensor, c_mn, c_tensor = _prepare_matmul_tensors(a, b)
    return a_tensor, b_tensor, c_mn, c_tensor


def _get_max_active_clusters(mod):
    hardware = utils.HardwareInfo()
    preferred = hardware.get_max_active_clusters(
        mod.preferred_cluster_shape_mnk[0] * mod.preferred_cluster_shape_mnk[1]
    )
    fallback = hardware.get_max_active_clusters(
        mod.fallback_cluster_shape_mnk[0] * mod.fallback_cluster_shape_mnk[1]
    )
    return preferred, fallback


def export_aot(
    mnk: Tuple[int, int, int],
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    file_name: str = "cute_fp16_gemm_4",
    function_prefix: str = "cute_fp16_gemm_4",
):
    mod = load_external_fp16_gemm_4()
    m, n, k = mnk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype = cutlass_torch.dtype(mod.io_dtype)
    a = torch.empty((m, k), dtype=torch.int32).random_(-2, 2).to(device="cuda", dtype=dtype)
    b = (
        torch.empty((n, k), dtype=torch.int32)
        .random_(-2, 2)
        .to(device="cuda", dtype=dtype)
        .transpose(0, 1)
        .contiguous()
    )
    a_tensor, b_tensor, _, c_tensor = prepare_cute_arguments(a, b)
    preferred_max_active_clusters, fallback_max_active_clusters = _get_max_active_clusters(mod)

    compile_options = f"--dump-dir={output_path} --keep-cubin"
    compiled = cute.compile(
        mod.host_function,
        a_tensor,
        b_tensor,
        c_tensor,
        preferred_max_active_clusters,
        fallback_max_active_clusters,
        options=compile_options,
    )

    object_file = output_path / f"{file_name}.o"
    header_file = output_path / f"{file_name}.h"
    export_mode = "cubin-only"
    if hasattr(compiled, "export_to_c"):
        compiled.export_to_c(
            file_path=str(output_path),
            file_name=file_name,
            function_prefix=function_prefix,
        )
        export_mode = "export_to_c"
    elif hasattr(compiled, "engine") and hasattr(compiled.engine, "dump_to_object_file"):
        compiled.engine.dump_to_object_file(str(object_file))
        export_mode = "engine.dump_to_object_file"

    cubin_path = output_path / f"{file_name}.cubin"
    if getattr(compiled, "__cubin__", None) is not None:
        cubin_path.write_bytes(compiled.__cubin__)

    kernel_names = list(getattr(compiled, "kernel_info", {}).keys())
    function_name = getattr(compiled, "function_name", function_prefix)
    c_iface_symbol = f"_mlir_ciface_{function_name}"
    metadata_path = output_path / f"{file_name}.json"
    metadata = {
        "source_script": str(EXTERNAL_FP16_GEMM_4),
        "function_prefix": function_prefix,
        "function_name": function_name,
        "c_iface_symbol": c_iface_symbol,
        "kernel_names": kernel_names,
        "export_mode": export_mode,
        "object_file": str(object_file.resolve()),
        "header_file": str(header_file.resolve()),
        "cubin_file": str(cubin_path.resolve()),
        "input_dtype": str(dtype),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "output_dir": str(output_path.resolve()),
        "object_file": str(object_file.resolve()),
        "header_file": str(header_file.resolve()),
        "cubin_file": str(cubin_path.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "function_prefix": function_prefix,
        "function_name": function_name,
        "c_iface_symbol": c_iface_symbol,
        "kernel_names": kernel_names,
        "export_mode": export_mode,
        "source_script": str(EXTERNAL_FP16_GEMM_4),
    }


def make_benchmark_matmul_aot(
    a: torch.Tensor,
    b: torch.Tensor,
    artifacts_root: Path = DEFAULT_OUTPUT_DIR,
):
    m, k = a.shape
    n = b.shape[1]
    file_stem = f"cute_fp16_gemm_4_m{m}_n{n}_k{k}"
    output_dir = Path(artifacts_root) / file_stem
    metadata_path = output_dir / f"{file_stem}.json"

    if not metadata_path.exists():
        export_aot(
            (m, n, k),
            output_dir=str(output_dir),
            file_name=file_stem,
            function_prefix=file_stem,
        )

    metadata = json.loads(metadata_path.read_text())
    object_file = Path(metadata["object_file"])
    cubin_file = Path(metadata["cubin_file"])
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