#!/usr/bin/env python3
"""
PyTorch FC-Layer GEMM Benchmark
================================
Measures the performance of a single nn.Linear (fully-connected) layer in two modes:

  Mode 1 – CUDA Core baseline (FP32):
    torch.backends.cuda.matmul.allow_tf32 = False
    Forces strict IEEE-754 FP32 arithmetic executed on CUDA cores.

  Mode 2 – Tensor Core path (TF32):
    torch.backends.cuda.matmul.allow_tf32 = True
    PyTorch passes the matmul to cuBLAS with CUBLAS_COMPUTE_32F_FAST_TF32,
    which rounds FP32 mantissa to 10 bits internally, routes the GEMM through
    Tensor Cores, and returns an FP32 result. Requires Ampere (SM80+) or newer.

Benchmark protocol:
  - <warmup> iterations to let cuBLAS warm up and pick the optimal algorithm.
  - <iters>  timed iterations using torch.cuda.Event for sub-millisecond GPU timing.
  - Average latency = total_gpu_time / iters.
  - Throughput (TFLOPS) = 2·M·K·N / (latency_s) / 1e12   (multiply-add = 2 FLOPs).

Usage:
  python benchmark_pytorch.py [--warmup 20] [--iters 100] [--output pytorch_results.json]
"""

import argparse
import json
import sys

import torch
import torch.nn as nn

# ── Size sweep ─────────────────────────────────────────────────────────────────
# Each tuple: (batch_size M, in_features K, out_features N)
# All dimensions are multiples of 16 so Tensor Core tiles are fully utilised.
SIZES = [
    (64,   1024, 1024),
    (64,   2048, 2048),
    (64,   4096, 4096),
    (128,  1024, 1024),
    (128,  2048, 2048),
    (128,  4096, 4096),
    (256,  1024, 1024),
    (256,  2048, 2048),
    (256,  4096, 4096),
    (512,  4096, 4096),
    (1024, 4096, 4096),
]


# ── Core benchmark function ────────────────────────────────────────────────────

def benchmark_fc_layer(
    batch_size: int,
    in_features: int,
    out_features: int,
    use_tf32: bool,
    device: str,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    """
    Run a single nn.Linear forward pass benchmark.

    Returns
    -------
    avg_latency_ms : float
        Mean GPU wall-time per forward pass in milliseconds.
    tflops : float
        Effective throughput in TFLOPS.
    """
    # TF32 flag must be set before the kernel is launched.
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32       = use_tf32

    # Build model and inputs on device.
    layer = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32).to(device)
    layer.eval()

    x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)

    # ── Warm-up: forces cuBLAS to select and cache optimal kernel/algorithm ──
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)
    torch.cuda.synchronize()

    # ── Timed benchmark using CUDA events (GPU-side clock, no PCIe round-trips) ──
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_ev.record()
        for _ in range(iters):
            _ = layer(x)
        end_ev.record()

    torch.cuda.synchronize()

    elapsed_ms      = start_ev.elapsed_time(end_ev)   # total time for all iters
    avg_latency_ms  = elapsed_ms / iters

    # FLOPs: each multiply-add counts as 2 ops → 2·M·K·N
    flops  = 2.0 * batch_size * in_features * out_features
    tflops = (flops / (avg_latency_ms * 1e-3)) / 1e12

    return avg_latency_ms, tflops


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyTorch FC-Layer GEMM Benchmark: CUDA Core FP32 vs Tensor Core TF32"
    )
    p.add_argument("--warmup", type=int, default=20,
                   help="Warm-up iterations before timing (default: 20)")
    p.add_argument("--iters",  type=int, default=100,
                   help="Timed iterations per configuration (default: 100)")
    p.add_argument("--output", type=str, default="pytorch_results.json",
                   help="Output JSON file path (default: pytorch_results.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. A CUDA-capable GPU is required.")
        sys.exit(1)

    device   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    sm_major, sm_minor = torch.cuda.get_device_capability(0)

    print("=" * 82)
    print("  PyTorch FC-Layer GEMM Benchmark  –  CUDA Core FP32 vs Tensor Core TF32")
    print("=" * 82)
    print(f"  GPU            : {gpu_name}  (SM {sm_major}{sm_minor})")
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  CUDA           : {torch.version.cuda}")
    print(f"  Warm-up iters  : {args.warmup}")
    print(f"  Bench iters    : {args.iters}")

    if sm_major < 8:
        print("\n  [WARNING] TF32 Tensor Cores require Ampere (SM80+) or newer.")
        print("            On this GPU the TF32 path will fall back to FP32 CUDA cores.")
        print("            Results may show little or no speedup.")

    print("=" * 82)

    fmt    = "{:<23} {:>8} {:>8} {:>8} {:>14} {:>10}"
    header = fmt.format("Mode", "Batch", "InFeat", "OutFeat", "Latency(ms)", "TFLOPS")
    sep    = "-" * len(header)
    print(f"\n{header}\n{sep}")

    results = []

    for batch, in_f, out_f in SIZES:
        for use_tf32, mode_name in [
            (False, "CUDA Core (FP32)"),
            (True,  "Tensor Core (TF32)"),
        ]:
            lat, tflops = benchmark_fc_layer(
                batch, in_f, out_f,
                use_tf32, device,
                args.warmup, args.iters,
            )
            results.append({
                "mode":        mode_name,
                "batch_size":  batch,
                "in_features": in_f,
                "out_features": out_f,
                "latency_ms":  lat,
                "tflops":      tflops,
                "use_tf32":    use_tf32,
            })
            print(fmt.format(mode_name, batch, in_f, out_f,
                             f"{lat:.4f}", f"{tflops:.3f}"))

    print(sep)

    # ── Speedup summary table ──────────────────────────────────────────────────
    print("\n  Speedup Summary  (Tensor Core TF32  vs  CUDA Core FP32)")
    sfmt   = "  {:<30} {:>10} {:>13} {:>13}"
    sheadr = sfmt.format("Size", "Speedup", "TC TFLOPS", "CUDA TFLOPS")
    print(sheadr)
    print("  " + "-" * (len(sheadr) - 2))

    for batch, in_f, out_f in SIZES:
        cuda_r  = next(r for r in results
                       if r["batch_size"] == batch and r["in_features"] == in_f
                       and not r["use_tf32"])
        tc_r    = next(r for r in results
                       if r["batch_size"] == batch and r["in_features"] == in_f
                       and r["use_tf32"])
        speedup = cuda_r["latency_ms"] / tc_r["latency_ms"]
        label   = f"B={batch}  {in_f}×{out_f}"
        print(sfmt.format(label, f"{speedup:.2f}×",
                          f"{tc_r['tflops']:.3f}", f"{cuda_r['tflops']:.3f}"))

    # ── Persist results ────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {args.output}")


if __name__ == "__main__":
    main()
