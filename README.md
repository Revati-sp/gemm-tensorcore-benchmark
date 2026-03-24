# GEMM Tensor Core Benchmark

Measures and explains the performance difference between the **CUDA Core (FP32)** path and the **Tensor Core (TF32)** path for a General Matrix Multiply (GEMM) that is equivalent to a single fully-connected (FC) layer forward pass.

Two independent implementations are provided:

| Implementation | File | API |
|---|---|---|
| PyTorch (Python) | `benchmark_pytorch.py` | `nn.Linear`, `torch.backends.cuda.matmul.allow_tf32` |
| CUDA C++ | `benchmark_cublas.cu` | `cublasSgemm` / `cublasGemmEx` |

---

## Background: CUDA Cores vs Tensor Cores

### CUDA Cores (FP32 baseline)
Traditional CUDA cores execute one floating-point multiply-add (FMA) per clock cycle per core.  On a modern GPU with, say, 10 000 FP32 CUDA cores running at 1.7 GHz the peak FP32 throughput is ≈ 34 TFLOPS.  Each operand is kept at full 23-bit mantissa precision throughout.

### Tensor Cores (TF32 path)
Tensor Cores are dedicated matrix-multiply-accumulate (MMA) units introduced on Volta (V100) and significantly expanded on Ampere (A100, RTX 3090) and later GPUs. They operate on tiles of matrices in a single instruction, giving a **4× to 8× throughput advantage** for GEMM workloads compared to CUDA cores.

**TF32 (TensorFloat-32)** is a storage format with:
- 8-bit exponent  (same range as FP32)
- 10-bit mantissa (vs 23 bits in FP32; same precision as FP16)
- 1 sign bit

cuBLAS (and PyTorch) can transparently round each FP32 operand to TF32 *before* passing it to the Tensor Core MMA hardware, then accumulate the result back in FP32. The net effect:
- **Inputs and outputs remain FP32** — no user code changes needed.
- **Internal compute uses Tensor Cores** — large throughput gain.
- **Slight numerical difference** (≤ 0.2 % relative error in practice) due to mantissa truncation.

This is the exact behaviour controlled by `torch.backends.cuda.matmul.allow_tf32` in PyTorch and by `CUBLAS_COMPUTE_32F_FAST_TF32` / `CUBLAS_TF32_TENSOR_OP_MATH` in cuBLAS.

---

## Requirements

| Dependency | Minimum version | Notes |
|---|---|---|
| CUDA Toolkit | 11.0 | `CUBLAS_COMPUTE_32F_FAST_TF32` added in 11.0 |
| GPU | SM 8.0 (Ampere) | TF32 Tensor Cores available from Ampere; benchmark still runs on older GPUs but TF32 gives no speedup |
| Python | 3.9+ | Uses `tuple[…]` generics (PEP 585, available from 3.9) |
| PyTorch | 2.0+ | CUDA build required |
| numpy | 1.24+ | |
| pandas | 2.0+ | |
| matplotlib | 3.7+ | |

### Install Python packages

```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA separately (replace `cu121` with your CUDA version):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Project Structure

```
gemm-tensorcore-benchmark/
├── benchmark_pytorch.py    # PyTorch FC-layer benchmark
├── benchmark_cublas.cu     # CUDA C++ cuBLAS benchmark
├── plot_results.py         # Unified plotting script
├── Makefile                # Build and run helpers
├── requirements.txt        # Python dependencies
├── run_all.sh              # One-shot end-to-end runner
└── README.md               # This file
```

---

## Quick Start

### Option A — one-shot script (recommended)

```bash
# Ampere GPU (RTX 3090 / A100 / A30):
./run_all.sh --arch sm_80 --warmup 20 --iters 100

# RTX 3080 Ti / A40 / A10:
./run_all.sh --arch sm_86

# RTX 4090 (Ada Lovelace):
./run_all.sh --arch sm_89

# H100 (Hopper):
./run_all.sh --arch sm_90

# PyTorch only (no nvcc required):
./run_all.sh --skip-cublas
```

### Option B — individual steps

```bash
# 1. PyTorch benchmark
python benchmark_pytorch.py --warmup 20 --iters 100

# 2. cuBLAS benchmark
make ARCH=sm_80         # compile
./benchmark_cublas      # run  (writes cublas_results.csv)

# 3. Generate plots
python plot_results.py --gpu "NVIDIA A100 SXM4 80GB" --outdir results/
```

---

## Benchmark Protocol

Both implementations follow the same protocol to ensure fair comparison:

1. **Warm-up** (default 20 iterations) — forces cuBLAS to select and JIT-compile the optimal algorithm, warms up GPU caches and power management.
2. **Timed loop** (default 100 iterations) — all iterations are issued without host synchronisation. A single GPU timestamp is captured before the first kernel and after the last.
3. **Average latency** = `total_gpu_time / num_iters`
4. **TFLOPS** = `2 · M · K · N / latency_s / 1e12`  
   (Each multiply-add is counted as 2 floating-point operations.)

GPU timing uses `torch.cuda.Event` (PyTorch) and `cudaEventRecord` (CUDA C++) — both measure elapsed time on the GPU clock with sub-millisecond precision, avoiding PCIe round-trip overhead.

---

## Size Sweep

Eleven configurations are benchmarked, varying both batch size and matrix dimension:

| Batch (M) | In-features (K) | Out-features (N) | GFLOPs |
|---|---|---|---|
| 64 | 1 024 | 1 024 | 0.13 |
| 64 | 2 048 | 2 048 | 0.54 |
| 64 | 4 096 | 4 096 | 2.15 |
| 128 | 1 024 | 1 024 | 0.27 |
| 128 | 2 048 | 2 048 | 1.07 |
| 128 | 4 096 | 4 096 | 4.29 |
| 256 | 1 024 | 1 024 | 0.54 |
| 256 | 2 048 | 2 048 | 2.15 |
| 256 | 4 096 | 4 096 | 8.59 |
| 512 | 4 096 | 4 096 | 17.2 |
| 1 024 | 4 096 | 4 096 | 34.4 |

All dimensions are multiples of 16, ensuring Tensor Core tiles (16×16×8 for TF32) are fully utilised with no padding.

---

## Output Files

| File | Description |
|---|---|
| `pytorch_results.json` | Raw PyTorch benchmark results (JSON array) |
| `cublas_results.csv` | Raw cuBLAS benchmark results (CSV) |
| `plot_latency_tflops.png` | Grouped bar charts: avg latency and TFLOPS per configuration |
| `plot_speedup.png` | Tensor Core speedup over CUDA Core per configuration |
| `plot_tflops_vs_size.png` | TFLOPS vs matrix dimension (line chart, averaged over batch sizes) |
| `plot_throughput_all.png` | TFLOPS across all configurations for both frameworks (when both available) |

---

## Expected Results (NVIDIA A100 SXM4 80GB)

The table below shows representative numbers; actual values depend on your GPU model, clocks, and driver version.

```
Mode                    Batch   InFeat  OutFeat   Latency(ms)    TFLOPS
──────────────────────────────────────────────────────────────────────────
CUDA Core (FP32)           64     4096     4096        0.0892    38.4
Tensor Core (TF32)         64     4096     4096        0.0214   159.7
CUDA Core (FP32)          256     4096     4096        0.3423    40.1
Tensor Core (TF32)        256     4096     4096        0.0853   161.0
CUDA Core (FP32)         1024     4096     4096        1.3511    40.7
Tensor Core (TF32)       1024     4096     4096        0.3378   162.8
```

Typical speedups:
- Small matrices (B=64, 1K×1K): **1.5–2×** (memory-bandwidth bound, limited tile reuse)
- Large matrices (B=1024, 4K×4K): **3.5–5×** (compute bound, full Tensor Core utilisation)

---

## Implementation Details

### PyTorch — `benchmark_pytorch.py`

```
torch.backends.cuda.matmul.allow_tf32 = False   # Mode 1: CUDA cores, strict FP32
torch.backends.cuda.matmul.allow_tf32 = True    # Mode 2: Tensor Cores, TF32
```

PyTorch translates `nn.Linear` into a `cublasSgemm` / `cublasGemmEx` call internally. Setting `allow_tf32 = True` tells PyTorch to pass `CUBLAS_COMPUTE_32F_FAST_TF32` to cuBLAS.

### cuBLAS — `benchmark_cublas.cu`

**CUDA Core path** (`cublasSgemm` + `CUBLAS_PEDANTIC_MATH`):
```c
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);  // Disables TF32 down-casting
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, ...);
```

**Tensor Core path** (`cublasGemmEx` + `CUBLAS_COMPUTE_32F_FAST_TF32`):
```c
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
             &alpha,
             d_B, CUDA_R_32F, N,
             d_A, CUDA_R_32F, K,
             &beta,
             d_C, CUDA_R_32F, N,
             CUBLAS_COMPUTE_32F_FAST_TF32,       // TF32 Tensor Core compute
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);      // cuBLAS picks best algorithm
```

The column-major convention of cuBLAS is handled via the standard row-major transpose trick:  `C = A·B` (row-major) ↔ `Cᵀ = Bᵀ·Aᵀ` (column-major), issued as `gemm(N, M, K, B, N, A, K, C, N)` with `CUBLAS_OP_N` on both operands.

---

## Makefile Targets

```
make [ARCH=sm_XX]      Compile benchmark_cublas (default target)
make run               Compile + run cuBLAS benchmark
make pytorch_bench     Run PyTorch benchmark
make plot              Generate plots from existing result files
make all_bench         Run both benchmarks then plot
make clean             Remove build artefacts and result files
make help              Show full help message
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA not available` (Python) | Install a CUDA-enabled PyTorch build |
| `nvcc: command not found` | Add CUDA Toolkit `bin/` to `$PATH` |
| cuBLAS link error | Install full CUDA Toolkit (not just driver); add `-lcublas` |
| TF32 shows no speedup | Your GPU is older than SM80 (Ampere); TF32 is only accelerated on Ampere+ |
| `CUBLAS_PEDANTIC_MATH` undeclared | Upgrade to CUDA 11.0+ or use the provided `#ifndef` fallback |
| Out of memory | Reduce the largest sizes in the `SIZES` list |

---

## Running Without a Local NVIDIA GPU

No NVIDIA GPU? Use **Google Colab** — it's free and has CUDA + PyTorch pre-installed.

> **Note on TF32:** Free Colab gives a **T4 (SM75, Turing)** — TF32 Tensor Cores are not available on T4, so both modes will show similar speed. To observe the full 3–5× speedup you need **Colab Pro/Pro+** (A100, SM80) or any Ampere+ GPU (RTX 3090, RTX 4090, A100, H100).

---

### Step-by-step: Google Colab

**1. Open Colab and enable GPU**

Go to [colab.research.google.com](https://colab.research.google.com) → New notebook → `Runtime → Change runtime type → GPU`.

**2. Check your GPU**

```python
!nvidia-smi
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("SM:", torch.cuda.get_device_capability(0))  # need (8,0)+ for TF32
```

**3. Upload the project files**

Drag-and-drop all files into the Colab file browser (left sidebar → Files), or from a zip:

```python
!unzip gemm-tensorcore-benchmark.zip
%cd gemm-tensorcore-benchmark
```

**4. Install dependencies**

```python
!pip install numpy pandas matplotlib
import torch; print(torch.__version__, torch.version.cuda)  # PyTorch already in Colab
```

**5. Run the PyTorch benchmark**

```python
!python benchmark_pytorch.py --warmup 20 --iters 100
```

Expected output on A100:
```
CUDA Core (FP32)    B=1024  4096x4096   ~0.85ms   ~40 TFLOPS
Tensor Core (TF32)  B=1024  4096x4096   ~0.21ms  ~160 TFLOPS
```

**6. Compile and run the cuBLAS benchmark**

```python
!nvcc -O3 -arch=sm_80 -std=c++17 -o benchmark_cublas benchmark_cublas.cu -lcublas
!./benchmark_cublas
```

Use `-arch=sm_75` if you have a T4, `-arch=sm_89` for RTX 4090.

**7. Generate plots**

```python
import subprocess
gpu = subprocess.check_output(
    "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
).decode().strip()
!python plot_results.py --gpu "{gpu}" --outdir plots/
```

**8. View plots inside Colab**

```python
from IPython.display import Image, display
import glob
for f in sorted(glob.glob("plots/plot_*.png")):
    display(Image(f))
```

**9. Download plots**

Right-click any PNG in the Colab file browser sidebar → Download.

---

### What to Verify Once It Runs

**Check 1 — Speedup grows with matrix size**

```python
import json
with open("pytorch_results.json") as f:
    results = json.load(f)

print(f"{'Size':<28} {'CUDA TFLOPS':>12} {'TC TFLOPS':>12} {'Speedup':>10}")
print("-" * 66)
for b, k, n in [(64,1024,1024),(64,4096,4096),(256,4096,4096),(1024,4096,4096)]:
    cuda = next(r for r in results if r["batch_size"]==b and r["in_features"]==k and not r["use_tf32"])
    tc   = next(r for r in results if r["batch_size"]==b and r["in_features"]==k and r["use_tf32"])
    print(f"B={b:<4} {k}x{n:<20} {cuda['tflops']:>12.2f} {tc['tflops']:>12.2f} {cuda['latency_ms']/tc['latency_ms']:>9.2f}x")
```

Expected (A100): ~1.5× speedup at small sizes, ~4× at large sizes.

**Check 2 — TFLOPS formula is internally consistent**

```python
import json
with open("pytorch_results.json") as f:
    results = json.load(f)

all_pass = True
for r in results:
    product  = r["latency_ms"] * 1e-3 * r["tflops"] * 1e12
    expected = 2 * r["batch_size"] * r["in_features"] * r["out_features"]
    if abs(product - expected) / expected > 0.001:
        print(f"FAIL: {r['mode']} B={r['batch_size']} {r['in_features']}x{r['out_features']}")
        all_pass = False
print("ALL 22 ROWS PASS" if all_pass else "FAILURES DETECTED")
```

**Check 3 — TF32 produces a slightly different result (confirms it is active)**

```python
import torch, torch.nn as nn
torch.manual_seed(42)
layer = nn.Linear(4096, 4096, bias=False).cuda().float()
x = torch.randn(256, 4096, device="cuda")

torch.backends.cuda.matmul.allow_tf32 = False
out_fp32 = layer(x).clone()

torch.backends.cuda.matmul.allow_tf32 = True
out_tf32 = layer(x).clone()

rel_diff = (out_fp32 - out_tf32).abs().max() / out_fp32.abs().max() * 100
print(f"Relative diff: {rel_diff.item():.4f}%")
# Ampere+:    0.001–0.1%  → TF32 is active (mantissa truncated 23→10 bits)
# Pre-Ampere: 0.000%      → TF32 not engaged, both paths are identical FP32
```

**Check 4 — Plots look correct**

| Plot | What to expect |
|---|---|
| `plot_latency_tflops.png` | Orange bars (TC) shorter than blue (CUDA) at large sizes |
| `plot_speedup.png` | All bars green (> 1×), growing taller with matrix size |
| `plot_tflops_vs_size.png` | Orange line clearly above blue, gap widens at 4K |
| `plot_throughput_all.png` | Both lines rise with size; orange consistently higher |
