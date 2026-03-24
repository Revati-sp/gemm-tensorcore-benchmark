#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_all.sh  –  End-to-end benchmark runner
# ──────────────────────────────────────────────────────────────────────────────
# Runs both benchmarks (PyTorch + cuBLAS), then generates all comparison plots.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh [--arch sm_80] [--warmup 20] [--iters 100] [--outdir results/]
#
# Options:
#   --arch    CUDA SM architecture for NVCC (default: sm_80)
#   --warmup  Warm-up iterations           (default: 20)
#   --iters   Timed iterations             (default: 100)
#   --outdir  Directory for plots          (default: .)
#   --skip-cublas   Skip cuBLAS benchmark  (use if no nvcc available)
#   --skip-pytorch  Skip PyTorch benchmark
#   --skip-plot     Skip plot generation
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ARCH="sm_80"
WARMUP=20
ITERS=100
OUTDIR="."
SKIP_CUBLAS=false
SKIP_PYTORCH=false
SKIP_PLOT=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)         ARCH="$2";     shift 2 ;;
        --warmup)       WARMUP="$2";   shift 2 ;;
        --iters)        ITERS="$2";    shift 2 ;;
        --outdir)       OUTDIR="$2";   shift 2 ;;
        --skip-cublas)  SKIP_CUBLAS=true;  shift ;;
        --skip-pytorch) SKIP_PYTORCH=true; shift ;;
        --skip-plot)    SKIP_PLOT=true;    shift ;;
        -h|--help)
            sed -n '3,30p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  GEMM Tensor Core Benchmark Suite"
echo "============================================================"
echo "  ARCH    : $ARCH"
echo "  WARMUP  : $WARMUP"
echo "  ITERS   : $ITERS"
echo "  OUTDIR  : $OUTDIR"
echo "============================================================"
echo ""

mkdir -p "$OUTDIR"

# ── 1. Detect GPU ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    echo "  GPU detected: $GPU_NAME"
else
    GPU_NAME="GPU"
    echo "  [WARNING] nvidia-smi not found. GPU name unavailable."
fi
echo ""

# ── 2. PyTorch benchmark ──────────────────────────────────────────────────────
if [[ "$SKIP_PYTORCH" == false ]]; then
    echo "------------------------------------------------------------"
    echo "  Step 1/3: PyTorch FC-Layer Benchmark"
    echo "------------------------------------------------------------"
    python3 benchmark_pytorch.py --warmup "$WARMUP" --iters "$ITERS" \
            --output pytorch_results.json
    echo ""
else
    echo "  [SKIP] PyTorch benchmark skipped."
    echo ""
fi

# ── 3. cuBLAS benchmark ───────────────────────────────────────────────────────
if [[ "$SKIP_CUBLAS" == false ]]; then
    echo "------------------------------------------------------------"
    echo "  Step 2/3: cuBLAS Benchmark"
    echo "------------------------------------------------------------"

    if ! command -v nvcc &>/dev/null; then
        echo "  [ERROR] nvcc not found. Install the CUDA Toolkit or use --skip-cublas."
        exit 1
    fi

    echo "  Compiling benchmark_cublas.cu  (arch=$ARCH) …"
    nvcc -O3 -arch="$ARCH" -std=c++17 --expt-relaxed-constexpr \
         -Xcompiler -Wall \
         -o benchmark_cublas benchmark_cublas.cu -lcublas
    echo "  Compilation successful."
    echo ""

    echo "  Running cuBLAS benchmark …"
    ./benchmark_cublas
    echo ""
else
    echo "  [SKIP] cuBLAS benchmark skipped."
    echo ""
fi

# ── 4. Generate plots ─────────────────────────────────────────────────────────
if [[ "$SKIP_PLOT" == false ]]; then
    echo "------------------------------------------------------------"
    echo "  Step 3/3: Generating Plots"
    echo "------------------------------------------------------------"

    PLOT_ARGS=""
    [[ -f pytorch_results.json ]] && PLOT_ARGS="$PLOT_ARGS --pytorch pytorch_results.json"
    [[ -f cublas_results.csv  ]] && PLOT_ARGS="$PLOT_ARGS --cublas  cublas_results.csv"

    if [[ -z "$PLOT_ARGS" ]]; then
        echo "  [WARNING] No result files found. Skipping plot generation."
    else
        python3 plot_results.py $PLOT_ARGS --gpu "$GPU_NAME" --outdir "$OUTDIR"
    fi
else
    echo "  [SKIP] Plot generation skipped."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All done! Output files:"
echo "============================================================"
[[ -f pytorch_results.json ]] && echo "  pytorch_results.json"
[[ -f cublas_results.csv  ]] && echo "  cublas_results.csv"
for f in "$OUTDIR"/plot_*.png; do
    [[ -f "$f" ]] && echo "  $f"
done
echo ""
