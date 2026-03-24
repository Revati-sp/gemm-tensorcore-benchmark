#!/usr/bin/env python3
"""
GEMM Benchmark Visualiser
==========================
Reads PyTorch (JSON) and / or cuBLAS (CSV) result files and produces four
publication-quality figures:

  1. plot_latency_tflops.png  – Side-by-side grouped bar charts of avg latency
                                and TFLOPS for every (batch, M, N) configuration.
  2. plot_speedup.png         – Speedup of Tensor Core (TF32) over CUDA Core (FP32)
                                per configuration.
  3. plot_tflops_vs_size.png  – TFLOPS vs matrix dimension (line chart) averaged
                                over all batch sizes for each mode.
  4. plot_throughput_all.png  – TFLOPS line chart for every configuration showing
                                both frameworks side-by-side (only when both
                                result files are present).

Usage:
  python plot_results.py
  python plot_results.py --pytorch pytorch_results.json --cublas cublas_results.csv
  python plot_results.py --outdir results/  --gpu "NVIDIA A100 80GB"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ── Colour palette & global style ─────────────────────────────────────────────
CUDA_COLOR   = "#3A7FBD"    # steel blue  – CUDA Core (FP32)
TC_COLOR     = "#E07B39"    # burnt orange – Tensor Core (TF32)
SPEEDUP_POS  = "#2CA02C"    # green  – speedup > 1
SPEEDUP_NEG  = "#D62728"    # red    – speedup < 1 (regression)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "#DDDDDD",
    "grid.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   8,
    "figure.dpi":        150,
})

FIG_DPI = 150


# ── Helpers ────────────────────────────────────────────────────────────────────

def k_str(n: int) -> str:
    """Format an integer as 'XK' if divisible by 1024, otherwise as-is."""
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def make_label(row) -> str:
    return f"B{row['batch_size']}\n{k_str(row['in_features'])}×{k_str(row['out_features'])}"


def split_modes(df: pd.DataFrame):
    """Split a result DataFrame into (cuda_df, tc_df) subsets."""
    cuda = df[df["mode"].str.contains("CUDA")].reset_index(drop=True)
    tc   = df[df["mode"].str.contains("Tensor")].reset_index(drop=True)
    return cuda, tc


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_pytorch(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["label"] = df.apply(make_label, axis=1)
    return df


def load_cublas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df.apply(make_label, axis=1)
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _annotate_bars(ax, bars, fmt="{:.2f}", rot=90, fontsize=6):
    """Place value labels above each bar."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h * 1.015,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, rotation=rot,
        )


def plot_grouped_bars(ax, df: pd.DataFrame, metric: str, ylabel: str, title: str):
    """
    Grouped bar chart: one group per size configuration.
    Each group has two bars – CUDA Core (blue) and Tensor Core (orange).
    """
    cuda_df, tc_df = split_modes(df)
    labels = cuda_df["label"].tolist()
    x = np.arange(len(labels))
    w = 0.36

    bars_cuda = ax.bar(
        x - w / 2, cuda_df[metric].values, w,
        label="CUDA Core (FP32)", color=CUDA_COLOR, alpha=0.88,
        edgecolor="white", linewidth=0.6, zorder=3,
    )
    bars_tc = ax.bar(
        x + w / 2, tc_df[metric].values, w,
        label="Tensor Core (TF32)", color=TC_COLOR, alpha=0.88,
        edgecolor="white", linewidth=0.6, zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.85)

    # Annotate bars with their numeric value.
    fmt = "{:.3f}" if metric == "latency_ms" else "{:.1f}"
    _annotate_bars(ax, list(bars_cuda) + list(bars_tc), fmt=fmt)

    peak = max(cuda_df[metric].max(), tc_df[metric].max())
    ax.set_ylim(0, peak * 1.35)


def plot_speedup_bars(ax, df: pd.DataFrame, title: str):
    """
    Horizontal/vertical bar chart of Tensor-Core speedup over CUDA Core.
    Bars coloured green for speedup > 1, red for speedup < 1.
    """
    cuda_df, tc_df = split_modes(df)
    speedup = (cuda_df["latency_ms"].values / tc_df["latency_ms"].values)
    labels  = cuda_df["label"].tolist()
    x       = np.arange(len(labels))

    colors = [SPEEDUP_POS if s >= 1.0 else SPEEDUP_NEG for s in speedup]
    bars   = ax.bar(x, speedup, color=colors, alpha=0.85,
                    edgecolor="white", linewidth=0.6, zorder=3)

    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.2,
               label="Baseline (1×)", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup (Tensor Core / CUDA Core)  [×]")
    ax.set_title(title)
    ax.legend(framealpha=0.85)

    for bar, s in zip(bars, speedup):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03,
            f"{s:.2f}×",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    ax.set_ylim(0, max(speedup) * 1.25 + 0.4)


def plot_tflops_line(ax, df: pd.DataFrame, title: str):
    """
    Line chart: average TFLOPS vs matrix dimension (square matrices only),
    collapsed over all batch sizes.
    """
    sq = df[df["in_features"] == df["out_features"]].copy()
    sq["dim"] = sq["in_features"]

    for mode, color, marker in [
        ("CUDA Core (FP32)",   CUDA_COLOR, "o"),
        ("Tensor Core (TF32)", TC_COLOR,   "s"),
    ]:
        sub = (sq[sq["mode"] == mode]
               .groupby("dim", sort=True)["tflops"]
               .mean()
               .reset_index())
        ax.plot(sub["dim"], sub["tflops"],
                label=mode, color=color, marker=marker,
                linewidth=2.2, markersize=7, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(f"{row['tflops']:.1f}",
                        xy=(row["dim"], row["tflops"]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color=color)

    ax.set_xlabel("Matrix Dimension  (square N×N)")
    ax.set_ylabel("Avg TFLOPS  (mean over all batch sizes)")
    ax.set_title(title)
    ax.set_xticks(sorted(sq["dim"].unique()))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: k_str(int(v))))
    ax.legend(framealpha=0.85)


# ── Figure factories ───────────────────────────────────────────────────────────

def fig_latency_tflops(pytorch_df, cublas_df, gpu_name: str) -> plt.Figure:
    """
    2-row × 2-col grid:
      Row 0: PyTorch  – Latency | TFLOPS
      Row 1: cuBLAS   – Latency | TFLOPS
    (rows skipped if the corresponding result file is absent)
    """
    rows_data = [(d, fw) for d, fw in
                 [(pytorch_df, "PyTorch"), (cublas_df, "cuBLAS")] if d is not None]
    n_rows = len(rows_data)

    fig = plt.figure(figsize=(20, 6.5 * n_rows))
    gs  = GridSpec(n_rows, 2, figure=fig, hspace=0.55, wspace=0.30)

    for row_idx, (df, fw) in enumerate(rows_data):
        ax_lat = fig.add_subplot(gs[row_idx, 0])
        ax_tfl = fig.add_subplot(gs[row_idx, 1])
        plot_grouped_bars(ax_lat, df, "latency_ms", "Avg Latency (ms)",
                          f"{fw} — Avg Latency per Forward Pass")
        plot_grouped_bars(ax_tfl, df, "tflops",     "TFLOPS",
                          f"{fw} — Throughput (TFLOPS)")

    fig.suptitle(
        f"GEMM Benchmark: CUDA Core (FP32) vs Tensor Core (TF32)\n{gpu_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    return fig


def fig_speedup(pytorch_df, cublas_df, gpu_name: str) -> plt.Figure:
    """One speedup bar chart per available framework."""
    cols_data = [(d, fw) for d, fw in
                 [(pytorch_df, "PyTorch"), (cublas_df, "cuBLAS")] if d is not None]
    n_cols = len(cols_data)

    fig, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    for ax, (df, fw) in zip(axes, cols_data):
        plot_speedup_bars(ax, df, f"{fw} — Tensor Core Speedup over CUDA Core")

    legend_patches = [
        Patch(color=SPEEDUP_POS, label="Speedup > 1× (faster)"),
        Patch(color=SPEEDUP_NEG, label="Speedup < 1× (slower)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, framealpha=0.85, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        f"Speedup: Tensor Core TF32 vs CUDA Core FP32\n{gpu_name}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def fig_tflops_vs_size(pytorch_df, cublas_df, gpu_name: str) -> plt.Figure:
    """TFLOPS line chart vs matrix dimension, one panel per framework."""
    cols_data = [(d, fw) for d, fw in
                 [(pytorch_df, "PyTorch"), (cublas_df, "cuBLAS")] if d is not None]
    n_cols = len(cols_data)

    fig, axes = plt.subplots(1, n_cols, figsize=(8.5 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    for ax, (df, fw) in zip(axes, cols_data):
        plot_tflops_line(ax, df, f"{fw} — TFLOPS vs Matrix Dimension")

    fig.suptitle(
        f"Throughput vs Matrix Size  (averaged over all batch sizes)\n{gpu_name}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def fig_throughput_all(pytorch_df: pd.DataFrame, cublas_df: pd.DataFrame,
                       gpu_name: str) -> plt.Figure:
    """
    Side-by-side line charts of TFLOPS across every configuration for both
    frameworks (only generated when both result files are available).
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.5))

    for ax, (df, fw) in zip(axes, [(pytorch_df, "PyTorch"), (cublas_df, "cuBLAS")]):
        cuda_df, tc_df = split_modes(df)
        x_vals  = np.arange(len(cuda_df))
        x_labels = cuda_df["label"].tolist()

        ax.plot(x_vals, cuda_df["tflops"].values, "o-",
                color=CUDA_COLOR, label="CUDA Core (FP32)",
                linewidth=2, markersize=5, zorder=3)
        ax.fill_between(x_vals, 0, cuda_df["tflops"].values,
                         alpha=0.08, color=CUDA_COLOR)

        ax.plot(x_vals, tc_df["tflops"].values, "s-",
                color=TC_COLOR, label="Tensor Core (TF32)",
                linewidth=2, markersize=5, zorder=3)
        ax.fill_between(x_vals, 0, tc_df["tflops"].values,
                         alpha=0.08, color=TC_COLOR)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylabel("TFLOPS")
        ax.set_title(f"{fw} — Throughput per Configuration")
        ax.legend(framealpha=0.85)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        f"Throughput Across All Configurations\n{gpu_name}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(pytorch_df, cublas_df):
    for df, fw in [(pytorch_df, "PyTorch"), (cublas_df, "cuBLAS")]:
        if df is None:
            continue
        cuda_df, tc_df = split_modes(df)
        speedups = cuda_df["latency_ms"].values / tc_df["latency_ms"].values
        print(f"\n  {fw} Speedup Statistics")
        print(f"  {'─' * 40}")
        print(f"  Mean speedup  : {speedups.mean():.2f}×")
        print(f"  Max  speedup  : {speedups.max():.2f}×  "
              f"(B={cuda_df.iloc[speedups.argmax()]['batch_size']}, "
              f"{k_str(cuda_df.iloc[speedups.argmax()]['in_features'])}×"
              f"{k_str(cuda_df.iloc[speedups.argmax()]['out_features'])})")
        print(f"  Min  speedup  : {speedups.min():.2f}×")
        print(f"  Peak TC TFLOPS: {tc_df['tflops'].max():.2f}")
        print(f"  Peak FP32 TFLOPS: {cuda_df['tflops'].max():.2f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise GEMM benchmark results (PyTorch + cuBLAS)"
    )
    p.add_argument("--pytorch", default="pytorch_results.json",
                   help="Path to PyTorch JSON results (default: pytorch_results.json)")
    p.add_argument("--cublas",  default="cublas_results.csv",
                   help="Path to cuBLAS CSV results  (default: cublas_results.csv)")
    p.add_argument("--gpu",     default="",
                   help="GPU name to display in plot titles")
    p.add_argument("--outdir",  default=".",
                   help="Directory to write PNG files (default: current dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pytorch_df: Optional[pd.DataFrame] = None
    cublas_df:  Optional[pd.DataFrame] = None

    if Path(args.pytorch).exists():
        pytorch_df = load_pytorch(args.pytorch)
        print(f"  Loaded PyTorch results : {len(pytorch_df)} rows  ({args.pytorch})")
    else:
        print(f"  PyTorch results not found : {args.pytorch}")

    if Path(args.cublas).exists():
        cublas_df = load_cublas(args.cublas)
        print(f"  Loaded cuBLAS results  : {len(cublas_df)} rows  ({args.cublas})")
    else:
        print(f"  cuBLAS results not found  : {args.cublas}")

    if pytorch_df is None and cublas_df is None:
        print("\n  No result files found. Run the benchmarks first, then re-run this script.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    gpu_name = args.gpu or "GPU"

    saved = []

    # Figure 1 – Latency & TFLOPS grouped bar charts
    fig1 = fig_latency_tflops(pytorch_df, cublas_df, gpu_name)
    p1 = os.path.join(args.outdir, "plot_latency_tflops.png")
    fig1.savefig(p1, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig1)
    saved.append(p1)

    # Figure 2 – Speedup bar chart
    fig2 = fig_speedup(pytorch_df, cublas_df, gpu_name)
    p2 = os.path.join(args.outdir, "plot_speedup.png")
    fig2.savefig(p2, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig2)
    saved.append(p2)

    # Figure 3 – TFLOPS vs matrix dimension
    fig3 = fig_tflops_vs_size(pytorch_df, cublas_df, gpu_name)
    p3 = os.path.join(args.outdir, "plot_tflops_vs_size.png")
    fig3.savefig(p3, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig3)
    saved.append(p3)

    # Figure 4 – Full throughput overview (only when both datasets available)
    if pytorch_df is not None and cublas_df is not None:
        fig4 = fig_throughput_all(pytorch_df, cublas_df, gpu_name)
        p4 = os.path.join(args.outdir, "plot_throughput_all.png")
        fig4.savefig(p4, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig4)
        saved.append(p4)

    print_summary(pytorch_df, cublas_df)

    print("\n  Plots saved:")
    for p in saved:
        print(f"    {p}")
    print()


if __name__ == "__main__":
    main()
