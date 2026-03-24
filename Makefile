# ──────────────────────────────────────────────────────────────────────────────
# Makefile  –  cuBLAS GEMM Benchmark
# ──────────────────────────────────────────────────────────────────────────────
# Targets:
#   make            – compile benchmark_cublas
#   make run        – compile and run the cuBLAS benchmark
#   make all_bench  – run both PyTorch and cuBLAS benchmarks, then plot
#   make plot       – generate plots from existing result files
#   make clean      – remove build artefacts and result CSVs
#
# Configurable variables (override on the command line):
#   ARCH   – CUDA architecture (default sm_80 for Ampere / A100 / RTX 3090)
#             Use sm_86 for RTX 3080 Ti / A40, sm_89 for RTX 4090, sm_90 for H100.
#             Examples:
#               make ARCH=sm_86      # Laptop RTX 3080 Ti
#               make ARCH=sm_89      # RTX 4090 (Ada Lovelace)
#               make ARCH=sm_90      # H100 (Hopper)
#   WARMUP – warm-up iterations (default 20)
#   ITERS  – timed iterations   (default 100)
# ──────────────────────────────────────────────────────────────────────────────

NVCC      ?= nvcc
ARCH      ?= sm_80
WARMUP    ?= 20
ITERS     ?= 100

TARGET    := benchmark_cublas
SRC       := benchmark_cublas.cu

# Compiler flags
NVCC_FLAGS  := -O3 -arch=$(ARCH) -std=c++17
NVCC_FLAGS  += -Xcompiler -Wall,-Wextra
NVCC_FLAGS  += --expt-relaxed-constexpr
LINK_FLAGS  := -lcublas

# Python environment (override to point at a virtualenv)
PYTHON    ?= python3

# ── Build ─────────────────────────────────────────────────────────────────────
.PHONY: all build run all_bench plot clean help

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	@echo "  [NVCC]  Compiling $< → $@  (arch=$(ARCH))"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LINK_FLAGS)
	@echo "  [OK]    Build successful: ./$(TARGET)"

# ── Run targets ───────────────────────────────────────────────────────────────
run: build
	@echo ""
	@echo "  Running cuBLAS benchmark …"
	./$(TARGET)

pytorch_bench:
	@echo ""
	@echo "  Running PyTorch benchmark …"
	$(PYTHON) benchmark_pytorch.py --warmup $(WARMUP) --iters $(ITERS)

plot:
	@echo ""
	@echo "  Generating plots …"
	$(PYTHON) plot_results.py

all_bench: pytorch_bench run plot
	@echo ""
	@echo "  All benchmarks complete. Check the PNG files for plots."

# ── Utility ───────────────────────────────────────────────────────────────────
clean:
	@echo "  Removing build artefacts …"
	rm -f $(TARGET) *.o
	rm -f cublas_results.csv pytorch_results.json
	rm -f plot_*.png

help:
	@echo ""
	@echo "  Usage:"
	@echo "    make [ARCH=sm_XX] [WARMUP=N] [ITERS=N]  <target>"
	@echo ""
	@echo "  Targets:"
	@echo "    build        – compile benchmark_cublas (default)"
	@echo "    run          – compile + run cuBLAS benchmark"
	@echo "    pytorch_bench– run PyTorch benchmark"
	@echo "    plot         – plot existing result files"
	@echo "    all_bench    – run both benchmarks then plot"
	@echo "    clean        – remove build and result files"
	@echo ""
	@echo "  Architecture presets:"
	@echo "    sm_80  Ampere: A100, RTX 3090, A30   (default)"
	@echo "    sm_86  Ampere: RTX 3080 Ti, A40, A10"
	@echo "    sm_89  Ada:    RTX 4090, RTX 4080"
	@echo "    sm_90  Hopper: H100"
	@echo ""
