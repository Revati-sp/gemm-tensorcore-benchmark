/**
 * cuBLAS GEMM Benchmark
 * =====================
 * Compares two GEMM paths for a single FC-layer equivalent matmul
 * C(M×N) = A(M×K) × B(K×N)  where  M=batch, K=in_features, N=out_features.
 *
 * Mode 1 – CUDA Core baseline (FP32):
 *   cublasSgemm with CUBLAS_PEDANTIC_MATH, which disables TF32 down-casting
 *   and forces strict IEEE-754 FP32 arithmetic on CUDA cores.
 *
 * Mode 2 – Tensor Core path (TF32):
 *   cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 + CUBLAS_GEMM_DEFAULT_TENSOR_OP.
 *   cuBLAS truncates each FP32 operand's mantissa from 23 to 10 bits, routes
 *   the tiles through Tensor Cores, and accumulates in FP32.
 *   Requires Ampere (SM80+) for actual Tensor Core acceleration.
 *
 * Column-major note:
 *   cuBLAS is column-major. For row-major C arrays we exploit the identity:
 *     C = A · B  (row-major)
 *   ⟺  Cᵀ = Bᵀ · Aᵀ  (column-major)
 *   so we call cublasSgemm(N, M, K, B, N, A, K, C, N) with CUBLAS_OP_N on both.
 *
 * Benchmark protocol:
 *   - WARMUP warm-up calls (cuBLAS algorithm selection + GPU ramp-up).
 *   - ITERS  timed calls bracketed by cudaEventRecord.
 *   - Average latency = cudaEventElapsedTime / ITERS.
 *   - Throughput (TFLOPS) = 2·M·K·N / latency_s / 1e12.
 *
 * Requires: CUDA 11.0+, cuBLAS
 * Compile:  see Makefile  (nvcc -O3 -arch=sm_80 -lcublas)
 * Output:   cublas_results.csv
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ── CUBLAS_PEDANTIC_MATH guard ──────────────────────────────────────────────
 * Defined in cuBLAS ≥ 11.0. Provide a numeric fallback for older headers
 * that may ship the enum without this enumerator even on CUDA 11 builds.     */
#ifndef CUBLAS_PEDANTIC_MATH
#  define CUBLAS_PEDANTIC_MATH ((cublasMath_t)2)
#endif

/* ── Error-checking macros ───────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA]  %s  (file %s, line %d)\n",               \
                    cudaGetErrorString(_err), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "[cuBLAS] status %d  (file %s, line %d)\n",       \
                    (int)_s, __FILE__, __LINE__);                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

/* ── Warm initialisation kernel ─────────────────────────────────────────────
 * Fills the matrix buffer with small non-zero values using a simple LCG so
 * the GEMM doesn't degenerate into trivial zero-multiplications.             */
__global__ void fill_rand_kernel(float* __restrict__ arr, int n, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int state = (unsigned int)(idx) * 1664525u + 1013904223u;
    state = state * 1664525u + 1013904223u;
    arr[idx] = (float)(state & 0xFFFF) / 32768.0f * scale - scale * 0.5f;
}

static void fill_rand(float* d_arr, int n, float scale = 0.01f)
{
    const int threads = 256;
    fill_rand_kernel<<<(n + threads - 1) / threads, threads>>>(d_arr, n, scale);
    CUDA_CHECK(cudaGetLastError());
}

/* ── GEMM mode tag ───────────────────────────────────────────────────────── */
enum GemmMode { CUDA_CORE_FP32, TENSOR_CORE_TF32 };

/* ── Core benchmark routine ──────────────────────────────────────────────────
 * Performs GEMM C(M×N) = A(M×K) × B(K×N) in either mode.
 * Returns average latency in milliseconds over `iters` iterations.          */
static double run_benchmark(
    cublasHandle_t  handle,
    int             M, int K, int N,
    const float*    d_A,   /* device, M×K row-major */
    const float*    d_B,   /* device, K×N row-major */
    float*          d_C,   /* device, M×N row-major */
    GemmMode        mode,
    int             warmup,
    int             iters)
{
    const float alpha = 1.0f, beta = 0.0f;

    /* Configure math mode for this handle before any GEMM call. */
    if (mode == CUDA_CORE_FP32) {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    } else {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    /* Lambda that issues one GEMM using the column-major transpose trick. */
    auto do_gemm = [&]() {
        if (mode == CUDA_CORE_FP32) {
            /* cublasSgemm: strict FP32, no Tensor Core usage. */
            CUBLAS_CHECK(cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,          /* leading-dim sizes after transposition */
                &alpha,
                d_B, N,           /* Bᵀ in column-major = B (N×K) */
                d_A, K,           /* Aᵀ in column-major = A (K×M) */
                &beta,
                d_C, N            /* Cᵀ = result (N×M) = C row-major */
            ));
        } else {
            /* cublasGemmEx: FP32 I/O, TF32 Tensor Core compute. */
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, CUDA_R_32F, N,
                d_A, CUDA_R_32F, K,
                &beta,
                d_C, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_TF32,   /* TF32 Tensor Core compute */
                CUBLAS_GEMM_DEFAULT_TENSOR_OP   /* let cuBLAS pick best algo  */
            ));
        }
    };

    /* ── Warm-up ─────────────────────────────────────────────────────────── */
    for (int i = 0; i < warmup; i++) do_gemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Timed benchmark ─────────────────────────────────────────────────── */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; i++) do_gemm();
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return (double)elapsed_ms / iters;
}

/* ── Benchmark configurations ───────────────────────────────────────────── */
struct BenchSize { int batch, in_feat, out_feat; };

static const BenchSize SIZES[] = {
    { 64,   1024, 1024 },
    { 64,   2048, 2048 },
    { 64,   4096, 4096 },
    { 128,  1024, 1024 },
    { 128,  2048, 2048 },
    { 128,  4096, 4096 },
    { 256,  1024, 1024 },
    { 256,  2048, 2048 },
    { 256,  4096, 4096 },
    { 512,  4096, 4096 },
    { 1024, 4096, 4096 },
};
static const int NUM_SIZES = (int)(sizeof(SIZES) / sizeof(SIZES[0]));

static const int WARMUP = 20;
static const int ITERS  = 100;

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void)
{
    /* Print device info. */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("================================================================================\n");
    printf("  cuBLAS GEMM Benchmark  –  CUDA Core FP32 vs Tensor Core TF32\n");
    printf("================================================================================\n");
    printf("  GPU   : %s  (SM %d%d)\n", prop.name, prop.major, prop.minor);
    printf("  CUDA  : %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    if (prop.major < 8) {
        printf("  [WARNING] TF32 Tensor Cores require Ampere (SM80+).\n"
               "            The TF32 path will not be faster on this GPU.\n");
    }
    printf("  Warm-up: %d   Iters: %d\n", WARMUP, ITERS);
    printf("================================================================================\n\n");

    /* Create cuBLAS handle (one per thread is the recommended pattern). */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    /* Print table header. */
    printf("%-27s %6s %8s %8s %14s %10s\n",
           "Mode", "Batch", "InFeat", "OutFeat", "Latency(ms)", "TFLOPS");
    printf("%-87s\n",
           "---------------------------------------------------------------------------------"
           "--------");

    /* Allocate device buffers large enough for the biggest configuration. */
    const int MAX_M = 1024, MAX_K = 4096, MAX_N = 4096;
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)MAX_M * MAX_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)MAX_K * MAX_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)MAX_M * MAX_N * sizeof(float)));

    fill_rand(d_A, MAX_M * MAX_K);
    fill_rand(d_B, MAX_K * MAX_N);
    CUDA_CHECK(cudaMemset(d_C, 0, (size_t)MAX_M * MAX_N * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Open CSV output file. */
    FILE* csv = fopen("cublas_results.csv", "w");
    if (!csv) {
        fprintf(stderr, "Cannot open cublas_results.csv for writing\n");
        return EXIT_FAILURE;
    }
    fprintf(csv, "mode,batch_size,in_features,out_features,latency_ms,tflops\n");

    /* ── Size/mode sweep ──────────────────────────────────────────────────── */
    for (int s = 0; s < NUM_SIZES; s++) {
        const int M = SIZES[s].batch;
        const int K = SIZES[s].in_feat;
        const int N = SIZES[s].out_feat;

        for (int m = 0; m < 2; m++) {
            GemmMode    mode = (m == 0) ? CUDA_CORE_FP32  : TENSOR_CORE_TF32;
            const char* name = (m == 0) ? "CUDA Core (FP32)" : "Tensor Core (TF32)";

            double lat   = run_benchmark(handle, M, K, N, d_A, d_B, d_C,
                                         mode, WARMUP, ITERS);
            double flops  = 2.0 * M * K * N;
            double tflops = (flops / (lat * 1e-3)) / 1e12;

            printf("%-27s %6d %8d %8d %14.4f %10.3f\n", name, M, K, N, lat, tflops);
            fprintf(csv, "%s,%d,%d,%d,%.6f,%.6f\n", name, M, K, N, lat, tflops);
        }
    }

    fclose(csv);
    printf("\n  Results saved → cublas_results.csv\n");

    /* ── Cleanup ──────────────────────────────────────────────────────────── */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));

    return EXIT_SUCCESS;
}
