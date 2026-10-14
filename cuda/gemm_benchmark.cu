
//
// Square single-matrix multiply C = A * B (row-major, N x N) on one GPU.
//
//   cpu_ikj_1thread : single-threaded CPU reference (cache-friendly i-k-j loop order)
//   naive           : one thread per output element, every operand read from global memory
//   tiled           : 32x32 shared-memory tiles, one thread per output element
//   regtiled        : register-tiled: 64x64 block tile, 4x4 outputs per thread
//   regtiled_vec    : regtiled + transposed padded A tile + float4 shared loads
//   cublas_fp32     : cuBLAS, pure FP32 math (no tensor cores)
//   cublas_tf32     : cuBLAS, TF32 tensor cores (Ampere or newer)
//   cublas_fp16     : cuBLAS, FP16 inputs/outputs, FP32 accumulate (tensor cores)
//
// Measurement rules
//   * GPU kernels: 3 warm-up launches, then timed repetitions with CUDA events; the
//     median and the minimum are reported.
//   * CPU: wall clock (steady_clock). Repetitions shrink as N grows (see cpu_reps).
//   * Host<->device transfers are timed separately, from pageable and from pinned memory.
//   * Every implementation is checked against the CPU result (max abs error / max |C|).
//
// Usage:  gemm_benchmark.exe [peak_fp32_tflops] [csv_path]
//   peak_fp32_tflops : optional reference peak used for the "% of peak" column
//   csv_path         : default results/gemm_results.csv (the folder must already exist)

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err__));                                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__,          \
              __LINE__, (int)st__);                                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
constexpr int NAIVE_BLOCK = 16;  // naive kernel: 16x16 = 256 threads per block
constexpr int TILE = 32;         // tiled kernel: 32x32 = 1024 threads per block

// Naive: thread (tx, ty) of block (bx, by) computes C[row][col] with
//   row = by * blockDim.y + ty,  col = bx * blockDim.x + tx.
// The grid is ceil(n/16) x ceil(n/16) blocks, so every output element has exactly
// one thread. Each thread reads a full row of A and a full column of B from global
// memory: 2n loads for n multiply-adds, no data reuse between threads.
__global__ void gemm_naive(const float* A, const float* B, float* C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
      acc += A[(size_t)row * n + k] * B[(size_t)k * n + col];
    }
    C[(size_t)row * n + col] = acc;
  }
}

// Tiled: a 32x32 block computes a 32x32 tile of C. It walks along K in steps of 32.
// At each step the 1024 threads cooperatively copy one 32x32 tile of A and one of B
// into shared memory (one element per thread), synchronize, then every thread does 32
// multiply-adds out of shared memory. Each global value is loaded once per block
// instead of once per thread, so global traffic drops by about 32x.
__global__ void gemm_tiled(const float* A, const float* B, float* C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  int num_tiles = (n + TILE - 1) / TILE;
  for (int t = 0; t < num_tiles; ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < n && a_col < n) ? A[(size_t)row * n + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < n && col < n) ? B[(size_t)b_row * n + col] : 0.0f;
    __syncthreads();  // tile fully loaded before anyone reads it

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();  // everyone done reading before the next tile overwrites it
  }
  if (row < n && col < n) C[(size_t)row * n + col] = acc;
}

// Register-tiled: a 16x16 block (256 threads) computes a 64x64 tile of C and each
// thread computes a 4x4 block of outputs, keeping its 16 running sums in registers.
// Per k step a thread loads 4 values of A and 4 of B from shared memory (8 loads) and
// does 16 multiply-adds: 0.5 shared loads per multiply-add instead of 2 in gemm_tiled.
// A thread's rows are ty, ty+16, ty+32, ty+48 and its columns tx, tx+16, tx+32, tx+48,
// so neighbouring threads touch neighbouring shared-memory addresses (no bank
// conflicts). Out-of-range loads read 0 and out-of-range stores are skipped, so any n works.
constexpr int RT_BM = 64;   // rows of C per block
constexpr int RT_BN = 64;   // columns of C per block
constexpr int RT_BK = 16;   // K values per step
constexpr int RT_TM = 4;    // rows of C per thread
constexpr int RT_TN = 4;    // columns of C per thread
constexpr int RT_DIM = 16;  // threads per block side (16 x 16 = 256 threads)

__global__ void __launch_bounds__(RT_DIM * RT_DIM)
gemm_regtiled(const float* A, const float* B, float* C, int n) {
  __shared__ float As[RT_BM][RT_BK];
  __shared__ float Bs[RT_BK][RT_BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * RT_DIM + tx;
  const int block_row = blockIdx.y * RT_BM;
  const int block_col = blockIdx.x * RT_BN;

  float acc[RT_TM][RT_TN];
#pragma unroll
  for (int i = 0; i < RT_TM; ++i)
#pragma unroll
    for (int j = 0; j < RT_TN; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < n; k0 += RT_BK) {
    // A tile (64 x 16 = 1024 values, 4 per thread); consecutive threads read
    // consecutive k, so global reads are coalesced.
#pragma unroll
    for (int s = 0; s < (RT_BM * RT_BK) / (RT_DIM * RT_DIM); ++s) {
      const int idx = tid + s * RT_DIM * RT_DIM;
      const int m = idx / RT_BK;
      const int k = idx % RT_BK;
      const int gr = block_row + m;
      const int gc = k0 + k;
      As[m][k] = (gr < n && gc < n) ? A[(size_t)gr * n + gc] : 0.0f;
    }
    // B tile (16 x 64 = 1024 values, 4 per thread).
#pragma unroll
    for (int s = 0; s < (RT_BK * RT_BN) / (RT_DIM * RT_DIM); ++s) {
      const int idx = tid + s * RT_DIM * RT_DIM;
      const int k = idx / RT_BN;
      const int c = idx % RT_BN;
      const int gr = k0 + k;
      const int gc = block_col + c;
      Bs[k][c] = (gr < n && gc < n) ? B[(size_t)gr * n + gc] : 0.0f;
    }
    __syncthreads();  // both tiles fully loaded before anyone reads them

#pragma unroll
    for (int k = 0; k < RT_BK; ++k) {
      float a[RT_TM];
      float b[RT_TN];
#pragma unroll
      for (int i = 0; i < RT_TM; ++i) a[i] = As[ty + i * RT_DIM][k];
#pragma unroll
      for (int j = 0; j < RT_TN; ++j) b[j] = Bs[k][tx + j * RT_DIM];
#pragma unroll
      for (int i = 0; i < RT_TM; ++i)
#pragma unroll
        for (int j = 0; j < RT_TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();  // everyone done reading before the next tiles overwrite them
  }

#pragma unroll
  for (int i = 0; i < RT_TM; ++i) {
    const int r = block_row + ty + i * RT_DIM;
#pragma unroll
    for (int j = 0; j < RT_TN; ++j) {
      const int c = block_col + tx + j * RT_DIM;
      if (r < n && c < n) C[(size_t)r * n + c] = acc[i][j];
    }
  }
}

// ---------------------------------------------------------------------------
// A5 change 1: vectorized shared-memory loads.
//
// Same 64x64 block tile, 16x16 threads, BK = 16, 4x4 outputs per thread as
// gemm_regtiled. Two things change:
//   (a) the A tile is stored TRANSPOSED in shared memory (As[k][m]) with 4 floats of
//       padding per row, so a thread's 4 A values sit next to each other and one
//       128-bit load reads them
//   (b) a thread owns a contiguous 4x4 block (rows ty*4..ty*4+3, cols tx*4..tx*4+3)
//       instead of rows ty, ty+16, ty+32, ty+48, so its 4 B values are contiguous too
// Per k step: 2 float4 shared loads instead of 8 scalar loads, same 16 multiply-adds.
// ---------------------------------------------------------------------------
constexpr int RV_LDA = RT_BM + 4;  // padded row length (floats) of the transposed A tile

__global__ void __launch_bounds__(RT_DIM * RT_DIM, 4)
gemm_regtiled_vec(const float* A, const float* B, float* C, int n) {
  __shared__ __align__(16) float As[RT_BK][RV_LDA];  // transposed: As[k][m]
  __shared__ __align__(16) float Bs[RT_BK][RT_BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * RT_DIM + tx;
  const int block_row = blockIdx.y * RT_BM;
  const int block_col = blockIdx.x * RT_BN;

  float acc[RT_TM][RT_TN];
#pragma unroll
  for (int i = 0; i < RT_TM; ++i)
#pragma unroll
    for (int j = 0; j < RT_TN; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < n; k0 += RT_BK) {
    // A tile (64 x 16): global reads stay coalesced, the shared store is transposed.
#pragma unroll
    for (int s = 0; s < (RT_BM * RT_BK) / (RT_DIM * RT_DIM); ++s) {
      const int idx = tid + s * RT_DIM * RT_DIM;
      const int m = idx / RT_BK;
      const int k = idx % RT_BK;
      const int gr = block_row + m;
      const int gc = k0 + k;
      As[k][m] = (gr < n && gc < n) ? A[(size_t)gr * n + gc] : 0.0f;
    }
    // B tile (16 x 64): unchanged.
#pragma unroll
    for (int s = 0; s < (RT_BK * RT_BN) / (RT_DIM * RT_DIM); ++s) {
      const int idx = tid + s * RT_DIM * RT_DIM;
      const int k = idx / RT_BN;
      const int c = idx % RT_BN;
      const int gr = k0 + k;
      const int gc = block_col + c;
      Bs[k][c] = (gr < n && gc < n) ? B[(size_t)gr * n + gc] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < RT_BK; ++k) {
      const float4 a4 = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
      const float4 b4 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
      const float a[RT_TM] = {a4.x, a4.y, a4.z, a4.w};
      const float b[RT_TN] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
      for (int i = 0; i < RT_TM; ++i)
#pragma unroll
        for (int j = 0; j < RT_TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

  // 4 contiguous floats per row. Vector store only when n is a multiple of 4 and the
  // whole group is in range; otherwise guarded scalar stores, so any n works.
#pragma unroll
  for (int i = 0; i < RT_TM; ++i) {
    const int r = block_row + ty * 4 + i;
    const int c = block_col + tx * 4;
    if (r >= n) continue;
    if (((n & 3) == 0) && c + 3 < n) {
      *reinterpret_cast<float4*>(&C[(size_t)r * n + c]) =
          make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    } else {
#pragma unroll
      for (int j = 0; j < RT_TN; ++j)
        if (c + j < n) C[(size_t)r * n + c + j] = acc[i][j];
    }
  }
}

__global__ void float_to_half(const float* in, __half* out, size_t count) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = __float2half(in[i]);
}

__global__ void half_to_float(const __half* in, float* out, size_t count) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = __half2float(in[i]);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct Timing {
  double median_ms;
  double min_ms;
};

// Times fn() with CUDA events (default stream): `warmup` untimed calls, then `reps`
// timed calls. Returns median and minimum.
template <typename F>
Timing time_gpu(F&& fn, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> t(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&t[i], start, stop));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(t.begin(), t.end());
  return {(double)t[reps / 2], (double)t.front()};
}

static void gemm_cpu_ikj(const float* A, const float* B, float* C, int n) {
  std::fill(C, C + (size_t)n * n, 0.0f);
  for (int i = 0; i < n; ++i) {
    float* c = C + (size_t)i * n;
    for (int k = 0; k < n; ++k) {
      const float a = A[(size_t)i * n + k];
      const float* b = B + (size_t)k * n;
      for (int j = 0; j < n; ++j) c[j] += a * b[j];
    }
  }
}

// max |x - ref| / max |ref|
static double norm_max_err(const std::vector<float>& x,
                           const std::vector<float>& ref) {
  double max_ref = 0.0, max_err = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_ref = std::max(max_ref, (double)std::fabs(ref[i]));
    max_err = std::max(max_err, (double)std::fabs(x[i] - ref[i]));
  }
  return max_ref > 0.0 ? max_err / max_ref : max_err;
}

static void put(FILE* f, double v, const char* fmt) {
  if (!std::isnan(v)) fprintf(f, fmt, v);
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  const double peak_gflops = (argc > 1) ? std::atof(argv[1]) * 1000.0 : 0.0;
  const char* csv_path = (argc > 2) ? argv[2] : "results/gemm_results.csv";

  FILE* csv = std::fopen(csv_path, "w");
  if (!csv) {
    fprintf(stderr, "Cannot open %s for writing (does the folder exist?)\n",
            csv_path);
    return 1;
  }
  fprintf(csv,
          "device,n,impl,reps,median_ms,min_ms,gflops,gb_per_s,"
          "pct_of_peak_fp32,max_norm_err,status,e2e_ms,e2e_speedup_vs_cpu\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const std::string dev_name = prop.name;
  printf("Device: %s | SMs: %d | compute capability %d.%d | %.2f GB\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor,
         prop.totalGlobalMem / 1073741824.0);
  if (peak_gflops > 0.0)
    printf("Reference FP32 peak for %% column: %.2f TFLOPS\n",
           peak_gflops / 1000.0);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const int sizes[] = {256, 512, 1024, 2048, 4096};
  bool all_ok = true;

  printf("\n%-5s %-16s %12s %12s %10s  %-9s %s\n", "N", "impl", "median_ms",
         "GFLOPS", "GB/s", "max_err", "status");

  for (int n : sizes) {
    const size_t elems = (size_t)n * n;
    const size_t bytes = elems * sizeof(float);
    const double flops = 2.0 * (double)n * n * n;
    const int gpu_reps = (n <= 1024) ? 20 : (n <= 2048 ? 10 : 5);
    const int cpu_reps = (n <= 512) ? 5 : (n == 1024 ? 3 : 1);
    const int xfer_reps = 10;

    std::vector<float> hA(elems), hB(elems), hC(elems), hC_cpu(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    // One row of output: prints to the console and appends to the CSV.
    // Pass NAN for any column that does not apply.
    auto emit = [&](const char* impl, int reps, double med, double mn,
                    double gflops, double gbps, double pct, double err,
                    const char* status, double e2e, double speedup) {
      printf("%-5d %-16s %12.4f ", n, impl, med);
      if (!std::isnan(gflops)) printf("%12.1f ", gflops); else printf("%12s ", "-");
      if (!std::isnan(gbps)) printf("%10.2f  ", gbps); else printf("%10s  ", "-");
      if (!std::isnan(err)) printf("%-9.2e ", err); else printf("%-9s ", "-");
      printf("%s\n", status);
      fflush(stdout);

      fprintf(csv, "\"%s\",%d,%s,%d,", dev_name.c_str(), n, impl, reps);
      put(csv, med, "%.4f");      fputc(',', csv);
      put(csv, mn, "%.4f");       fputc(',', csv);
      put(csv, gflops, "%.2f");   fputc(',', csv);
      put(csv, gbps, "%.2f");     fputc(',', csv);
      put(csv, pct, "%.2f");      fputc(',', csv);
      put(csv, err, "%.3e");      fprintf(csv, ",%s,", status);
      put(csv, e2e, "%.4f");      fputc(',', csv);
      put(csv, speedup, "%.2f");  fputc('\n', csv);
      fflush(csv);
    };

    // ---- CPU baseline -----------------------------------------------------
    std::vector<double> cpu_t;
    for (int r = 0; r < cpu_reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      gemm_cpu_ikj(hA.data(), hB.data(), hC_cpu.data(), n);
      auto t1 = std::chrono::steady_clock::now();
      cpu_t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(cpu_t.begin(), cpu_t.end());
    const double cpu_ms = cpu_t[cpu_t.size() / 2];
    emit("cpu_ikj_1thread", cpu_reps, cpu_ms, cpu_t.front(),
         flops / cpu_ms / 1e6, NAN, NAN, NAN, "reference", NAN, NAN);

    // ---- Device buffers ---------------------------------------------------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // ---- Transfers (measured before the kernels so e2e can use them) ------
    Timing h2d = time_gpu(
        [&] {
          CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
        },
        2, xfer_reps);
    Timing d2h = time_gpu(
        [&] {
          CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        },
        2, xfer_reps);
    emit("h2d_pageable", xfer_reps, h2d.median_ms, h2d.min_ms, NAN,
         2.0 * bytes / (h2d.median_ms * 1e6), NAN, NAN, "measured", NAN, NAN);
    emit("d2h_pageable", xfer_reps, d2h.median_ms, d2h.min_ms, NAN,
         (double)bytes / (d2h.median_ms * 1e6), NAN, NAN, "measured", NAN, NAN);

    {
      float *pA, *pB, *pC;
      CUDA_CHECK(cudaMallocHost(&pA, bytes));
      CUDA_CHECK(cudaMallocHost(&pB, bytes));
      CUDA_CHECK(cudaMallocHost(&pC, bytes));
      std::memcpy(pA, hA.data(), bytes);
      std::memcpy(pB, hB.data(), bytes);
      Timing h2d_p = time_gpu(
          [&] {
            CUDA_CHECK(cudaMemcpy(dA, pA, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, pB, bytes, cudaMemcpyHostToDevice));
          },
          2, xfer_reps);
      Timing d2h_p = time_gpu(
          [&] { CUDA_CHECK(cudaMemcpy(pC, dC, bytes, cudaMemcpyDeviceToHost)); },
          2, xfer_reps);
      emit("h2d_pinned", xfer_reps, h2d_p.median_ms, h2d_p.min_ms, NAN,
           2.0 * bytes / (h2d_p.median_ms * 1e6), NAN, NAN, "measured", NAN, NAN);
      emit("d2h_pinned", xfer_reps, d2h_p.median_ms, d2h_p.min_ms, NAN,
           (double)bytes / (d2h_p.median_ms * 1e6), NAN, NAN, "measured", NAN, NAN);
      CUDA_CHECK(cudaFreeHost(pA));
      CUDA_CHECK(cudaFreeHost(pB));
      CUDA_CHECK(cudaFreeHost(pC));
    }
    // dA and dB now hold the same data as hA and hB.

    // ---- Float-input implementations -------------------------------------
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C = A*B is column-major C^T = B^T * A^T, so swap the operands.
    auto cublas_float = [&](cublasComputeType_t ct) {
      CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                dB, CUDA_R_32F, n, dA, CUDA_R_32F, n, &beta, dC,
                                CUDA_R_32F, n, ct, CUBLAS_GEMM_DEFAULT));
    };

    auto bench_float = [&](const char* impl, auto&& launch, double tol,
                           bool pct_of_peak) {
      CUDA_CHECK(cudaMemset(dC, 0, bytes));
      Timing t = time_gpu(launch, 3, gpu_reps);
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
      const double err = norm_max_err(hC, hC_cpu);
      const bool ok = err <= tol;
      all_ok = all_ok && ok;
      const double gflops = flops / t.median_ms / 1e6;
      const double pct =
          (pct_of_peak && peak_gflops > 0.0) ? 100.0 * gflops / peak_gflops : NAN;
      const double e2e = h2d.median_ms + t.median_ms + d2h.median_ms;
      emit(impl, gpu_reps, t.median_ms, t.min_ms, gflops, NAN, pct, err,
           ok ? "PASS" : "FAIL", e2e, cpu_ms / e2e);
    };

    bench_float("cublas_fp32", [&] { cublas_float(CUBLAS_COMPUTE_32F); }, 1e-4, true);
    bench_float("cublas_tf32", [&] { cublas_float(CUBLAS_COMPUTE_32F_FAST_TF32); },
                5e-3, false);

    {
      dim3 block(NAIVE_BLOCK, NAIVE_BLOCK);
      dim3 grid((n + NAIVE_BLOCK - 1) / NAIVE_BLOCK, (n + NAIVE_BLOCK - 1) / NAIVE_BLOCK);
      bench_float("naive",
                  [&] {
                    gemm_naive<<<grid, block>>>(dA, dB, dC, n);
                    CUDA_CHECK(cudaGetLastError());
                  },
                  1e-4, true);
    }
    {
      dim3 block(TILE, TILE);
      dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
      bench_float("tiled",
                  [&] {
                    gemm_tiled<<<grid, block>>>(dA, dB, dC, n);
                    CUDA_CHECK(cudaGetLastError());
                  },
                  1e-4, true);
    }

    {
      dim3 block(RT_DIM, RT_DIM);
      dim3 grid((n + RT_BN - 1) / RT_BN, (n + RT_BM - 1) / RT_BM);
      bench_float("regtiled",
                  [&] {
                    gemm_regtiled<<<grid, block>>>(dA, dB, dC, n);
                    CUDA_CHECK(cudaGetLastError());
                  },
                  1e-4, true);
    }

    {
      dim3 block(RT_DIM, RT_DIM);
      dim3 grid((n + RT_BN - 1) / RT_BN, (n + RT_BM - 1) / RT_BM);
      bench_float("regtiled_vec",
                  [&] {
                    gemm_regtiled_vec<<<grid, block>>>(dA, dB, dC, n);
                    CUDA_CHECK(cudaGetLastError());
                  },
                  1e-4, true);
    }

    // ---- FP16 (tensor cores, FP32 accumulate) ----------------------------
    {
      const size_t hbytes = elems * sizeof(__half);
      __half *dAh, *dBh, *dCh;
      CUDA_CHECK(cudaMalloc(&dAh, hbytes));
      CUDA_CHECK(cudaMalloc(&dBh, hbytes));
      CUDA_CHECK(cudaMalloc(&dCh, hbytes));
      const int threads = 256;
      const int blocks = (int)((elems + threads - 1) / threads);
      float_to_half<<<blocks, threads>>>(dA, dAh, elems);
      float_to_half<<<blocks, threads>>>(dB, dBh, elems);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      auto launch = [&] {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                  dBh, CUDA_R_16F, n, dAh, CUDA_R_16F, n, &beta,
                                  dCh, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
      };
      CUDA_CHECK(cudaMemset(dCh, 0, hbytes));
      Timing t = time_gpu(launch, 3, gpu_reps);
      half_to_float<<<blocks, threads>>>(dCh, dC, elems);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
      const double err = norm_max_err(hC, hC_cpu);
      const bool ok = err <= 2e-2;
      all_ok = all_ok && ok;
      // No e2e figure for FP16: it would need half-size transfers plus conversion.
      emit("cublas_fp16", gpu_reps, t.median_ms, t.min_ms, flops / t.median_ms / 1e6,
           NAN, NAN, err, ok ? "PASS" : "FAIL", NAN, NAN);

      CUDA_CHECK(cudaFree(dAh));
      CUDA_CHECK(cudaFree(dBh));
      CUDA_CHECK(cudaFree(dCh));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    printf("\n");
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  std::fclose(csv);
  printf("%s\nResults written to %s\n",
         all_ok ? "All correctness checks PASSED." : "WARNING: at least one check FAILED.",
         csv_path);
  return all_ok ? 0 : 2;
}