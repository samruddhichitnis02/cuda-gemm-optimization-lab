// profile_gemm.cu
//
// Runs ONE GEMM kernel ONCE at ONE size, so Nsight Compute can profile it in seconds.
// This is a profiling helper, not a benchmark: it does no timing.
//
// Usage:  profile_gemm.exe <naive|tiled|regtiled|regtiled_vec|cublas> [N]      (N defaults to 2048)
//
// naive, tiled and cublas are copied unchanged from gemm_benchmark.cu so the profile
// matches what the benchmark measures. regtiled is the new register-tiled kernel.
// After the launch, 16 sampled output entries are checked
// against a double-precision CPU result, so we never profile a broken kernel.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

constexpr int NAIVE_BLOCK = 16;
constexpr int TILE = 32;

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
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < n && col < n) C[(size_t)row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Register-tiled kernel.
//
// A 16x16 block (256 threads) computes a 64x64 tile of C. Each thread computes a
// 4x4 block of outputs and keeps all 16 running sums in registers.
//
// Per step along K (16 values at a time):
//   * the block copies a 64x16 tile of A and a 16x64 tile of B into shared memory
//   * each thread then, for every k, loads 4 values of A and 4 values of B from
//     shared memory (8 loads) and does 16 multiply-adds with them
// so shared memory is read 0.5 times per multiply-add instead of 2 times in gemm_tiled.
//
// A thread's 4 rows are ty, ty+16, ty+32, ty+48 and its 4 columns are tx, tx+16,
// tx+32, tx+48. Spreading them out (instead of 4 neighbours) means threads that are
// next to each other in a warp read neighbouring shared-memory addresses, which
// avoids bank conflicts. Edges are handled: out-of-range loads read 0 and
// out-of-range stores are skipped, so any N works.
// ---------------------------------------------------------------------------
constexpr int RT_BM = 64;  // rows of C per block
constexpr int RT_BN = 64;  // columns of C per block
constexpr int RT_BK = 16;  // K values per step
constexpr int RT_TM = 4;   // rows of C per thread
constexpr int RT_TN = 4;   // columns of C per thread
constexpr int RT_DIM = 16; // threads per block side (16 x 16 = 256 threads)

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
    // Copy the A tile (64 x 16 = 1024 values, 4 per thread). Consecutive threads
    // read consecutive k, so global reads are coalesced.
#pragma unroll
    for (int s = 0; s < (RT_BM * RT_BK) / (RT_DIM * RT_DIM); ++s) {
      const int idx = tid + s * RT_DIM * RT_DIM;
      const int m = idx / RT_BK;
      const int k = idx % RT_BK;
      const int gr = block_row + m;
      const int gc = k0 + k;
      As[m][k] = (gr < n && gc < n) ? A[(size_t)gr * n + gc] : 0.0f;
    }
    // Copy the B tile (16 x 64 = 1024 values, 4 per thread).
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

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <naive|tiled|regtiled|regtiled_vec|cublas> [N]\n", argv[0]);
    return 1;
  }
  const char* which = argv[1];
  const int n = (argc > 2) ? std::atoi(argv[2]) : 2048;
  if (std::strcmp(which, "naive") && std::strcmp(which, "tiled") &&
      std::strcmp(which, "regtiled") && std::strcmp(which, "regtiled_vec") &&
      std::strcmp(which, "cublas")) {
    fprintf(stderr, "Unknown kernel '%s' (use naive, tiled, regtiled, regtiled_vec or cublas)\n", which);
    return 1;
  }
  if (n <= 0) {
    fprintf(stderr, "N must be positive\n");
    return 1;
  }

  const size_t elems = (size_t)n * n;
  const size_t bytes = elems * sizeof(float);

  std::vector<float> hA(elems), hB(elems), hC(elems);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : hA) v = dist(rng);
  for (auto& v : hB) v = dist(rng);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, bytes));

  // ---- the single launch that gets profiled ------------------------------
  if (!std::strcmp(which, "naive")) {
    dim3 block(NAIVE_BLOCK, NAIVE_BLOCK);
    dim3 grid((n + NAIVE_BLOCK - 1) / NAIVE_BLOCK, (n + NAIVE_BLOCK - 1) / NAIVE_BLOCK);
    gemm_naive<<<grid, block>>>(dA, dB, dC, n);
  } else if (!std::strcmp(which, "tiled")) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    gemm_tiled<<<grid, block>>>(dA, dB, dC, n);
  } else if (!std::strcmp(which, "regtiled")) {
    dim3 block(RT_DIM, RT_DIM);
    dim3 grid((n + RT_BN - 1) / RT_BN, (n + RT_BM - 1) / RT_BM);
    gemm_regtiled<<<grid, block>>>(dA, dB, dC, n);
  } else if (!std::strcmp(which, "regtiled_vec")) {
    dim3 block(RT_DIM, RT_DIM);
    dim3 grid((n + RT_BN - 1) / RT_BN, (n + RT_BM - 1) / RT_BM);
    gemm_regtiled_vec<<<grid, block>>>(dA, dB, dC, n);
  } else {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cublasCreate failed\n");
      return 1;
    }
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C = A*B is column-major C^T = B^T * A^T, so swap the operands.
    cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                     dB, CUDA_R_32F, n, dA, CUDA_R_32F, n, &beta, dC,
                                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cublasGemmEx failed: status %d\n", (int)st);
      return 1;
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- spot-check 16 entries against a double-precision CPU result -------
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
  std::mt19937 pick(7);
  std::uniform_int_distribution<int> idx(0, n - 1);
  double worst = 0.0;
  for (int s = 0; s < 16; ++s) {
    const int i = idx(pick), j = idx(pick);
    double ref = 0.0;
    for (int k = 0; k < n; ++k)
      ref += (double)hA[(size_t)i * n + k] * (double)hB[(size_t)k * n + j];
    const double err = std::fabs((double)hC[(size_t)i * n + j] - ref);
    if (err > worst) worst = err;
  }
  const bool ok = worst < 1e-2;  // loose absolute bound; entries are O(sqrt(N))
  printf("kernel=%s N=%d spot-check max abs err=%.3e -> %s\n", which, n, worst,
         ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return ok ? 0 : 2;
}
