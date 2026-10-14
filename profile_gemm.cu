// profile_gemm.cu
//
// Runs ONE GEMM kernel ONCE at ONE size, so Nsight Compute can profile it in seconds.
// This is a profiling helper, not a benchmark: it does no timing.
//
// Usage:  profile_gemm.exe <naive|tiled|cublas> [N]      (N defaults to 2048)
//
// Kernels are copied unchanged from gemm_benchmark.cu so the profile matches what
// the benchmark measures. After the launch, 16 sampled output entries are checked
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

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <naive|tiled|cublas> [N]\n", argv[0]);
    return 1;
  }
  const char* which = argv[1];
  const int n = (argc > 2) ? std::atoi(argv[2]) : 2048;
  if (std::strcmp(which, "naive") && std::strcmp(which, "tiled") &&
      std::strcmp(which, "cublas")) {
    fprintf(stderr, "Unknown kernel '%s' (use naive, tiled or cublas)\n", which);
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
