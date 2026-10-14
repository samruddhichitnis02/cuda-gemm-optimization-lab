# CUDA GEMM Optimization Lab

Performance analysis of matrix multiplication on an NVIDIA GPU, from a hand-written CUDA kernel up to framework-level fusion with Triton and `torch.compile`. Every number below was measured in this repo and is backed by a file in `results/` or `Profiling_Results/`.

**Hardware / software:** RTX 3060 Laptop GPU (30 SMs, compute capability 8.6, 6 GB), CUDA 13.1, Nsight Compute 2025.4.1, PyTorch 2.11 (cu128), Triton 3.8.0, Windows. FP32 peak reference used for percentages: 13.08 TFLOPS.

## Headline results

- Hand-written register-tiled FP32 kernel: **5.2x faster than the naive kernel at N=4096 (645 to 3375 GFLOPS) and 7.3x at N=1024 (670 to 4923)**, and 3.5x to 6.4x faster than the shared-memory tiled kernel.
- Best hand-written kernel reaches **51% to 85% of cuBLAS FP32** (depending on N) and 26% to 38% of the 13.08 TFLOPS FP32 reference peak.
- Nsight Compute profile of the kernels explains the gains: instructions executed fell from 1467 M (naive) to 412 M, and share of FP32 peak rose from ~7% to 43% at the profiler's locked clock.
- cuBLAS at N=4096: FP32 5.1 TFLOPS, TF32 11.7 TFLOPS (2.3x), FP16 with FP32 accumulate 23.2 TFLOPS (4.5x).
- Fused Triton matmul + bias + GELU (FP16) is **1.56x faster than PyTorch eager at N=1024**, and 5.8% slower at N=4096; `torch.compile` was not faster than eager at any size tested.

## Part A: GEMM kernels in CUDA C++

All kernels compute `C = A * B` for square FP32 matrices, in `cuda/gemm_benchmark.cu`.

| Kernel | Idea |
|---|---|
| `cpu_ikj_1thread` | single-threaded CPU reference (cache-friendly loop order) |
| `naive` | one thread per output, every operand read from global memory |
| `tiled` | 32x32 shared-memory tiles |
| `regtiled` | 64x64 block tile; each thread holds a 4x4 block of results in registers |
| `regtiled_vec` | `regtiled` plus a transposed, padded A tile so each thread reads its operands with `float4` shared loads |
| `cublas_fp32 / tf32 / fp16` | cuBLAS reference points (TF32 and FP16 use tensor cores) |

### Benchmark (GFLOPS, median of 3 runs, range in brackets)

| N | naive | tiled | regtiled | regtiled_vec | cuBLAS FP32 |
|---|---|---|---|---|---|
| 1024 | 670 | 764 | 3799 (3168 to 3813) | 4923 (4153 to 5066) | 5761 |
| 2048 | 668 | 966 | 3694 (3494 to 3784) | 3732 (3598 to 4314) | 7307 |
| 4096 | 645 | 970 | 3173 (3151 to 3195) | 3375 (3300 to 3430) | 5119 |

Source: `results/gemm_regtiled_vec_run1..3.csv`. cuBLAS readings differ between sessions (6556 vs 7307 GFLOPS at N=2048), so percentages of cuBLAS are only compared within one session.

- Register tiling: 3.3x to 5.0x over `tiled`.
- Vectorized loads over `regtiled`: +30% at N=1024, +1% at N=2048 (ranges overlap, no real difference), +6% at N=4096.
- Single-threaded CPU at N=4096: 7 GFLOPS (`results/gemm_results.csv`), so the best GPU kernel is about 480x faster.
- Host-device transfers are timed separately. At N=4096, pinned memory moved data at 10.2 GB/s host-to-device vs 9.4 GB/s pageable (`results/gemm_results.csv`).

### Profiling with Nsight Compute (N=2048)

Profiles are in `Profiling_Results/`. Nsight Compute locks the SM clock to about 0.9 GHz, so durations differ from the benchmark.

| Metric | naive | tiled | regtiled | regtiled_vec |
|---|---|---|---|---|
| Instructions executed | 1467 M | 861 M | 465 M | 412 M |
| Warp cycles per issued instruction | 35.1 | 33.4 | 13.9 | see CSV |
| Share of FP32 peak (roofline) | ~7% | ~7% | 31% | 43% |
| L1/TEX throughput busy | | | 96.5% | 83.7% |
| Issue slots busy | | | 54.3% | 66.3% |
| Shared-load instructions | | | 83.9 M | 33.6 M |
| Registers per thread / spills | | | 64 / 0 | 64 / 0 |
| Kernel duration at profiler clock | | | 7.93 ms | 5.76 ms |

What the profiles showed:
- `naive` and `tiled` both saturate the L1/TEX pipeline. `tiled` stalls mainly on the shared-memory instruction queue (2 shared loads per multiply-add).
- Register tiling cut shared loads per multiply-add to 0.5 and the instruction count nearly in half. No register spills, and the profile reported no bank-conflict warning for `regtiled`.
- Vectorized loads cut shared-load instructions from 83.9 M to 33.6 M and the `mio_throttle` stall from 3.59 to 2.20 warps per issue. Shared-load wavefronts were unchanged (about 100.7 M), so the change removed instruction pressure, not shared-memory traffic.
- The transposed A store in `regtiled_vec` introduces 2-way bank conflicts, about 4% of shared wavefronts.
- A profile-based hypothesis was written before the change (`Profiling_Results/profile_hypothesis.md`). It was partly wrong: `regtiled` already issued 5 shared loads per k step, not 8, because the compiler had merged the A reads. Details in `Profiling_Results/vectorized_loads.md`.
- At N=2048 the profiler shows `regtiled_vec` 27% shorter than `regtiled`, while the benchmark shows +1%. SM clocks were not logged during the benchmark, so no cause is claimed.

### Correctness

Every implementation is checked against the CPU result in the benchmark. `regtiled_vec` was also spot-checked at N=2048, 1000 and 1023 (non-multiples of the tile size), all PASS, max abs error 2.4e-5 or less (`Profiling_Results/regtiled_vec_correctness.txt`).

## Part B: framework-level fusion with Triton and torch.compile

`triton/fused_bench.py` compares three FP16 implementations of `GELU(A @ B + bias)` (tanh form), square N x N x N:

- **eager**: PyTorch ops, `F.gelu(a @ b + bias)`
- **torch_compile**: the same function through `torch.compile`
- **triton_fused**: one Triton kernel doing matmul, bias and GELU, with the epilogue in FP32 before a single FP16 store; autotuned over 4 tile configurations

Method: 25 warm-up calls, 100 timed calls with CUDA events, median per run; 4 full runs; error checked against an FP32 reference.

| N | eager (ms) | torch.compile (ms) | Triton fused (ms) | Triton vs eager |
|---|---|---|---|---|
| 1024 | 0.318 (0.280 to 0.322) | 0.352 (0.320 to 0.402) | **0.204 (0.182 to 0.222)** | 1.56x faster |
| 2048 | 0.991 (0.973 to 1.005) | 1.062 (1.014 to 1.105) | 0.932 (0.892 to 0.991) | 1.06x faster (ranges overlap) |
| 4096 | 6.509 (6.447 to 6.548) | 6.460 (6.430 to 6.534) | 6.886 (6.846 to 6.902) | 5.8% slower |

Median of 4 runs; range in brackets (`results/fused_run1..4.txt`, `results/fused_results.csv`). At N=4096 eager reaches 21.1 TFLOPS and Triton 20.0 TFLOPS.

- Triton fused output is more accurate than eager against the FP32 reference (max error 1.95e-3 to 1.98e-3 vs 3.25e-3 to 3.84e-3), consistent with the fused kernel keeping bias and GELU in FP32 before one rounding.
- `torch.compile` was not faster than eager at any of the three sizes. It was slower at N=1024 and N=2048 and within noise at N=4096.
- The Triton kernel supports sizes that are multiples of 128 only.
- The reason for the Triton result at each size was not profiled.

## How Part A and Part B connect

Both parts are about the same question: how much time goes to arithmetic and how much to moving data. In Part A, the naive kernel spends most of its instructions on loads, the tiled kernel is limited by shared-memory instruction throughput, and each step that reduces loads per multiply-add (shared-memory tiles, then register tiles, then vectorized loads) moves the kernel closer to FP32 peak. The profiles identify which resource is saturated before and after each change. cuBLAS at TF32 and FP16 then shows how much more throughput the tensor-core path delivers on the same hardware (2.3x and 4.5x over cuBLAS FP32).

Part B applies the same idea above the kernel level. An eager `matmul + bias + GELU` runs as separate operations; the Triton kernel does all three in one pass using the tensor-core path from Part A's cuBLAS comparison. The measurements show where this helps and where it does not: a clear win at N=1024, parity at N=2048, and a loss at N=4096 where the large matmul dominates the runtime and PyTorch's cuBLAS-backed matmul is faster than the autotuned Triton tile. This is why the kernel-level analysis matters for graph compilers and inference libraries: whether fusion pays off depends on how close the underlying matmul already is to hardware limits, which is what Part A measures.

## Measurement method

- GPU timing with CUDA events; warm-up before timing; median of repeated runs; 3 to 4 full runs per result, ranges reported.
- Every result is checked for correctness against a reference before its timing is used.
- Host-device transfers are timed separately from kernel time.
- GPU and toolchain state (`nvidia-smi`, `nvcc --version`) recorded with each run in `results/`.
- Small sizes (N of 512 or less) are excluded from kernel comparisons because runs are too short and clocks varied between runs.

## Repo layout

```
cuda/gemm_benchmark.cu     all GEMM kernels + benchmark harness (C++/CUDA, cuBLAS)
profile_gemm.cu            runs one kernel once at one size for Nsight Compute
triton/fused_bench.py      eager vs torch.compile vs Triton fused matmul+bias+GELU
scripts/                   env, build and benchmark scripts (PowerShell)
results/                   raw benchmark CSVs and run logs
Profiling_Results/         Nsight Compute CSVs, profile hypothesis, vectorized-load analysis
```

## Reproduce

Windows, x64 Native Tools Command Prompt for VS 2022 / PowerShell with CUDA 13.1:

```
. .\scripts\env.ps1
.\scripts\build.ps1
.\scripts\run_benchmark.ps1
python triton\fused_bench.py
```

Profile one kernel (administrator terminal needed for GPU counters):

```
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo profile_gemm.cu -o build\profile_gemm.exe -lcublas
ncu --set full --kernel-name regex:gemm_ --csv --page details --log-file Profiling_Results\out.csv .\build\profile_gemm.exe regtiled_vec 2048
```
