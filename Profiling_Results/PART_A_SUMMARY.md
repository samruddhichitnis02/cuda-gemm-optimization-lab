# Part A summary: FP32 GEMM optimization on an RTX 3060 Laptop GPU

Hardware: RTX 3060 Laptop (30 SMs, CC 8.6, 6 GB), CUDA 13.1, Nsight Compute 2025.4.1, FP32 peak reference 13.08 TFLOPS.
All numbers are measured in this repo. Each benchmark result is the median of 3 runs (5 timed iterations each, after warm-up) and is checked against a reference.

## The kernels

| Kernel | Idea | What limits it (from the profile) |
|---|---|---|
| naive | 1 thread per output, every operand read from global memory | L1/TEX pipe saturated by loads; most instructions are loads, not math |
| tiled | 32x32 tiles staged in shared memory | shared-memory instruction queue (2 shared loads per multiply-add) |
| regtiled | 64x64 block tile, each thread keeps a 4x4 block of results in registers | L1/TEX pipe still ~96% busy |
| regtiled_vec | same, but A is stored transposed so each thread reads its operands with `float4` shared loads | fewer load instructions, but the same amount of shared-memory data movement |

## Benchmark (GFLOPS, median, range in brackets)

| N | naive | tiled | regtiled | regtiled_vec | cuBLAS FP32 |
|---|---|---|---|---|---|
| 1024 | 670 | 764 | 3799 (3168 to 3813) | 4923 (4153 to 5066) | 5761 |
| 2048 | 668 | 966 | 3694 (3494 to 3784) | 3732 (3598 to 4314) | 7307 |
| 4096 | 645 | 970 | 3173 (3151 to 3195) | 3375 (3300 to 3430) | 5119 |

All from one session (`gemm_regtiled_vec_run1..3`). cuBLAS varied between sessions (6556 vs 7307 GFLOPS at N=2048), so percentages of cuBLAS only compare within a session.

What this supports:
- regtiled is about 3.3x to 5.0x faster than tiled (3799/764 at N=1024, 3694/966 at N=2048, 3173/970 at N=4096).
- regtiled_vec adds +30% at N=1024, +1% at N=2048 (ranges overlap, so no real difference), and +6% at N=4096 over regtiled.
- Best kernel reaches about 51% to 85% of cuBLAS FP32 depending on N, and about 26% to 38% of the 13.08 TFLOPS peak.

## What the profile showed (N=2048, ncu locks the clock to about 0.9 GHz)

| Metric | naive | tiled | regtiled | regtiled_vec |
|---|---|---|---|---|
| Instructions executed | 1467 M | 861 M | 465 M | 412 M |
| Warp cycles per issued instruction | 35.1 | 33.4 | 13.9 | not recorded here (see ncu CSV) |
| Share of FP32 peak (roofline, profiler clock) | ~7% | ~7% | 31% | 43% |
| Registers per thread | not recorded here | not recorded here | 64 | 64 |

The hypothesis (`profile_hypothesis.md`) was partly right. Shared-memory instruction pressure was the tiled kernel's bottleneck, and reducing it gave the big win. Details that were wrong are in `vectorized_loads.md`: regtiled issued 5 shared loads per k step, not 8, because the compiler already merged the A reads.

## Note on N=2048

At the profiler's fixed clock regtiled_vec is 27% shorter than regtiled, but in the benchmark the difference at N=2048 is +1% (ranges overlap). SM clocks were not logged during the benchmark, so I do not claim a cause.
