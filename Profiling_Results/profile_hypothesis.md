# Profile findings and hypothesis (before writing the register-tiled kernel)

Hardware: RTX 3060 Laptop GPU (30 SMs, compute capability 8.6), CUDA 13.1, Nsight Compute 2025.4.1.
Profiles: `ncu_naive_2048.csv`, `ncu_tiled_2048.csv` (one launch each, N = 2048, `--set full`).

Caveat: Nsight Compute locks the GPU to its base clock (~0.9 GHz), so profile durations are
longer than benchmark times. The profiles are used to explain *why*; the benchmark
(`gemm_results.csv`, `RUN_LOG.txt`) is used for *how fast*.

## Reference speeds (benchmark, FP32)

| Kernel | GFLOPS (N = 2048 to 4096) |
|---|---|
| naive | about 665 to 680 |
| tiled | about 970 to 980 |
| cuBLAS FP32 | about 5,300 to 6,500 |

## What the profiles measured (N = 2048)

| Metric | naive | tiled |
|---|---|---|
| L1/TEX throughput (busy %) | 99.6 | 82.3 |
| DRAM throughput (busy %) | 16.5 | 10.3 |
| Instructions executed | 1.47 billion | 0.86 billion |
| Achieved occupancy (%) | 99.5 | 66.6 |
| Registers per thread | 40 | 38 |
| Largest stall (share of cycles between instructions) | global-load queue, 64% | shared-memory (MIO) queue, 59% |
| Eligible warps per scheduler | 1.58 | 1.12 |
| Share of FP32 peak (profiler roofline) | 6% | 7% |

Other tiled observations: one 1024-thread block fits per SM (theoretical occupancy 66.7%,
limited by block size and registers); shared-store bank conflicts of about 1.3-way
(21% of store wavefronts); 76% of cycles with no eligible warp to issue.

## Hypothesis H1 (main)

The tiled kernel is limited by shared-memory instruction throughput, not by main memory or math.
Its inner loop does two shared-memory loads (`As[ty][k]` and `Bs[k][tx]`) for every one
multiply-add.

Supporting evidence: the largest stall is the shared-memory instruction queue (59%),
L1/TEX (which includes shared memory) is 82% busy while DRAM is only 10% busy, and the
kernel reaches only 7% of FP32 peak.

Not yet verified: the exact ratio of shared loads to multiply-adds. Check it in the compiled
instruction mix (`cuobjdump -sass`) when we compare kernels.

## Hypothesis H2 (secondary)

Occupancy is limited by the 1024-thread block: only one block fits per SM, so when its warps wait at
`__syncthreads()` there is no other block to run. The profiler estimates this is worth up to about 18%;
that is an estimate, not a measurement.

## Open question

The shared-memory bank conflicts on stores are not explained by reading the code (the stores look
conflict-free). Treat as unexplained until a later profile says more.

## Prediction for the register-tiled kernel (A3)

If H1 is right, letting each thread compute an R x R block of outputs in registers means each
thread loads 2R values from shared memory for R x R multiply-adds, i.e. 2/R loads per
multiply-add (R = 1 today: 2 loads per multiply-add; R = 4: 0.5; R = 8: 0.25).
Expected direction: the shared-memory stall share and L1/TEX busy % go down, share of FP32 peak goes up.

No speed target is set. The result is whatever we measure.

## What would prove H1 wrong or incomplete

- The shared-memory (MIO) stall is still the largest stall after register tiling and GFLOPS did not improve.
- GFLOPS drops because more registers per thread cause spilling or low occupancy (check
  "Local Memory Spilling Requests" and occupancy in the new profile).

## What we record for the new kernel

Median GFLOPS at N = 1024, 2048, 4096 (3 or more runs); percent of cuBLAS FP32; and from Nsight Compute:
largest stall, L1/TEX busy %, registers per thread, spilling requests, achieved occupancy.
