# change 1: vectorized shared loads (`regtiled_vec`)

Hardware: RTX 3060 Laptop GPU (30 SMs, CC 8.6), CUDA 13.1, Nsight Compute 2025.4.1.
Kernel: `gemm_regtiled_vec` in `cuda/gemm_benchmark.cu` and `profile_gemm.cu`.
Change: A tile stored transposed in shared memory (padded to 68 floats per row), each thread owns a
contiguous 4x4 block, inner loop reads A and B with one `float4` shared load each. Nothing else changed.

Files:
- Correctness: `Profiling_Results/regtiled_vec_correctness.txt` (N = 2048, 1000, 1023, all PASS)
- Benchmark: `results/gemm_regtiled_vec_run1..3.csv` and `.txt`, `results/gpu_state_regtiled_vec.txt`
- Profiles (N = 2048): `ncu_regtiled_vec_2048.csv`, `ncu_regtiled_counters_2048.csv`, `ncu_regtiled_vec_counters_2048.csv`

## Benchmark (3 runs in one session, GFLOPS, median and range)

| N | tiled | regtiled | regtiled_vec | change vs regtiled | cuBLAS FP32 | regtiled_vec as % of cuBLAS |
|---|---|---|---|---|---|---|
| 1024 | 764 | 3799 (3168 to 3813) | 4923 (4153 to 5066) | +30% | 5761 | 85% |
| 2048 | 966 | 3694 (3494 to 3784) | 3732 (3598 to 4314) | +1% (ranges overlap) | 7307 | 51% |
| 4096 | 970 | 3173 (3151 to 3195) | 3375 (3300 to 3430) | +6% | 5119 | 66% |

Percentages are only comparable within one session: cuBLAS read 7307 GFLOPS at N = 2048 in this session
and 6556 in the earlier regtiled session, so the same kernel shows a different % of cuBLAS.
N = 256 and 512 are not used: runs are too short and clocks varied between runs.

## Profile at N = 2048 (Nsight Compute locks the SM clock to about 0.9 GHz)

| Metric | regtiled | regtiled_vec |
|---|---|---|
| Duration | 7.93 ms | 5.76 ms (27% shorter) |
| L1/TEX throughput (busy %) | 96.5 | 83.7 |
| Memory pipes busy (%) | 94.2 | 64.9 |
| FMA pipe (% of elapsed cycles) | 33.3 | 46.6 |
| Share of FP32 peak (roofline, at profiler clock) | 31% | 43% |
| Issue slots busy (%) | 54.3 | 66.3 |
| Executed IPC | 2.22 | 2.72 |
| Eligible warps per scheduler | 2.46 | 2.95 |
| Cycles with no eligible warp (%) | 44.4 | 32.0 |
| Instructions executed | 464.6 million | 412.1 million |
| Shared-load instructions | 83.9 million | 33.6 million |
| Shared-load instructions per warp per k step | 5 | 2 |
| Shared-load wavefronts | 100.67 million | 100.68 million |
| Shared-load wavefronts per warp per k step | 6.0 | 6.0 |
| Shared-store bank conflicts | 0 | 4.19 million (about 4% of all shared wavefronts) |
| Stall mio_throttle (warps per issue) | 3.59 | 2.20 |
| Stall barrier | 1.68 | 1.52 |
| Stall short scoreboard | 1.09 | 0.89 |
| Stall long scoreboard | 0.46 | 0.68 |
| Stall math pipe throttle | 0.31 | 0.43 |
| Registers per thread / spilling | 64 / 0 | 64 / 0 |
| Achieved occupancy (%) | 64.4 | 64.5 |

"Per warp per k step" = counter / 16,777,216 (1024 blocks x 8 warps x 128 k tiles x 16 k steps).

## What the hypothesis got right and wrong

Right:
- Shared-memory instruction pressure went down: shared-load instructions fell 2.5x, the mio_throttle
  stall fell from 3.59 to 2.20, memory pipes busy fell from 94% to 65%, the FMA pipe rose from 33% to 47%,
  and the kernel is 27% shorter at the profiler clock.

Wrong:
- The hypothesis assumed regtiled issued 8 scalar shared loads per k step. The counter shows 5. The
  compiler had already merged the A reads into 128-bit loads (As[m][k] is contiguous in k), so only the
  4 B reads were scalar. The real reduction is 5 to 2 loads per k step (2.5x), not 8 to 2 (4x).
- Shared-load wavefronts did not change (about 100.7 million, 6.0 per warp per k step, in both kernels).
  The change removed instruction-issue pressure, not shared-memory data traffic. The kernel is still
  limited by shared-memory work per multiply-add.
- The transposed A store causes 2-way bank conflicts (ncu reports 1.5-way on average over A and B
  stores), about 4% of all shared wavefronts.

## Benchmark and profiler disagree at large N

At the profiler's fixed clock the new kernel is 27% shorter at N = 2048. In the benchmark (normal
clocks) it is +30% at N = 1024, +1% at N = 2048 and +6% at N = 4096. The cause is not established.
SM clocks were not logged during the benchmark, so clock or power behaviour on longer runs is a
candidate, not a finding.

## Decision: stop after one change

- Returns at the larger sizes are small (+1% and +6%).
- The remaining profile-guided fixes are small: the store conflicts are about 4% of shared wavefronts, and
  the occupancy limit comes from 64 registers per thread.
- The next real lever is a larger per-thread tile (8x8) to cut shared wavefronts per multiply-add. That is
  a rewrite, not a tweak. Not done.

Not done: 8x8 register tile, removing the A-store bank conflicts, double buffering, logging SM clocks during
the benchmark.
