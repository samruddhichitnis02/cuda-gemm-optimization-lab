# Records the GPU/toolchain state, runs the benchmark, records the state again.
# Run from the repo root after building:
#     .\scripts\run_benchmark.ps1
#
# Before running: plug in the charger, set the laptop to its high-performance mode,
# and close other GPU-heavy programs (browsers with video, games).

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force results | Out-Null

$query = "name,uuid,driver_version,power.limit,power.draw,clocks.max.sm,clocks.max.mem,temperature.gpu"

nvidia-smi -q | Out-File -Encoding utf8 results\nvidia-smi-q.txt
nvcc --version | Out-File -Encoding utf8 results\nvcc_version.txt
"Run started: $(Get-Date -Format o)" | Out-File -Encoding utf8 results\gpu_state.txt
nvidia-smi --query-gpu=$query --format=csv | Out-File -Encoding utf8 -Append results\gpu_state.txt

# 13.08 = FP32 reference peak (TFLOPS) used in the GPU Lab for this laptop GPU.
# It is a reference value, not a guaranteed application-level number.
.\build\gemm_benchmark.exe 13.08 results\gemm_results.csv 2>&1 | Tee-Object -FilePath results\RUN_LOG.txt

"Run finished: $(Get-Date -Format o)" | Out-File -Encoding utf8 -Append results\gpu_state.txt
nvidia-smi --query-gpu=$query --format=csv | Out-File -Encoding utf8 -Append results\gpu_state.txt

Write-Host "Saved: results\gemm_results.csv, RUN_LOG.txt, gpu_state.txt, nvidia-smi-q.txt, nvcc_version.txt"