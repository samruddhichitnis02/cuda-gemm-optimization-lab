# Builds build\gemm_benchmark.exe. Run from the repo root after:  . .\scripts\env.ps1
#     .\scripts\build.ps1
#
# -arch=sm_86      RTX 3060 Laptop GPU (Ampere GA106, compute capability 8.6)
# -lineinfo        keeps source-line info so Nsight Compute can map metrics to code
# /O2,/arch:AVX2   host (CPU baseline) compiler flags: optimized, AVX2 vectorization

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force build, results | Out-Null

nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xcompiler "/O2,/arch:AVX2" cuda\gemm_benchmark.cu -o build\gemm_benchmark.exe -lcublas
if ($LASTEXITCODE -ne 0) { throw "nvcc failed (exit code $LASTEXITCODE)" }

Write-Host "Built build\gemm_benchmark.exe"