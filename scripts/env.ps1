# Loads the Visual Studio 2022 C++ environment and puts Nsight on PATH.
# Run it "dot-sourced" from the repo root, in any PowerShell window:
#     . .\scripts\env.ps1

$vsDevShell = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
& $vsDevShell -Arch amd64 -SkipAutomaticLocation

$nsightDirs = @(
    "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\target\windows-desktop-win7-x64",
    "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64"
)
foreach ($dir in $nsightDirs) {
    if ((Test-Path $dir) -and ($env:Path -notlike "*$dir*")) {
        $env:Path += ";$dir"
    }
}

Write-Host "cl:   " (Get-Command cl -ErrorAction SilentlyContinue).Source
Write-Host "nvcc: " (Get-Command nvcc -ErrorAction SilentlyContinue).Source
Write-Host "ncu:  " (Get-Command ncu -ErrorAction SilentlyContinue).Source
Write-Host "nsys: " (Get-Command nsys -ErrorAction SilentlyContinue).Source