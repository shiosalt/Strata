<#
.SYNOPSIS
  Builds the Nemotron 3.5 Super engine (strata.exe with nemotron_h_moe, strata-nemotron.exe) from this checkout, on Windows.

.DESCRIPTION
  Finds Visual Studio 2022 (C++ build tools) and the CUDA toolkit, installs CMake and Ninja into the checkout's .venv
  (requirements-build.txt), configures build\nemotron-cuda for the GPUs in this PC and builds it.  With -Vision it also
  fetches llama.cpp at a fixed commit (the one that knows this mmproj, "nemotron_omni") and builds the picture reader.

  Measured only on Windows 11, VS 2022 Build Tools 17.14, CUDA 13.1, RTX 5060 Ti (sm_120).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1
  powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1 -Vision -Tests
#>
[CmdletBinding()]
param(
    [string]$Arch = "",          # CUDA architectures, e.g. "120" or "89;120" (default: the GPUs in this PC)
    [switch]$Vision,             # also build strata-vision (CPU) in build\vision-cpu
    [switch]$VisionGpu,          # also build strata-vision (CUDA) in build\vision-cuda
    [switch]$Tests,              # also build the Nemotron tests (nemotron_kernels_parity, nemotron_engine_parity, ...)
    [int]$Jobs = 0,              # parallel compile jobs (default: the logical processors)
    [string]$BuildDir = "build\nemotron-cuda",
    [string]$VisionBuildDir = "build\vision-cpu",
    [switch]$Portable            # a build for other PCs (tools\nemotron\package.ps1): AVX2 baseline, static MSVC runtime, the release
                                 # GPU list "86-real;89-real;120" unless -Arch says otherwise
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
# the llama.cpp the picture reader is built against (the engine's own ggml stays at third_party/strata-ggml)
$LlamaVisionCommit = "aa75ff3239f9cc675b081697deda9e1b112345c6"

function Say($m) { Write-Host "[build-nemotron] $m" }
function Fail($m) { Write-Host "[build-nemotron] ERROR: $m" -ForegroundColor Red; exit 1 }

# ---- Visual Studio (C++ x64 tools)
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Fail "Visual Studio 2022 (or its Build Tools) with 'Desktop development with C++' is needed: https://visualstudio.microsoft.com/downloads/" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { Fail "no Visual Studio with the C++ x64 tools was found (install the 'Desktop development with C++' workload)" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { Fail "vcvars64.bat is missing in $vs" }
Say "Visual Studio: $vs"

# ---- CUDA
$cuda = $env:CUDA_PATH
if (-not $cuda -or -not (Test-Path (Join-Path $cuda "bin\nvcc.exe"))) {
    $base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $cuda = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path (Join-Path $_.FullName "bin\nvcc.exe") } |
            Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cuda) { Fail "the CUDA toolkit (12.8 or newer; 13.1 was tested) is needed: https://developer.nvidia.com/cuda-downloads" }
Say "CUDA: $cuda"

# ---- GPU architectures
if (-not $Arch -and $Portable) { $Arch = "86-real;89-real;120" }   # (the list upstream's Windows CUDA 13 engines are built for)
if (-not $Arch) {
    $caps = @()
    try { $caps = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null } catch { }
    $caps = @($caps | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+\.\d+$' } | ForEach-Object { $_.Replace(".", "") } | Sort-Object -Unique)
    if ($caps.Count -eq 0) { Fail "no NVIDIA GPU was found (nvidia-smi); pass -Arch, e.g. -Arch 120" }
    $Arch = $caps -join ";"
}
if ($Arch -ne "120") { Say "WARNING: built for sm $Arch. Only sm_120 (RTX 50 series) has been tested." }
$port = if ($Portable) { "ON" } else { "OFF" }
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }

# ---- CMake and Ninja in .venv
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $sys = Get-Command python -ErrorAction SilentlyContinue
    if (-not $sys) { Fail "Python 3.10+ is needed (https://www.python.org/downloads/)" }
    Say "creating .venv"
    & $sys.Source -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { Fail "python -m venv failed" }
}
# the build's (cmake, ninja), the server's and the Terminal's (requirements.txt, textual: what setup.py installs for a run)
$textual = (Select-String -Path (Join-Path $Root "requirements-dev.txt") -Pattern '^textual==').Line
& $py -m pip install --disable-pip-version-check -q -r (Join-Path $Root "requirements-build.txt") -r (Join-Path $Root "requirements.txt") $textual
if ($LASTEXITCODE -ne 0) { Fail "pip could not install requirements-build.txt / requirements.txt" }
$scripts = Join-Path $Root ".venv\Scripts"

# every step runs in one cmd with the compiler's environment
function Invoke-Vs([string[]]$lines, [string]$log) {
    $f = Join-Path $env:TEMP ("strata-build-nemotron-" + [guid]::NewGuid().ToString("N") + ".cmd")
    $body = @("@echo off", "call `"$vcvars`" >nul || exit /b 1", "set `"CUDA_PATH=$cuda`"",
              "set `"PATH=$cuda\bin;$scripts;%PATH%`"", "cd /d `"$Root`"") + ($lines | ForEach-Object { "$_ || exit /b 1" })
    Set-Content -Path $f -Value $body -Encoding ASCII
    try {
        & cmd /c "`"$f`" > `"$log`" 2>&1"
        return $LASTEXITCODE
    } finally { Remove-Item $f -ErrorAction SilentlyContinue }
}
New-Item -ItemType Directory -Force -Path (Join-Path $Root "build") | Out-Null
$log = Join-Path $Root "build\build-nemotron.log"
# a running server holds the programs open (the link would fail on them)
function Assert-NotRunning([string]$dir) {
    $full = [IO.Path]::GetFullPath((Join-Path $Root $dir)).TrimEnd('\') + '\'
    $busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($full, [StringComparison]::OrdinalIgnoreCase) })
    if ($busy.Count -gt 0) { Fail "$($busy[0].Path) is running (a Strata server?). Stop it first: .venv\Scripts\python.exe strata.py launch --stop" }
}
Assert-NotRunning $BuildDir
if ($Vision) { Assert-NotRunning $VisionBuildDir }
if ($VisionGpu) { Assert-NotRunning "build\vision-cuda" }

# ---- the engine
# (the release flags are given: this tree's configure leaves CMAKE_CXX_FLAGS_RELEASE empty on MSVC, and the CPU experts then run without /O2)
$targets = "strata strata-nemotron"
if ($Tests) { $targets += " nemotron_kernels_parity nemotron_engine_parity nemotron_cpu_experts_test" }
$flags = '"-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG" "-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG"'
$tests_on = if ($Tests) { "ON" } else { "OFF" }
Say "configuring $BuildDir (sm $Arch) and building $targets with $Jobs jobs; this takes a while (log: build\build-nemotron.log)"
$rc = Invoke-Vs @(
    "cmake -S . -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=$tests_on -DSTRATA_PORTABLE=$port `"-DCMAKE_CUDA_ARCHITECTURES=$Arch`" $flags",
    "cmake --build `"$BuildDir`" --target $targets -j $Jobs"
) $log
if ($rc -ne 0) { Get-Content $log -Tail 30; Fail "the engine build failed (the end of the log is above)" }
$exe = Join-Path $Root "$BuildDir\strata.exe"
$fam = & cmd /c "set `"PATH=$cuda\bin;%PATH%`" && `"$exe`" --families" 2>&1 | Out-String
if ($fam -notmatch "nemotron_h_moe") { Fail "$exe was built but does not list nemotron_h_moe: $fam" }
Say "engine: $exe"

# ---- the picture reader
if ($Vision -or $VisionGpu) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) { Fail "git is needed to fetch llama.cpp for the picture reader" }
    $ld = Join-Path $Root "build\llama.cpp-vision"
    if (-not (Test-Path (Join-Path $ld ".git"))) {
        Say "fetching llama.cpp $LlamaVisionCommit into build\llama.cpp-vision"
        & git init -q $ld; & git -C $ld remote add origin https://github.com/ggml-org/llama.cpp.git
    }
    $have = (& git -C $ld rev-parse HEAD 2>$null)
    if ($have -ne $LlamaVisionCommit) {
        & git -C $ld fetch -q --depth 1 origin $LlamaVisionCommit
        if ($LASTEXITCODE -ne 0) { Fail "could not fetch llama.cpp $LlamaVisionCommit" }
        & git -C $ld checkout -q --detach FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { Fail "could not check out llama.cpp $LlamaVisionCommit" }
    }
    $ldc = $ld.Replace("\", "/")
    $vdir = $VisionBuildDir.Replace("\", "/")
    if ($Vision) {
        Say "building strata-vision (CPU) in $VisionBuildDir"
        $rc = Invoke-Vs @(
            "cmake -S tools/vision -B `"$vdir`" -G Ninja -DCMAKE_BUILD_TYPE=Release `"-DLLAMA_DIR=$ldc`" -DSTRATA_PORTABLE=$port",
            "cmake --build `"$vdir`" --target strata-vision -j $Jobs"
        ) (Join-Path $Root "build\build-vision-cpu.log")
        if ($rc -ne 0) { Get-Content (Join-Path $Root "build\build-vision-cpu.log") -Tail 30; Fail "the picture reader (CPU) build failed" }
    }
    if ($VisionGpu) {
        Say "building strata-vision (CUDA) in build\vision-cuda"
        $rc = Invoke-Vs @(
            "cmake -S tools/vision -B build/vision-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release `"-DLLAMA_DIR=$ldc`" -DSTRATA_PORTABLE=OFF -DSTRATA_VISION_CUDA=ON `"-DCMAKE_CUDA_ARCHITECTURES=$Arch`" -DGGML_CUDA_NCCL=OFF",
            "cmake --build build/vision-cuda --target strata-vision -j $Jobs"
        ) (Join-Path $Root "build\build-vision-cuda.log")
        if ($rc -ne 0) { Get-Content (Join-Path $Root "build\build-vision-cuda.log") -Tail 30; Fail "the picture reader (CUDA) build failed" }
    }
}
Say "done. Next: .venv\Scripts\python.exe strata.py launch --plan --family nemotron35 --model UD-IQ3_XXS"
