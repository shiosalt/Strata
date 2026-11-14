<#
.SYNOPSIS
  Makes the ready-made Windows engine for Nemotron 3.5 Super: dist\strata-nemotron-<version>-windows-x64-cuda13.zip.

.DESCRIPTION
  Takes a portable build (tools\nemotron\build.ps1 -Portable -Vision -BuildDir build\nemotron-release -VisionBuildDir build\vision-release)
  and puts into one zip, under engine-nemotron\: strata.exe, strata-nemotron.exe, strata-vision.exe (the CPU picture reader), the CUDA
  libraries they load (cuBLAS, from the CUDA toolkit; NVIDIA lets them be redistributed with an application), the licenses, a README and
  SHA256SUMS.txt. Unzipped into a checkout of this repository it is the engine `strata.py launch --family nemotron35` finds.
  Nothing is uploaded.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\nemotron\package.ps1 -Version 0.1.0
#>
[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [string]$BuildDir = "build\nemotron-release",
    [string]$VisionBuildDir = "build\vision-release",
    [string]$Out = "dist"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
function Fail($m) { Write-Host "[package-nemotron] ERROR: $m" -ForegroundColor Red; exit 1 }
function Say($m) { Write-Host "[package-nemotron] $m" }

$cuda = $env:CUDA_PATH
if (-not $cuda -or -not (Test-Path (Join-Path $cuda "bin\nvcc.exe"))) {
    $cuda = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "v13*" -and (Test-Path (Join-Path $_.FullName "bin\nvcc.exe")) } |
            Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cuda) { Fail "the CUDA 13 toolkit (its cuBLAS DLLs go into the package) was not found" }
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

$files = [ordered]@{
    "strata.exe"          = Join-Path $Root "$BuildDir\strata.exe"
    "strata-nemotron.exe" = Join-Path $Root "$BuildDir\strata-nemotron.exe"
    "strata-vision.exe"   = Join-Path $Root "$VisionBuildDir\bin\strata-vision.exe"
}
foreach ($k in $files.Keys) { if (-not (Test-Path $files[$k])) { Fail "$($files[$k]) is missing: build it first (see the top of this script)" } }
# a portable build: the static MSVC runtime and the AVX2 baseline (STRATA_PORTABLE=ON in its CMake cache)
foreach ($d in @($BuildDir, $VisionBuildDir)) {
    $cache = Join-Path $Root "$d\CMakeCache.txt"
    if (-not (Select-String -Path $cache -Pattern '^STRATA_PORTABLE:BOOL=ON' -Quiet)) { Fail "$d is not a portable build (STRATA_PORTABLE=ON)" }
}
$arch = (Select-String -Path (Join-Path $Root "$BuildDir\CMakeCache.txt") -Pattern '^CMAKE_CUDA_ARCHITECTURES:[A-Z]+=(.*)$').Matches[0].Groups[1].Value

# the DLLs the programs load that Windows does not have: every one must be found beside them or in the CUDA toolkit
function Get-Deps([string]$exe) {
    $o = & cmd /c "call `"$vcvars`" >nul && dumpbin /nologo /dependents `"$exe`""
    return @($o | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^[A-Za-z0-9_.-]+\.dll$' })
}
$system = '^(kernel32|advapi32|user32|gdi32|shell32|ole32|oleaut32|ws2_32|bcrypt|ntdll|winmm|dbghelp|psapi|shlwapi|comdlg32|winspool|uuid|nvcuda|iphlpapi|setupapi|version|crypt32|secur32|userenv)\.dll$|^api-ms-win-'
$need = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($v in $files.Values) { $queue.Enqueue($v) }
$extra = [ordered]@{}
while ($queue.Count -gt 0) {
    foreach ($dll in (Get-Deps $queue.Dequeue())) {
        if ($dll -match $system -or -not $need.Add($dll)) { continue }
        $src = Join-Path $cuda "bin\x64\$dll"
        if (-not (Test-Path $src)) { $src = Join-Path $cuda "bin\$dll" }
        if (-not (Test-Path $src)) { Fail "$dll (needed by the programs) is neither a Windows DLL nor in $cuda\bin: the build is not portable" }
        $extra[$dll] = $src
        $queue.Enqueue($src)
    }
}

$name = "strata-nemotron-$Version-windows-x64-cuda13"
$stage = Join-Path $Root "build\package\$name"
$eng = Join-Path $stage "engine-nemotron"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $eng | Out-Null
foreach ($k in $files.Keys) { Copy-Item $files[$k] (Join-Path $eng $k) }
foreach ($k in $extra.Keys) { Copy-Item $extra[$k] (Join-Path $eng $k) }
Copy-Item (Join-Path $Root "LICENSE") (Join-Path $eng "LICENSE.txt")
$eula = Join-Path $cuda "EULA.txt"
if (Test-Path $eula) { Copy-Item $eula (Join-Path $eng "CUDA-EULA.txt") } else { Fail "$eula is missing (the CUDA libraries are shipped under it)" }
$commit = (& git -C $Root rev-parse HEAD).Trim()
$dirty = if ((& git -C $Root status --porcelain --untracked-files=no) -ne $null) { " (with uncommitted changes)" } else { "" }
$llama = (& git -C (Join-Path $Root "build\llama.cpp-vision") rev-parse HEAD).Trim()
$cudaver = Split-Path $cuda -Leaf
$readme = @"
Strata fork: the Nemotron 3.5 Super engine for Windows ($name)
https://github.com/shiosalt/Strata (an unofficial fork of https://github.com/Niko1221/Strata)

Unzip this into a checkout of the repository above (it makes engine-nemotron\). Then:
  .venv\Scripts\python.exe strata.py launch --family nemotron35 --model UD-IQ3_XXS --yes --no-tui
The steps are in the repository's README ("Install").

  strata.exe            the engine (strata --families lists nemotron_h_moe)
  strata-nemotron.exe   the same engine as a command (--bench-cpu, tests)
  strata-vision.exe     the picture reader (CPU), built against llama.cpp $llama
  $(($extra.Keys) -join ', ')   from the CUDA toolkit $cudaver, under CUDA-EULA.txt

Built from commit $commit$dirty, CUDA $cudaver, GPU code for $arch, AVX2 baseline, static MSVC runtime.
Needs: Windows 10/11 x64, an NVIDIA driver 580 or newer, a CPU with AVX2.
Checked only on Windows 11 with 2 x RTX 5060 Ti 16 GB (sm_120). RTX 30 / 40 (sm_86 / sm_89) code is in it but was not tried.
Strata is MIT licensed (LICENSE.txt). The model's weights are NVIDIA's and are not in this package.
"@
Set-Content -Path (Join-Path $eng "README.txt") -Value $readme -Encoding UTF8
# the programs start (no GPU work): with PATH holding no CUDA folder, so a DLL missing from the package shows here
$clean = ($env:PATH -split ';' | Where-Object { $_ -and $_ -notmatch 'CUDA' }) -join ';'
$fam = & cmd /c "set `"PATH=$clean`" && `"$eng\strata.exe`" --families" 2>&1 | Out-String
if ($fam -notmatch "nemotron_h_moe") { Fail "the packaged strata.exe did not list nemotron_h_moe without the CUDA toolkit on PATH: $fam" }
& cmd /c "set `"PATH=$clean`" && `"$eng\strata-vision.exe`" --help >nul 2>&1"   # (it answers an unknown argument; it must only start)
if ($LASTEXITCODE -eq -1073741515) { Fail "the packaged strata-vision.exe misses a DLL (0xC0000135)" }
$sums = Get-ChildItem $eng -File | Sort-Object Name | ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name }
Set-Content -Path (Join-Path $eng "SHA256SUMS.txt") -Value $sums -Encoding ASCII
New-Item -ItemType Directory -Force -Path (Join-Path $Root $Out) | Out-Null
$zip = Join-Path $Root "$Out\$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $eng -DestinationPath $zip -CompressionLevel Optimal
$h = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
Set-Content -Path "$zip.sha256" -Value "$h  $name.zip" -Encoding ASCII
Say ("{0} ({1:N0} MB), SHA-256 {2}" -f $zip, ((Get-Item $zip).Length / 1MB), $h)
