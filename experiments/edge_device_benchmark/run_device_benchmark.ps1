<#
.SYNOPSIS
    Collects on-device BioDINO-Edge edge benchmarks from a physical Android phone.

.DESCRIPTION
    Installs the research 'benchmark' APK, pushes the Tier-2 companion model,
    resets battery stats, launches BenchmarkActivity, waits for completion,
    pulls benchmark_report.json + batterystats, and generates LaTeX tables.

    Build the APK first (from android_app/):
        .\gradlew.bat assembleBenchmark
    Output: android_app\app\build\outputs\apk\benchmark\BioDINO-Edge-v1.0.1-benchmark.apk

.EXAMPLE
    .\run_device_benchmark.ps1 -SustainedSeconds 600
    .\run_device_benchmark.ps1 -SkipInstall -SustainedSeconds 120   # quick re-run
#>
param(
    [string]$Adb = "adb",
    [string]$ApkPath = "",
    [string]$CompanionPath = "",
    [int]$SustainedSeconds = 600,
    [int]$NIters = 100,
    [int]$CooldownSeconds = 60,
    [string]$OutDir = "",
    [switch]$SkipInstall,
    [switch]$SkipEnergy
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
if ($OutDir -eq "") { $OutDir = Join-Path $ScriptDir "results" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Resolve adb: use PATH if available, otherwise the Android SDK platform-tools copy.
if ($Adb -eq "adb") {
    if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
        $candidate = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
        if (Test-Path $candidate) { $Adb = $candidate }
    }
}
Write-Host "Using adb: $Adb"

if ($ApkPath -eq "") {
    $ApkPath = Join-Path $RepoRoot "android_app\app\build\outputs\apk\benchmark\BioDINO-Edge-v1.0.1-benchmark.apk"
}
if ($CompanionPath -eq "") {
    $CompanionPath = Join-Path $RepoRoot "biodino_companion_int8.onnx"
}

$Package = "com.biodino.edge"
$Activity = "$Package/.benchmark.BenchmarkActivity"
$RemoteDir = "/sdcard/Android/data/$Package/files"
$RemoteReport = "$RemoteDir/benchmark_report.json"
$RemoteCompanion = "$RemoteDir/biodino_companion_int8.onnx"

function Invoke-Adb {
    param([string[]]$AdbArgs)
    & $Adb @AdbArgs
    if ($LASTEXITCODE -ne 0) { throw "adb failed: $($AdbArgs -join ' ')" }
}

Write-Host "== BioDINO-Edge on-device benchmark ==" -ForegroundColor Cyan

# 1. Device check
$devices = (& $Adb devices) -join "`n"
Write-Host $devices
if (-not ($devices -match "\bdevice\b")) { throw "No adb device found. Connect a phone with USB debugging enabled." }

# 2. Install
if (-not $SkipInstall) {
    if (-not (Test-Path $ApkPath)) {
        throw "APK not found: $ApkPath`nBuild it first: cd android_app; .\gradlew.bat assembleBenchmark"
    }
    Write-Host "Installing $ApkPath"
    Invoke-Adb @("install", "-r", $ApkPath)
}

# 3. Push companion model (Tier-2)
if (Test-Path $CompanionPath) {
    Write-Host "Pushing companion model -> $RemoteCompanion"
    Invoke-Adb @("shell", "mkdir", "-p", $RemoteDir)
    Invoke-Adb @("push", $CompanionPath, $RemoteCompanion)
} else {
    Write-Warning "Companion not found at $CompanionPath; Tier-2 + adaptation stages will be skipped."
}

# 4. Reset battery stats for energy estimation
if (-not $SkipEnergy) {
    Write-Host "Resetting batterystats"
    & $Adb shell dumpsys batterystats --reset | Out-Null
}

# 5. Launch benchmark
& $Adb logcat -c
Write-Host "Launching benchmark (sustained=${SustainedSeconds}s, iters=$NIters, cooldown=${CooldownSeconds}s)"
Invoke-Adb @("shell", "am", "start", "-n", $Activity,
    "--ei", "n_iters", "$NIters",
    "--ei", "sustained_seconds", "$SustainedSeconds",
    "--ei", "cooldown_seconds", "$CooldownSeconds")

# 6. Wait for BENCHMARK_DONE in logcat
$timeout = $SustainedSeconds + 600
$elapsed = 0
$done = $false
while ($elapsed -lt $timeout) {
    Start-Sleep -Seconds 5
    $elapsed += 5
    $log = (& $Adb logcat -d -s BioDinoBench:*) -join "`n"
    if ($log -match "BENCHMARK_DONE") { $done = $true; break }
    if ($log -match "ERROR:") { Write-Warning "Benchmark reported an error; see logcat." }
    Write-Host "  ...running (${elapsed}s / ${timeout}s)"
}
if (-not $done) { Write-Warning "Timed out waiting for BENCHMARK_DONE; attempting to pull partial report." }

# 7. Pull report + batterystats
$LocalReport = Join-Path $OutDir "benchmark_report.json"
Invoke-Adb @("pull", $RemoteReport, $LocalReport)
$LocalBattery = Join-Path $OutDir "batterystats.txt"
if (-not $SkipEnergy) {
    & $Adb shell dumpsys batterystats | Out-File -FilePath $LocalBattery -Encoding utf8
}

# 8. Generate LaTeX tables
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$MakeTables = Join-Path $ScriptDir "make_tables.py"
$TablesOut = Join-Path $OutDir "edge_device_tables.tex"
$genArgs = @($MakeTables, "--report", $LocalReport, "--out", $TablesOut)
if (-not $SkipEnergy -and (Test-Path $LocalBattery)) {
    $genArgs += @("--batterystats", $LocalBattery)
    $pkgInfo = (& $Adb shell dumpsys package $Package) -join "`n"
    if ($pkgInfo -match "userId=(\d+)") { $genArgs += @("--uid", $Matches[1]) }
}
& $Python @genArgs

Write-Host "== Done ==" -ForegroundColor Green
Write-Host "Report : $LocalReport"
Write-Host "Tables : $TablesOut"