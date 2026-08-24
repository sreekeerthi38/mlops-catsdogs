<#
.SYNOPSIS
  M4 smoke test (Windows / PowerShell): verify /health, /ready and one /predict.

.DESCRIPTION
  Exits with code 1 on any failure so a CI/CD pipeline fails the deploy.

  Uses curl.exe (not the `curl` alias, which is Invoke-WebRequest in Windows
  PowerShell and takes entirely different flags). curl.exe ships with
  Windows 10 1803+ and Windows 11.

.EXAMPLE
  .\scripts\smoke_test.ps1
  .\scripts\smoke_test.ps1 -BaseUrl http://localhost:8000 -Sample samples\cat.jpg
#>
param(
    [string]$BaseUrl = $(if ($env:BASE_URL) { $env:BASE_URL } else { "http://localhost:8000" }),
    [string]$Sample  = "samples\cat.jpg"
)

$ErrorActionPreference = "Stop"

function Fail($msg) {
    Write-Host "[smoke] FAIL: $msg" -ForegroundColor Red
    exit 1
}

# curl.exe must resolve to the real binary, not the PowerShell alias.
$curl = (Get-Command curl.exe -ErrorAction SilentlyContinue)
if (-not $curl) { Fail "curl.exe not found. Windows 10 1803+ ships it; otherwise install curl." }
$curl = $curl.Source

Write-Host "[smoke] target = $BaseUrl"

# --- 1) liveness -------------------------------------------------------------
Write-Host "[smoke] GET /health"
$health = & $curl -fsS "$BaseUrl/health" 2>$null
if ($LASTEXITCODE -ne 0) { Fail "/health did not respond" }
Write-Host "  -> $health"
if ($health -notmatch '"status"\s*:\s*"ok"') { Fail "health not ok" }

# --- 2) readiness (503 until the model is loaded) ----------------------------
Write-Host "[smoke] GET /ready"
$ready = & $curl -fsS "$BaseUrl/ready" 2>$null
if ($LASTEXITCODE -ne 0) { Fail "service not ready - model missing or still loading" }
Write-Host "  -> $ready"

# --- 3) prediction on a real image where available ---------------------------
$cleanup = $false
if (Test-Path $Sample) {
    $img = $Sample
    Write-Host "[smoke] using sample image $Sample"
} else {
    Write-Host "[smoke] WARNING: $Sample not found - generating a noise image." -ForegroundColor Yellow
    Write-Host "[smoke] A noise prediction proves the HTTP wiring only. Put a real" -ForegroundColor Yellow
    Write-Host "[smoke] cat photo at $Sample before recording the demo." -ForegroundColor Yellow
    $img = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.jpg'
    $cleanup = $true
    python -c "import sys, numpy as np; from PIL import Image; Image.fromarray(np.random.randint(0,256,(224,224,3),dtype=np.uint8)).save(sys.argv[1])" $img
    if ($LASTEXITCODE -ne 0) { Fail "could not generate a fallback image" }
}

Write-Host "[smoke] POST /predict"
$pred = & $curl -fsS -F "file=@$img;type=image/jpeg" "$BaseUrl/predict" 2>$null
if ($LASTEXITCODE -ne 0) { Fail "/predict returned an error status" }
Write-Host "  -> $pred"
if ($pred -notmatch '"label"') { Fail "no label in prediction response" }

if ($cleanup) { Remove-Item $img -Force -ErrorAction SilentlyContinue }

Write-Host "[smoke] PASS" -ForegroundColor Green
exit 0
