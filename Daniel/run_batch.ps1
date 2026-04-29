# Automated Batch Execution for Comparative Experiments

# Ensure we are in the correct directory
$root = Get-Location
$yamlPath = Join-Path $root "configs/gcn/rfid_lopo.yaml"
$lopoScript = Join-Path $root "run_lopo.ps1"

# --- Experiment 1: Lambda 0.5 ---
Write-Host "`n`n>>> [BATCH] STARTING EXPERIMENT 1: LAMBDA 0.5" -ForegroundColor Cyan

# Update Config: Set lambda to 0.5 and enable adversarial part
(Get-Content $yamlPath) | ForEach-Object {
    if ($_ -match "lambda_u:") { "  lambda_u: 0.5" }
    elseif ($_ -match "use:" -and $in_adv) { "  use: True" }
    else { 
        if ($_ -match "adv:") { $global:in_adv = $true }
        $_ 
    }
} | Set-Content $yamlPath
$global:in_adv = $false

# Update Script: Change output folder
(Get-Content $lopoScript) -replace "results_lopo_0.05", "results_lopo_0.5" | Set-Content $lopoScript

# Execute Experiment 1
powershell -ExecutionPolicy Bypass -File .\run_lopo.ps1


# --- Experiment 2: Lambda 1.0 ---
Write-Host "`n`n>>> [BATCH] STARTING EXPERIMENT 2: LAMBDA 1.0" -ForegroundColor Cyan

# Update Config: Set lambda to 1.0
(Get-Content $yamlPath) | ForEach-Object {
    if ($_ -match "lambda_u:") { "  lambda_u: 1.0" }
    elseif ($_ -match "use:" -and $in_adv) { "  use: True" }
    else { 
        if ($_ -match "adv:") { $global:in_adv = $true }
        $_ 
    }
} | Set-Content $yamlPath
$global:in_adv = $false

# Update Script: Change output folder
(Get-Content $lopoScript) -replace "results_lopo_0.5", "results_lopo_1.0" | Set-Content $lopoScript

# Execute Experiment 2
powershell -ExecutionPolicy Bypass -File .\run_lopo.ps1

Write-Host "`n`n>>> [BATCH] ALL EXPERIMENTS COMPLETE!" -ForegroundColor Green
