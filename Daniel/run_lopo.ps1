$subjects = @("DE 3m", "FQ 3m", "GM 3m", "IM 3m", "JH 3m", "LJ 3m", "LR 3m", "MA 3m", "MS 3m", "NH 3m", "NI 3m", "NS 3m", "RB 3m", "SG 3m", "SH 3m", "SS 3m")

# Create results directory if it doesn't exist
if (!(Test-Path "results_lopo_1.0")) {
    New-Item -ItemType Directory -Force -Path "results_lopo_1.0"
}

for ($i = 0; $i -lt $subjects.Length; $i++) {
    $test_subj = $subjects[$i]
    $val_subj = $subjects[($i + 1) % $subjects.Length]
    
    Write-Host "`n`n================================================================="
    Write-Host ">>> Running Fold $($i+1)/$($subjects.Length): Test=$test_subj, Val=$val_subj"
    Write-Host "================================================================="
    
    # We strip spaces from folder name to create a safe output directory name
    $safe_name = $test_subj.Replace(" ", "_")
    $out_dir = "results_lopo_1.0/$safe_name"

    # Run the training
    # We pass overrides directly to GraphGym: out_dir, dataset.test_subj, dataset.val_subj
    python main.py --cfg configs/gcn/rfid_lopo.yaml out_dir $out_dir dataset.test_subj "$test_subj" dataset.val_subj "$val_subj"
    
    # We rely on the script cleanly finishing epochs rather than checking Python's 
    # garbage collection exit codes (PyTorch Windows teardown issue).

}

Write-Host "`n`nLOPO Cross-Validation Complete!"
