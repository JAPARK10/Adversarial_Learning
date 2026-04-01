import subprocess
import os
import sys
import re
import time
import numpy as np
from datetime import datetime

# Setup configurations
CONFIG_FILE = "configs/gcn/rfid.yaml"
RUNNER_SCRIPT = "main_runner.py"
OUTPUT_REPORT = "final_summary_report.txt"
REPETITIONS = 5

# Define Matrix
TEST_METHODS = [
    {"name": "Person-Exclusive", "env": {"USE_PERSON_EXCLUSIVE_SPLIT": "true", "USE_STRESS_TEST": "false"}},
    {"name": "Stress-Test", "env": {"USE_PERSON_EXCLUSIVE_SPLIT": "false", "USE_STRESS_TEST": "true"}}
]

ADV_COMBINATIONS = [
    {"name": "Baseline (AdvF, ConF)", "env": {"USE_ADVERSARIAL_LAYERS": "false", "USE_CONTRASTIVE_LEARNING": "false"}},
    {"name": "Adversarial Only (AdvT, ConF)", "env": {"USE_ADVERSARIAL_LAYERS": "true", "USE_CONTRASTIVE_LEARNING": "false"}},
    {"name": "Contrastive Only (AdvF, ConT)", "env": {"USE_ADVERSARIAL_LAYERS": "false", "USE_CONTRASTIVE_LEARNING": "true"}},
    {"name": "Synergistic (AdvT, ConT)", "env": {"USE_ADVERSARIAL_LAYERS": "true", "USE_CONTRASTIVE_LEARNING": "true"}}
]

def parse_metrics(output):
    metrics = {
        "best_acc": None,
        "best_f1": None,
        "overall_agg": None,
        "prec": None,
        "rec": None,
        "peak_acc_epoch": None
    }
    
    # regex for BEST TEST ACC
    # logging.info(f"[*] BEST TEST ACC    : {best_stats.get('accuracy', 0)*100:.2f}%")
    m_acc = re.search(r"\[\*\] BEST TEST ACC\s+:\s+([\d\.]+)%", output)
    if m_acc: metrics["best_acc"] = float(m_acc.group(1))
    
    # regex for BEST TEST F1
    # logging.info(f"[*] BEST TEST F1     : {best_stats.get('f1', 0):.4f}")
    m_f1 = re.search(r"\[\*\] BEST TEST F1\s+:\s+([\d\.]+)", output)
    if m_f1: metrics["best_f1"] = float(m_f1.group(1))
    
    # regex for OVERALL AGG
    # logging.info(f"{'OVERALL AGG':<12} | {final_avg*100:>8.2f}% | {total_samples:<8}")
    m_agg = re.search(r"OVERALL AGG\s+\|\s+([\d\.]+)%", output)
    if m_agg: metrics["overall_agg"] = float(m_agg.group(1))
    
    # regex for Prec/Rec from the last BEST epoch line
    # logging.info(f"> Epoch {cur_epoch}: ... F1: {test_f1:.4f} Prec: {test_pre:.4f} Rec: {test_rec:.4f}")
    # We find all occurrences and take the one that matches the best_acc/best_f1 session result
    prec_rec_matches = re.findall(r"F1:\s+([\d\.]+)\s+Prec:\s+([\d\.]+)\s+Rec:\s+([\d\.]+)", output)
    if prec_rec_matches:
        # Since we want the one corresponding to the best epoch, we'll look for the one that matches best_f1 or just take the last one logged as peak
        # In custom_train.py, it logs this every eval epoch.
        last_match = prec_rec_matches[-1]
        metrics["prec"] = float(last_match[1])
        metrics["rec"] = float(last_match[2])
        
    return metrics

def run_suite():
    with open(OUTPUT_REPORT, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"      INVARIANT RECOGNITION RESEARCH: 40-RUN BATCH REPORT\n")
        f.write(f"      Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

    print(f"[*] Starting 40-run matrix automation...")
    
    for method in TEST_METHODS:
        print(f"\n>>> TESTING DOMAIN: {method['name']}")
        
        with open(OUTPUT_REPORT, "a") as f:
            f.write(f"\n### DOMAIN: {method['name']} ###\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Config':<35} | {'Run':<4} | {'BestAcc':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'Agg'}\n")
            f.write("-" * 80 + "\n")

        for combo in ADV_COMBINATIONS:
            print(f"  > Configuration: {combo['name']}")
            
            run_results = []
            
            for seed in range(REPETITIONS):
                print(f"    - Seed {seed}... ", end="", flush=True)
                
                # Setup environment
                current_env = os.environ.copy()
                current_env.update(method["env"])
                current_env.update(combo["env"])
                
                cmd = [sys.executable, RUNNER_SCRIPT, "--cfg", CONFIG_FILE, "--repeat", "1", "seed", str(seed)]
                
                try:
                    result = subprocess.run(cmd, env=current_env, capture_output=True, text=True, check=True)
                    metrics = parse_metrics(result.stdout + result.stderr)
                    run_results.append(metrics)
                    
                    with open(OUTPUT_REPORT, "a") as f:
                        f.write(f"{combo['name']:<35} | {seed:<4} | "
                                f"{metrics['best_acc'] or 0.0:>8.2f}% | "
                                f"{metrics['best_f1'] or 0.0:>8.4f} | "
                                f"{metrics['prec'] or 0.0:>8.4f} | "
                                f"{metrics['rec'] or 0.0:>8.4f} | "
                                f"{metrics['overall_agg'] or 0.0:>8.2f}%\n")
                    
                    print("DONE.")
                except subprocess.CalledProcessError as e:
                    print(f"FAILED (Seed {seed})")
                    print(f"--- ERROR LOG (Seed {seed}) ---")
                    print(e.stdout)
                    print(e.stderr)
                    print("-" * 30)
                    with open(OUTPUT_REPORT, "a") as f:
                        f.write(f"{combo['name']:<35} | {seed:<4} | ERROR\n")
            
            # Compute Average for this combo
            if run_results:
                valid_accs = [r["best_acc"] for r in run_results if r["best_acc"] is not None]
                valid_f1s = [r["best_f1"] for r in run_results if r["best_f1"] is not None]
                if valid_accs:
                    avg_acc = np.mean(valid_accs)
                    avg_f1 = np.mean(valid_f1s)
                    with open(OUTPUT_REPORT, "a") as f:
                        f.write(f"{'--> MEAN:':<35} | {'AVG':<4} | {avg_acc:>8.2f}% | {avg_f1:>8.4f} | {'-':>8} | {'-':>8} | {'-'}\n")
                        f.write("-" * 80 + "\n")

    with open(OUTPUT_REPORT, "a") as f:
        f.write(f"\n[!] Complete Experiment Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    run_suite()
