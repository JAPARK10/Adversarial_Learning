import subprocess
import os
import sys

# Experiment parameters
lambdas = [0.1, 0.3, 0.5, 1.0]
base_cfg = r"configs/gcn/rfid_lopo_adv.yaml"
working_dir = r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main"

def run_experiment(lambda_val):
    out_dir = f"results_lopo_adv_lambda_{lambda_val}"
    out_path = os.path.join(working_dir, out_dir)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created directory: {out_path}")
    
    print(f"\n" + "="*60)
    print(f"STARTING EXPERIMENT: lambda_u = {lambda_val}")
    print(f"OUTPUT DIRECTORY: {out_dir}")
    print("="*60 + "\n")
    
    # Construct the command
    # Using 'out_dir' as a positional argument at the end for YACS/GraphGym
    cmd = [
        "python", "main.py",
        "--cfg", base_cfg,
        "adv.lambda_u", str(lambda_val),
        "out_dir", out_dir
    ]
    
    # Run the process and stream output to console
    process = subprocess.Popen(
        cmd,
        cwd=working_dir,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True
    )
    
    exit_code = process.wait()
    
    if exit_code != 0:
        print(f"Experiment for lambda {lambda_val} failed with exit code {exit_code}")
    else:
        print(f"Experiment for lambda {lambda_val} completed successfully.")

if __name__ == "__main__":
    for l in lambdas:
        run_experiment(l)
    print("\nALL EXPERIMENTS COMPLETED.")
