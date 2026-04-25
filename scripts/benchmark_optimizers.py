import subprocess
import re
import os
import sys
from collections import defaultdict

# Model Configuration
BASE_CMD = [
    sys.executable, "-m", "scripts.base_train",
    "--depth", "4",
    "--aspect-ratio", "32",
    "--max-seq-len", "512",
    "--device-batch-size", "1",
    "--total-batch-size", "1024",
    "--num-iterations", "1000",
    "--run", "dummy",
    "--warmup-steps", "40",
    "--warmdown-ratio", "0.1",
    "--eval-every", "-1",
    "--sample-every", "-1",
    "--core-metric-every", "-1",
    "--save-every", "-1",
]

OPTIMIZERS = ["muon", "adamw_all", "vo_product_muon", "paired_ffn"]
ACTIVATIONS = ["relu", "relu_squared"]
LOG_DIR = "bench_logs"

def get_log_path(optimizer, activation):
    return os.path.join(LOG_DIR, f"bench_{optimizer}_{activation}.log")

def run_bench(optimizer, activation):
    log_path = get_log_path(optimizer, activation)
    
    if os.path.exists(log_path):
        print(f">>> Skipping {optimizer}/{activation} - Log already exists: {log_path}")
        return

    cmd = BASE_CMD + [
        "--optimizer-kind", optimizer,
        "--mlp-activation", activation
    ]
    
    # Auto-sync paired_ffn gate to derivative
    if optimizer == "paired_ffn":
        cmd += ["--ffn-pair-warmup-steps", "0"]
        if activation == "relu_squared":
            cmd += ["--ffn-pair-activation-kind", "relu2"]
        else:
            cmd += ["--ffn-pair-activation-kind", "relu"]

    print(f"\n>>> Running: Optimizer={optimizer}, Activation={activation}")
    print(f">>> Log: {log_path}")
    
    with open(log_path, "w", buffering=1) as f:
        # Run process and stream output to both stdout and file
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        process.wait()

def parse_best_loss(log_path):
    if not os.path.exists(log_path):
        return None
    
    best_loss = float('inf')
    loss_pattern = re.compile(r"loss: ([\d\.]+)")
    
    with open(log_path, "r") as f:
        for line in f:
            match = loss_pattern.search(line)
            if match:
                loss = float(match.group(1))
                if loss < best_loss:
                    best_loss = loss
                    
    return best_loss if best_loss != float('inf') else None

def generate_summary():
    print("\n\n" + "="*80)
    print("BENCHMARK SUMMARY (Best Loss Achievement)")
    print("="*80)
    print(f"{'Optimizer':<20} | {'Activation':<15} | {'Best Loss':<10}")
    print("-" * 50)
    
    results = defaultdict(dict)
    for opt in OPTIMIZERS:
        for act in ACTIVATIONS:
            log_path = get_log_path(opt, act)
            best_loss = parse_best_loss(log_path)
            results[opt][act] = best_loss
            
            loss_str = f"{best_loss:.6f}" if best_loss is not None else "N/A"
            print(f"{opt:<20} | {act:<15} | {loss_str:<10}")

def main():
    iterations = "1000000"
    activations = ["relu_squared"]
    
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        iterations = sys.argv[1]
    
    if "--summary" in sys.argv:
        generate_summary()
        return

    # Update BASE_CMD with new iterations
    for i, arg in enumerate(BASE_CMD):
        if arg == "--num-iterations":
            BASE_CMD[i+1] = iterations
        if arg == "--warmup-steps":
            # Scale warmup steps linearly: 40 for 1000 -> 40000 for 1M
            BASE_CMD[i+1] = str(int(int(iterations) * 0.04))

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    try:
        for opt in OPTIMIZERS:
            for act in activations:
                run_bench(opt, act)
    finally:
        generate_summary()

if __name__ == "__main__":
    main()
