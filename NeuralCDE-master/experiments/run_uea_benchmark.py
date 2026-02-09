"""
UEA CharacterTrajectories Benchmark Script
Neural CDE Project

Benchmarks ncde (modified) vs odernn (baseline) on CharacterTrajectories
with missing rates 0.3, 0.5, 0.7. Runs 3 repeats per configuration and
prints a summary table of Accuracy (mean +/- std) and Memory Usage.

Usage:
    python run_uea_benchmark.py
    python run_uea_benchmark.py --cpu
    python run_uea_benchmark.py --repeats 5
    python run_uea_benchmark.py --models ncde odernn gruode
    python run_uea_benchmark.py --dry-run
"""

import sys
import os
import argparse
import csv
import json
import time
from datetime import datetime
import tqdm

# Setup paths - this script lives in experiments/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import numpy as np
import torch


# Default hyperparameters per model (from uea.run_all)
MODEL_CONFIGS = {
    'ncde': dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
    'ncde-film': dict(hidden_channels=32, hidden_hidden_channels=None, num_hidden_layers=None),
    'ncde-spectral': dict(hidden_channels=32, hidden_hidden_channels=None, num_hidden_layers=None),
    'odernn': dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
    'dt': dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
    'decay': dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
    'gruode': dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
}

VALID_MODELS = list(MODEL_CONFIGS.keys())


def parse_args():
    parser = argparse.ArgumentParser(description='UEA CharacterTrajectories Benchmark')
    parser.add_argument('--dataset', type=str, default='CharacterTrajectories',
                        help='UEA dataset name (default: CharacterTrajectories)')
    parser.add_argument('--missing-rates', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                        help='Missing data rates (default: 0.3 0.5 0.7)')
    parser.add_argument('--models', type=str, nargs='+', default=['ncde', 'odernn'],
                        choices=VALID_MODELS,
                        help='Models to benchmark (default: ncde odernn)')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of repeats per config (default: 3)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs (default: 1000)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test mode, do not save results')
    parser.add_argument('--hidden-channels', type=int, default=None,
                        help='Override hidden_channels for all models')
    parser.add_argument('--hidden-hidden-channels', type=int, default=None,
                        help='Override hidden_hidden_channels for all models')
    parser.add_argument('--num-hidden-layers', type=int, default=None,
                        help='Override num_hidden_layers for all models')
    return parser.parse_args()


def get_device(force_cpu=False):
    if force_cpu:
        return 'cpu'
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return 'cuda'
    print("[WARN] CUDA not available, using CPU")
    return 'cpu'


def run_single(dataset_name, missing_rate, device, max_epochs, model_name, dry_run, config):
    """Run a single experiment and return the result."""
    import uea
    result = uea.main(
        dataset_name=dataset_name,
        missing_rate=missing_rate,
        device=device,
        max_epochs=max_epochs,
        model_name=model_name,
        dry_run=dry_run,
        **config
    )
    return result


def save_benchmark_results(all_results, run_logs, model_names, missing_rates, args):
    """Save detailed run logs and summary to results/benchmark/ directory."""
    results_dir = os.path.join(script_dir, 'results', 'benchmark')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{args.dataset}_{timestamp}"

    # 1) Save per-run detailed logs as JSON
    json_path = os.path.join(results_dir, f"{base_name}_detail.json")
    detail = {
        'config': {
            'dataset': args.dataset,
            'missing_rates': missing_rates,
            'models': model_names,
            'repeats': args.repeats,
            'max_epochs': args.epochs,
            'dry_run': args.dry_run,
            'timestamp': timestamp,
        },
        'runs': run_logs,
        'summary': {},
    }
    for model_name in model_names:
        for mr in missing_rates:
            key = (model_name, mr)
            if key not in all_results:
                continue
            accs = all_results[key]['accuracies']
            mems = all_results[key]['memories']
            valid_mems = [m for m in mems if m is not None]
            detail['summary'][f"{model_name}_miss{int(mr*100)}"] = {
                'acc_mean': float(np.mean(accs)),
                'acc_std': float(np.std(accs)),
                'acc_all': [float(a) for a in accs],
                'memory_mean_MB': float(np.mean(valid_mems) / 1e6) if valid_mems else None,
            }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    # 2) Save summary as CSV (easy to paste into papers / spreadsheets)
    csv_path = os.path.join(results_dir, f"{base_name}_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'MissingRate', 'AccMean', 'AccStd', 'MemoryMB'])
        for model_name in model_names:
            for mr in missing_rates:
                key = (model_name, mr)
                if key not in all_results:
                    continue
                accs = all_results[key]['accuracies']
                mems = all_results[key]['memories']
                valid_mems = [m for m in mems if m is not None]
                mem_val = f"{np.mean(valid_mems) / 1e6:.1f}" if valid_mems else ""
                writer.writerow([
                    model_name,
                    f"{mr:.1f}",
                    f"{np.mean(accs):.4f}",
                    f"{np.std(accs):.4f}",
                    mem_val,
                ])

    print(f"\n[SAVED] Detail  -> {json_path}")
    print(f"[SAVED] Summary -> {csv_path}")


def print_summary_table(all_results, model_names, missing_rates):
    """Print a formatted summary table."""
    sep = "=" * 80
    header_sep = "-" * 80

    print(f"\n{sep}")
    print("BENCHMARK SUMMARY: CharacterTrajectories")
    print(sep)
    print(f"{'Model':<10} {'Missing%':<10} {'Accuracy (mean +/- std)':<30} {'Memory (MB)':<15}")
    print(header_sep)

    for model_name in model_names:
        for mr in missing_rates:
            key = (model_name, mr)
            if key not in all_results:
                continue

            accuracies = all_results[key]['accuracies']
            memories = all_results[key]['memories']

            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies)

            valid_memories = [m for m in memories if m is not None]
            if valid_memories:
                mem_str = f"{np.mean(valid_memories) / 1e6:.1f}"
            else:
                mem_str = "N/A (CPU)"

            print(f"{model_name:<10} {mr:<10.0%} {acc_mean:.4f} +/- {acc_std:.4f}             {mem_str:<15}")

        print(header_sep)

    print(sep)


def main():
    args = parse_args()
    device = get_device(args.cpu)

    dataset_name = args.dataset
    missing_rates = args.missing_rates
    model_names = args.models
    repeats = args.repeats

    total_runs = len(model_names) * len(missing_rates) * repeats
    print(f"\n[CONFIG] Dataset: {dataset_name}")
    print(f"[CONFIG] Models: {model_names}")
    print(f"[CONFIG] Missing rates: {missing_rates}")
    print(f"[CONFIG] Repeats: {repeats}")
    print(f"[CONFIG] Max epochs: {args.epochs}")
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Total runs: {total_runs}")
    print()

    # Collect results: key=(model_name, missing_rate) -> {accuracies: [], memories: []}
    all_results = {}

    # Build flat list of all (model, missing_rate, repeat) combinations
    run_configs = []
    for model_name in model_names:
        config = MODEL_CONFIGS[model_name].copy()
        if args.hidden_channels is not None:
            config['hidden_channels'] = args.hidden_channels
        if args.hidden_hidden_channels is not None:
            config['hidden_hidden_channels'] = args.hidden_hidden_channels
        if args.num_hidden_layers is not None:
            config['num_hidden_layers'] = args.num_hidden_layers
        for mr in missing_rates:
            key = (model_name, mr)
            all_results[key] = {'accuracies': [], 'memories': []}
            for r in range(repeats):
                run_configs.append((model_name, mr, r, config, key))

    run_logs = []

    benchmark_pbar = tqdm.tqdm(run_configs, desc="Benchmark Progress", unit="run")
    for model_name, mr, r, config, key in benchmark_pbar:
        benchmark_pbar.set_description(
            f"[{model_name}|miss={mr:.0%}|rep={r+1}/{repeats}]"
        )
        benchmark_pbar.set_postfix(model=model_name, missing=f"{mr:.0%}", repeat=f"{r+1}/{repeats}")

        t_start = time.time()

        result = run_single(
            dataset_name=dataset_name,
            missing_rate=mr,
            device=device,
            max_epochs=args.epochs,
            model_name=model_name,
            dry_run=args.dry_run,
            config=config,
        )

        elapsed = time.time() - t_start
        acc = result.test_metrics.accuracy
        mem = result.memory_usage

        all_results[key]['accuracies'].append(acc)
        all_results[key]['memories'].append(mem)

        run_logs.append({
            'model': model_name,
            'missing_rate': mr,
            'repeat': r + 1,
            'test_accuracy': float(acc),
            'train_accuracy': float(result.train_metrics.accuracy),
            'val_accuracy': float(result.val_metrics.accuracy),
            'memory_bytes': mem,
            'parameters': result.parameters,
            'elapsed_seconds': round(elapsed, 1),
            'config': config,
        })

        mem_str = f"{mem / 1e6:.1f} MB" if mem is not None else "N/A"
        benchmark_pbar.write(
            f">>> [{model_name} | miss={mr:.0%} | run {r+1}] "
            f"Test Acc={acc:.4f}  Memory={mem_str}  Time={elapsed:.0f}s"
        )

    # Print summary
    print_summary_table(all_results, model_names, missing_rates)

    # Save results to files
    if not args.dry_run:
        save_benchmark_results(all_results, run_logs, model_names, missing_rates, args)
    else:
        print("\n[INFO] Dry-run mode, results not saved to disk.")


if __name__ == '__main__':
    main()
