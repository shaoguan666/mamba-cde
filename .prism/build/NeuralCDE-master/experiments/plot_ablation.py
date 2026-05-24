"""
Ablation Study Plotting Script
Usage:
    python plot_ablation.py                    # Sepsis (default)
    python plot_ablation.py --dataset speech   # Speech Commands

Reads results from results/ directory and generates:
1. Ablation bar chart (AUROC/Accuracy comparison)
2. Training curve comparison
3. Multi-metric comparison (Sepsis: AUROC+AP, Speech: Accuracy+Params)
4. Parameter efficiency scatter plot
"""
import argparse
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paper-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'paper_figures')


def get_results_dir(dataset):
    """Get results directory for a dataset."""
    if dataset == 'sepsis':
        return os.path.join(SCRIPT_DIR, 'results', 'sepsis_intensity')
    elif dataset == 'speech':
        return os.path.join(SCRIPT_DIR, 'results', 'speech_commands')
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def load_result(results_dir, file_id):
    """Load a single result JSON file."""
    path = os.path.join(results_dir, str(file_id))
    with open(path, 'r') as f:
        return json.load(f)


def scan_results(results_dir):
    """Scan results directory and return available results with model names."""
    results = {}
    if not os.path.exists(results_dir):
        return results

    for fname in sorted(os.listdir(results_dir)):
        fpath = os.path.join(results_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, 'r') as f:
                r = json.load(f)
            model_name = extract_model_name(r)
            results[fname] = {'data': r, 'model': model_name}
        except (json.JSONDecodeError, KeyError):
            pass
    return results


def extract_model_name(result):
    """Extract short model name from result."""
    vf = result.get('vector_field', '')
    if 'DeepFiLMVectorField' in vf:
        return 'DeepFiLM'
    elif 'DeepMLPDropoutVectorField' in vf:
        return 'DeepMLP+Dropout'
    elif 'DeepMLPVectorField' in vf:
        return 'DeepMLP'
    elif 'FinalTanh' in vf:
        return 'Baseline NCDE'
    return 'Unknown'


def get_training_curves(result, metric='auroc'):
    """Extract epoch-wise metric from history."""
    history = result.get('history', [])
    epochs = [h['epoch'] for h in history]
    train_vals = [h['train_metrics'].get(metric, h['train_metrics'].get('accuracy', 0))
                  for h in history]
    val_vals = [h['val_metrics'].get(metric, h['val_metrics'].get('accuracy', 0))
                for h in history]
    return epochs, train_vals, val_vals


# ======================================================================
# Figure 1: Ablation Bar Chart
# ======================================================================
def plot_ablation_bar(results_dict, save_dir=OUTPUT_DIR, metric_name='AUROC',
                      dataset_prefix=''):
    """
    Ablation study bar chart.

    Args:
        results_dict: OrderedDict of {label: value}
        metric_name: y-axis label
        dataset_prefix: prefix for filename
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = '{}_'.format(dataset_prefix) if dataset_prefix else ''
    save_path = os.path.join(save_dir, '{}fig_ablation_bar.pdf'.format(prefix))

    labels = list(results_dict.keys())
    values = list(results_dict.values())
    n = len(labels)

    # Color: last bar (Ours) highlighted
    colors = ['#B0BEC5', '#90A4AE', '#78909C'] + ['#1565C0']
    colors = colors[-n:]
    edge_colors = ['#78909C'] * (n - 1) + ['#0D47A1']

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    bars = ax.bar(range(n), values, color=colors, edgecolor=edge_colors,
                  linewidth=1.2, width=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.0008,
                '{:.4f}'.format(val), ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    baseline = values[0]
    ax.axhline(y=baseline, color='#C62828', linestyle='--', linewidth=1.2,
               alpha=0.7, label='Baseline ({:.4f})'.format(baseline))

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha='right')
    ax.set_ylabel('Test {}'.format(metric_name), fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component Contribution',
                 fontsize=13, fontweight='bold', pad=12)

    y_min = min(values) - 0.012
    y_max = max(values) + 0.015
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Ablation bar chart saved: {}".format(save_path))


# ======================================================================
# Figure 2: Training Curves
# ======================================================================
def plot_training_curves(curves_dict, save_dir=OUTPUT_DIR, metric_name='AUROC',
                         dataset_prefix=''):
    """
    Training curve comparison.

    Args:
        curves_dict: {label: (epochs, train_vals, val_vals)}
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = '{}_'.format(dataset_prefix) if dataset_prefix else ''
    save_path = os.path.join(save_dir, '{}fig_training_curves.pdf'.format(prefix))

    colors = {
        'Baseline NCDE': '#78909C',
        'DeepMLP': '#FF8F00',
        'DeepMLP+Dropout': '#43A047',
        'DeepFiLM (Ours)': '#1565C0',
    }
    markers = {
        'Baseline NCDE': 'o',
        'DeepMLP': 's',
        'DeepMLP+Dropout': '^',
        'DeepFiLM (Ours)': 'D',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    for label, (epochs, train_vals, val_vals) in curves_dict.items():
        c = colors.get(label, '#555555')
        m = markers.get(label, 'o')
        markevery = max(1, len(epochs) // 8)

        ax1.plot(epochs, train_vals, color=c, linewidth=2, label=label,
                 marker=m, markersize=5, markevery=markevery, alpha=0.9)
        ax2.plot(epochs, val_vals, color=c, linewidth=2, label=label,
                 marker=m, markersize=5, markevery=markevery, alpha=0.9)

    for ax, title in [(ax1, 'Train {}'.format(metric_name)),
                      (ax2, 'Validation {}'.format(metric_name))]:
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Training curves saved: {}".format(save_path))


# ======================================================================
# Figure 3: Multi-metric Comparison
# ======================================================================
def plot_multi_metric(metrics_dict, save_dir=OUTPUT_DIR, dataset_prefix='',
                      metric1_name='AUROC', metric2_name='Average Precision'):
    """
    Multi-metric grouped bar chart.

    Args:
        metrics_dict: {label: {'metric1': float, 'metric2': float, 'params': int}}
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = '{}_'.format(dataset_prefix) if dataset_prefix else ''
    save_path = os.path.join(save_dir, '{}fig_multi_metric.pdf'.format(prefix))

    labels = list(metrics_dict.keys())
    vals1 = [metrics_dict[k].get('metric1', metrics_dict[k].get('auroc', 0))
             for k in labels]
    vals2 = [metrics_dict[k].get('metric2', metrics_dict[k].get('ap', 0))
             for k in labels]
    n = len(labels)

    x = np.arange(n)
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    bars1 = ax.bar(x - width / 2, vals1, width, label=metric1_name,
                   color='#1565C0', alpha=0.85, edgecolor='#0D47A1',
                   linewidth=1.0, zorder=3)
    bars2 = ax.bar(x + width / 2, vals2, width, label=metric2_name,
                   color='#FF8F00', alpha=0.85, edgecolor='#E65100',
                   linewidth=1.0, zorder=3)

    for bar, val in zip(bars1, vals1):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.005,
                '{:.3f}'.format(val), ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.005,
                '{:.3f}'.format(val), ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: {} & {}'.format(metric1_name, metric2_name),
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Multi-metric chart saved: {}".format(save_path))


# ======================================================================
# Figure 4: Parameter Efficiency
# ======================================================================
def plot_param_efficiency(data_dict, save_dir=OUTPUT_DIR, metric_name='AUROC',
                          dataset_prefix=''):
    """
    Scatter plot: metric vs parameter count.
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = '{}_'.format(dataset_prefix) if dataset_prefix else ''
    save_path = os.path.join(save_dir, '{}fig_param_efficiency.pdf'.format(prefix))

    colors_map = {
        'Baseline NCDE': '#78909C',
        'DeepMLP': '#FF8F00',
        'DeepMLP+Dropout': '#43A047',
        'DeepFiLM (Ours)': '#1565C0',
    }
    marker_map = {
        'Baseline NCDE': 'o',
        'DeepMLP': 's',
        'DeepMLP+Dropout': '^',
        'DeepFiLM (Ours)': '*',
    }

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for label, d in data_dict.items():
        c = colors_map.get(label, '#555')
        m = marker_map.get(label, 'o')
        size = 200 if 'Ours' in label else 120
        metric_val = d.get('metric1', d.get('auroc', d.get('accuracy', 0)))
        ax.scatter(d['params'] / 1000, metric_val, c=c, marker=m,
                   s=size, label=label, zorder=5, edgecolors='white',
                   linewidth=1.5)

    ax.set_xlabel('Parameters (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test {}'.format(metric_name), fontsize=12, fontweight='bold')
    ax.set_title('Parameter Efficiency', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Param efficiency chart saved: {}".format(save_path))


# ======================================================================
# Main
# ======================================================================
def run_sepsis():
    """Run sepsis ablation plots with hardcoded file IDs."""
    results_dir = get_results_dir('sepsis')

    FILE_IDS = {
        'Baseline NCDE': 16,
        'DeepMLP': 15,
        'DeepMLP+Dropout': 18,
        'DeepFiLM (Ours)': 19,
    }

    print("=" * 60)
    print("Loading Sepsis ablation results")
    print("=" * 60)

    results = {}
    for label, fid in FILE_IDS.items():
        r = load_result(results_dir, fid)
        results[label] = r
        auroc = r['test_metrics']['auroc']
        ap = r['test_metrics']['average_precision']
        params = r['parameters']
        print("  {}: AUROC={:.4f}, AP={:.4f}, Params={}".format(
            label, auroc, ap, params))

    # Figure 1: Ablation Bar
    print("\n--- Figure 1: Ablation Bar Chart ---")
    ablation_auroc = {label: r['test_metrics']['auroc']
                      for label, r in results.items()}
    plot_ablation_bar(ablation_auroc, metric_name='AUROC', dataset_prefix='sepsis')

    # Figure 2: Training Curves
    print("\n--- Figure 2: Training Curves ---")
    curves = {}
    for label, r in results.items():
        epochs, train_auroc, val_auroc = get_training_curves(r, metric='auroc')
        curves[label] = (epochs, train_auroc, val_auroc)
    plot_training_curves(curves, metric_name='AUROC', dataset_prefix='sepsis')

    # Figure 3: Multi-metric
    print("\n--- Figure 3: Multi-metric (AUROC + AP) ---")
    multi = {label: {
        'metric1': r['test_metrics']['auroc'],
        'metric2': r['test_metrics']['average_precision'],
        'params': r['parameters'],
    } for label, r in results.items()}
    plot_multi_metric(multi, metric1_name='AUROC', metric2_name='Average Precision',
                      dataset_prefix='sepsis')

    # Figure 4: Param Efficiency
    print("\n--- Figure 4: Parameter Efficiency ---")
    plot_param_efficiency(multi, metric_name='AUROC', dataset_prefix='sepsis')


def run_speech():
    """Run Speech Commands ablation plots by scanning results directory."""
    results_dir = get_results_dir('speech')

    if not os.path.exists(results_dir):
        print("[ERROR] Speech Commands results directory not found: {}".format(results_dir))
        print("        Run experiments first:")
        print("        python run_experiment.py --dataset speech --ablation")
        return

    print("=" * 60)
    print("Loading Speech Commands ablation results")
    print("=" * 60)

    # Scan and find the latest result for each model type
    all_results = scan_results(results_dir)
    if not all_results:
        print("[ERROR] No results found in {}".format(results_dir))
        return

    # Group by model name, keep latest
    model_results = {}
    label_order = ['Baseline NCDE', 'DeepMLP', 'DeepMLP+Dropout', 'DeepFiLM']
    label_map = {
        'Baseline NCDE': 'Baseline NCDE',
        'DeepMLP': 'DeepMLP',
        'DeepMLP+Dropout': 'DeepMLP+Dropout',
        'DeepFiLM': 'DeepFiLM (Ours)',
    }

    for fname, info in all_results.items():
        mname = info['model']
        if mname in label_map:
            display_label = label_map[mname]
            model_results[display_label] = info['data']

    if not model_results:
        print("[ERROR] No recognized model results found.")
        print("        Available models: {}".format(
            [v['model'] for v in all_results.values()]))
        return

    for label, r in model_results.items():
        acc = r['test_metrics']['accuracy']
        params = r['parameters']
        print("  {}: Accuracy={:.4f}, Params={}".format(label, acc, params))

    # Figure 1: Ablation Bar (Accuracy)
    print("\n--- Figure 1: Ablation Bar Chart (Accuracy) ---")
    ablation_acc = {label: r['test_metrics']['accuracy']
                    for label, r in model_results.items()}
    plot_ablation_bar(ablation_acc, metric_name='Accuracy', dataset_prefix='speech')

    # Figure 2: Training Curves (Accuracy)
    print("\n--- Figure 2: Training Curves ---")
    curves = {}
    for label, r in model_results.items():
        epochs, train_acc, val_acc = get_training_curves(r, metric='accuracy')
        curves[label] = (epochs, train_acc, val_acc)
    plot_training_curves(curves, metric_name='Accuracy', dataset_prefix='speech')

    # Figure 3: Parameter Efficiency
    print("\n--- Figure 3: Parameter Efficiency ---")
    param_data = {label: {
        'accuracy': r['test_metrics']['accuracy'],
        'params': r['parameters'],
    } for label, r in model_results.items()}
    plot_param_efficiency(param_data, metric_name='Accuracy', dataset_prefix='speech')


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Study Plotting')
    parser.add_argument('--dataset', type=str, default='sepsis',
                        choices=['sepsis', 'speech'],
                        help='Dataset: sepsis or speech (default: sepsis)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.dataset == 'sepsis':
        run_sepsis()
    elif args.dataset == 'speech':
        run_speech()

    print("\n" + "=" * 60)
    print("All figures saved to: {}".format(os.path.abspath(OUTPUT_DIR)))
    print("=" * 60)


if __name__ == '__main__':
    main()
