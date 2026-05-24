"""
Decompensation data diagnostic script.
Checks: timestamps (regular/irregular), feature count, class imbalance, label alignment.
Run: python diagnose_decompensation.py --data-dir ../mimic3-benchmarks/data/decompensation
"""
import csv
import os
import sys
import argparse
import pathlib
import collections
import torch
import numpy as np

here = pathlib.Path(__file__).resolve().parent

def check_raw_csv_format(data_dir, split='train', n_samples=5):
    """Check the actual format of raw episode CSV files."""
    data_dir = pathlib.Path(data_dir)
    listfile = data_dir / split / 'listfile.csv'
    ts_dir = data_dir / split

    print("\n" + "="*60)
    print("CHECK 1: Raw CSV file format (first {} samples)".format(n_samples))
    print("="*60)

    stay_labels = collections.OrderedDict()
    with open(listfile, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay = row['stay']
            if stay not in stay_labels:
                stay_labels[stay] = []

    stays = list(stay_labels.keys())[:n_samples]
    for stay in stays:
        csv_path = ts_dir / stay
        if not csv_path.exists():
            print("  MISSING: {}".format(stay))
            continue
        print("\n  File: {}".format(stay))
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            print("  Columns: {} (total={})".format(header[:5], len(header)))
            rows = list(reader)
            print("  Total rows (measurements): {}".format(len(rows)))
            if rows:
                hours = [float(r[0]) for r in rows[:10] if r[0]]
                print("  First 10 hour values: {}".format(hours))
                print("  Hours are integer? {}".format(all(h == int(h) for h in hours)))
                print("  Hour range: [{:.2f}, {:.2f}]".format(
                    float(rows[0][0]), float(rows[-1][0])))


def check_listfile_labels(data_dir, split='train'):
    """Check label distribution in the listfile."""
    data_dir = pathlib.Path(data_dir)
    listfile = data_dir / split / 'listfile.csv'

    print("\n" + "="*60)
    print("CHECK 2: Listfile label distribution ({} split)".format(split))
    print("="*60)

    period_lengths = []
    labels = []
    stays_seen = set()
    stay_labels = collections.OrderedDict()

    with open(listfile, 'r') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print("  Listfile columns: {}".format(cols))
        for row in reader:
            stay = row['stay']
            period = float(row['period_length'])
            label = int(row['y_true'])
            period_lengths.append(period)
            labels.append(label)
            stays_seen.add(stay)
            if stay not in stay_labels:
                stay_labels[stay] = []
            stay_labels[stay].append((period, label))

    labels = np.array(labels)
    period_lengths = np.array(period_lengths)

    print("\n  Total entries (patient-hour pairs): {:,}".format(len(labels)))
    print("  Unique stays (patients): {:,}".format(len(stays_seen)))
    print("  Positive labels: {:,} ({:.2f}%)".format(
        labels.sum(), 100 * labels.mean()))
    print("  Negative labels: {:,} ({:.2f}%)".format(
        (labels == 0).sum(), 100 * (labels == 0).mean()))
    print("  Implied pos_weight: {:.1f}".format(
        (labels == 0).sum() / max(labels.sum(), 1)))

    print("\n  Period length stats:")
    print("    min={:.1f}, max={:.1f}, mean={:.1f}".format(
        period_lengths.min(), period_lengths.max(), period_lengths.mean()))
    print("    Are all integers? {}".format(
        all(p == int(p) for p in period_lengths[:1000])))

    # Check entries per stay
    entries_per_stay = [len(v) for v in stay_labels.values()]
    print("\n  Entries per stay: min={}, max={}, mean={:.1f}".format(
        min(entries_per_stay), max(entries_per_stay),
        np.mean(entries_per_stay)))

    # For each stay, check if labels are all 0, all 1, or mixed
    all_zero = sum(1 for v in stay_labels.values() if all(l == 0 for _, l in v))
    all_one = sum(1 for v in stay_labels.values() if all(l == 1 for _, l in v))
    mixed = len(stay_labels) - all_zero - all_one
    print("\n  Stay-level label distribution:")
    print("    All-negative stays: {:,} ({:.1f}%)".format(
        all_zero, 100 * all_zero / len(stay_labels)))
    print("    All-positive stays: {:,} ({:.1f}%)".format(
        all_one, 100 * all_one / len(stay_labels)))
    print("    Mixed stays: {:,} ({:.1f}%)".format(
        mixed, 100 * mixed / len(stay_labels)))

    return stay_labels


def check_temporal_alignment(data_dir, split='train', n_samples=3):
    """
    Check the critical question: are raw CSV timestamps regular (hourly)?
    If not, our code has a temporal alignment bug.
    """
    data_dir = pathlib.Path(data_dir)
    listfile = data_dir / split / 'listfile.csv'
    ts_dir = data_dir / split

    print("\n" + "="*60)
    print("CHECK 3: Temporal alignment (timestamps vs expected hourly grid)")
    print("="*60)

    stay_labels = collections.OrderedDict()
    with open(listfile, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay = row['stay']
            if stay not in stay_labels:
                stay_labels[stay] = []
            stay_labels[stay].append((float(row['period_length']), int(row['y_true'])))

    # Pick a few stays with many entries
    stays_sorted = sorted(stay_labels.keys(),
                          key=lambda s: len(stay_labels[s]), reverse=True)

    for stay in stays_sorted[:n_samples]:
        csv_path = ts_dir / stay
        if not csv_path.exists():
            continue

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        if not rows:
            continue

        print("\n  Stay: {}".format(stay))
        print("  Listfile entries (period_length, label): {}".format(
            stay_labels[stay][:5]))

        hours = [float(r[0]) for r in rows]
        print("  CSV hour values (first 10): {}".format(hours[:10]))
        print("  Total CSV rows: {}".format(len(rows)))
        print("  Expected rows if hourly [1,168]: 168")

        # Check if hours are approximately integers
        non_integer = sum(1 for h in hours if abs(h - round(h)) > 0.01)
        print("  Non-integer hour values: {}/{} ({:.1f}%)".format(
            non_integer, len(hours), 100 * non_integer / max(len(hours), 1)))

        # Check gaps
        if len(hours) > 1:
            gaps = [hours[i+1] - hours[i] for i in range(min(len(hours)-1, 10))]
            print("  First 10 time gaps: {}".format(
                ["{:.2f}".format(g) for g in gaps]))
            print("  Are all gaps = 1.0 (hourly)? {}".format(
                all(abs(g - 1.0) < 0.01 for g in gaps)))


def check_cached_tensors():
    """Check the cached processed data."""
    cache_dir = here / 'datasets' / 'processed_data' / 'decompensation_h168_notimeintensity'

    print("\n" + "="*60)
    print("CHECK 4: Cached tensor statistics")
    print("="*60)

    if not cache_dir.exists():
        print("  Cache not found at {}".format(cache_dir))
        return

    # Load key tensors
    train_y = torch.load(str(cache_dir / 'train_y.pt'), weights_only=False)
    val_y = torch.load(str(cache_dir / 'val_y.pt'), weights_only=False)
    test_y = torch.load(str(cache_dir / 'test_y.pt'), weights_only=False)
    train_final_index = torch.load(str(cache_dir / 'train_final_index.pt'), weights_only=False)
    train_a = torch.load(str(cache_dir / 'train_a.pt'), weights_only=False)
    times = torch.load(str(cache_dir / 'times.pt'), weights_only=False)

    print("\n  Tensor shapes:")
    print("    train_y: {}".format(train_y.shape))
    print("    val_y:   {}".format(val_y.shape))
    print("    test_y:  {}".format(test_y.shape))
    print("    train_a (spline coeff): {}".format(train_a.shape))
    print("    train_final_index: {}".format(train_final_index.shape))
    print("    times: {} [{:.1f}, {:.1f}]".format(
        times.shape, times[0].item(), times[-1].item()))

    print("\n  Train set label analysis:")
    # Valid positions: y >= 0
    valid_mask = train_y >= 0
    n_valid = valid_mask.sum().item()
    n_total = train_y.numel()
    print("    Total time steps: {:,}".format(n_total))
    print("    Valid time steps (label != -1): {:,} ({:.1f}%)".format(
        n_valid, 100 * n_valid / n_total))

    valid_labels = train_y[valid_mask]
    n_pos = (valid_labels > 0.5).sum().item()
    n_neg = (valid_labels < 0.5).sum().item()
    print("    Positive labels: {:,} ({:.2f}%)".format(
        n_pos, 100 * n_pos / max(n_valid, 1)))
    print("    Negative labels: {:,} ({:.2f}%)".format(
        n_neg, 100 * n_neg / max(n_valid, 1)))
    print("    Implied pos_weight: {:.1f}".format(n_neg / max(n_pos, 1)))

    print("\n  Final index stats (sequence lengths):")
    fi = train_final_index.float()
    print("    min={:.0f}, max={:.0f}, mean={:.1f}, median={:.0f}".format(
        fi.min(), fi.max(), fi.mean(), fi.median()))

    # Check spline coeff for NaN/Inf
    nan_count = torch.isnan(train_a).sum().item()
    inf_count = torch.isinf(train_a).sum().item()
    print("\n  Spline coefficient (train_a) NaN: {:,}, Inf: {:,}".format(
        nan_count, inf_count))

    # Check label sequence structure for a few patients
    print("\n  Example label sequences (first 3 patients):")
    for i in range(min(3, train_y.shape[0])):
        y_seq = train_y[i]
        valid = y_seq[y_seq >= 0]
        n_pos_i = (valid > 0.5).sum().item()
        print("    Patient {}: {} valid labels, {} positive, final_idx={}".format(
            i, len(valid), n_pos_i, train_final_index[i].item()))
        # Show first 20 label values
        print("      Label sequence[:20]: {}".format(
            ["{:.0f}".format(v.item()) for v in y_seq[:20]]))


def check_feature_count_vs_benchmark(data_dir, split='train'):
    """Compare our feature count to the benchmark's expected 76 features."""
    data_dir = pathlib.Path(data_dir)
    listfile = data_dir / split / 'listfile.csv'
    ts_dir = data_dir / split

    print("\n" + "="*60)
    print("CHECK 5: Feature count vs benchmark expectation")
    print("="*60)

    # Read first available CSV
    with open(listfile, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay = row['stay']
            csv_path = ts_dir / stay
            if csv_path.exists():
                break

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

    print("  First CSV file: {}".format(stay))
    print("  Total columns: {} (expected: 1 time + N features)".format(len(header)))
    print("  Feature count: {}".format(len(header) - 1))
    print("  Column names:")
    for i, col in enumerate(header):
        print("    [{}] {}".format(i, col))

    print("\n  Expected feature counts:")
    print("    Raw 17-variable MIMIC-III: 17 features")
    print("    Benchmark discretized (76-feature): 76 features")
    print("  -> Our data has {} features".format(len(header) - 1))

    if len(header) - 1 == 17:
        print("\n  [!] Using RAW 17-variable data.")
        print("  [!] The benchmark standard uses 76 engineered features.")
        print("  [!] This EXPLAINS the poor AUROC compared to benchmark papers.")
    elif len(header) - 1 == 76:
        print("\n  [OK] Using standard benchmark 76-feature data.")
    else:
        print("\n  [?] Unexpected feature count: {}".format(len(header) - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True,
                        help='Path to mimic3-benchmarks decompensation data dir')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("MIMIC-III Decompensation Data Diagnostic")
    print("="*60)

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        print("ERROR: data_dir does not exist: {}".format(data_dir))
        sys.exit(1)

    print("Data directory: {}".format(data_dir))

    # Run all checks
    check_raw_csv_format(data_dir, split='train', n_samples=3)
    check_listfile_labels(data_dir, split='train')
    check_temporal_alignment(data_dir, split='train', n_samples=3)
    check_cached_tensors()
    check_feature_count_vs_benchmark(data_dir, split='train')

    print("\n" + "="*60)
    print("Diagnostic complete.")
    print("="*60)


if __name__ == '__main__':
    main()
