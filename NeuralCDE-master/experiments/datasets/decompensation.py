import csv
import gc
import math
import os
import pathlib
import collections
import numpy as np
import torch
import sklearn.model_selection

from . import common

here = pathlib.Path(__file__).resolve().parent

import sys
sys.path.append(str(here / '..' / '..'))
import controldiffeq

# Column indices (0-based) of GCS text fields in the raw CSV.
# Order: Diastolic BP(0), FiO2(1), GCS eye(2), GCS motor(3), GCS total(4),
#        GCS verbal(5), Glucose(6), HR(7), Height(8), Mean BP(9),
#        O2 sat(10), RR(11), Systolic BP(12), Temp(13), Weight(14), pH(15),
#        Capillary refill rate(16)
_GCS_TEXT_COLS = frozenset([2, 3, 5])


def _compute_hourly_stats(window_values, num_raw_features):
    """Compute 76 features for a single hour window.

    For each numerical variable (14 cols): mean, min, max, std = 4 values.
    For each GCS text variable (3 cols): mean = 1 value.
    Per-variable observation mask for all 17 cols = 17 values.
    Total: 14*4 + 3 + 17 = 76 features.

    Arguments:
        window_values: list of value-lists (one per measurement in this window).
        num_raw_features: number of raw columns (17).

    Returns:
        list of 76 floats (NaN where no measurements in window).
    """
    col_values = [[] for _ in range(num_raw_features)]
    for vals in window_values:
        for ci, v in enumerate(vals):
            if v == v:  # not NaN (NaN != NaN)
                col_values[ci].append(v)

    stats = []
    for ci in range(num_raw_features):
        vs = col_values[ci]
        if ci in _GCS_TEXT_COLS:
            stats.append(sum(vs) / len(vs) if vs else float('nan'))
        else:
            if vs:
                n = len(vs)
                mean_v = sum(vs) / n
                min_v = min(vs)
                max_v = max(vs)
                if n > 1:
                    variance = sum((v - mean_v) ** 2 for v in vs) / (n - 1)
                    std_v = variance ** 0.5
                else:
                    std_v = 0.0
                stats.extend([mean_v, min_v, max_v, std_v])
            else:
                stats.extend([float('nan'), float('nan'), float('nan'), float('nan')])

    for ci in range(num_raw_features):
        stats.append(1.0 if col_values[ci] else 0.0)

    return stats


def _read_listfile(listfile_path):
    """Read a listfile.csv and group labels by stay.

    Returns:
        dict: {stay_filename: [(hour, label), ...]}
    """
    stay_labels = collections.OrderedDict()
    with open(listfile_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay = row['stay']
            hour = int(float(row['period_length']))
            label = int(row['y_true'])
            if stay not in stay_labels:
                stay_labels[stay] = []
            stay_labels[stay].append((hour, label))
    return stay_labels


def _parse_value(v):
    if v == '' or v == 'nan':
        return float('nan')
    try:
        return float(v)
    except ValueError:
        try:
            return float(v.split(' ')[0])
        except (ValueError, IndexError):
            return float('nan')


def _read_timeseries(csv_path, max_hours):
    """Read a patient timeseries CSV and compute 76 engineered features per hour.

    For each target hour t in [1..max_hours], collects all measurements in the
    window (t-1, t] and computes:
      - 14 numerical vars x (mean, min, max, std) = 56 features
      - 3 GCS text vars x mean = 3 features
      - 17 per-variable observation masks = 17 features
    Total: 76 features per hour.

    Hours with no measurements produce NaN stats and mask=0.

    Returns:
        features: list of max_hours vectors, each of length 76.
        num_features: 76 (0 if CSV is empty/unreadable).
        last_valid_hour: last hour that had at least one measurement.
    """
    raw = []
    num_raw_features = None

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        num_raw_features = len(header) - 1
        for row in reader:
            if not row or not row[0]:
                continue
            hour_val = float(row[0])
            if hour_val > max_hours:
                break
            values = [_parse_value(v) for v in row[1:]]
            raw.append((hour_val, values))

    if num_raw_features is None:
        return [], 0, 0

    n_gcs_text = len(_GCS_TEXT_COLS)
    n_numerical = num_raw_features - n_gcs_text
    num_features = n_numerical * 4 + n_gcs_text + num_raw_features  # 56+3+17=76

    features = []
    last_valid_hour = 0
    ptr = 0
    n_meas = len(raw)

    for t in range(1, max_hours + 1):
        # Collect measurements in the window (t-1, t].
        # ptr already points past all measurements with hour_val <= t-1
        # from the previous iteration, so no need to re-advance.
        window = []
        tmp = ptr
        while tmp < n_meas and raw[tmp][0] <= t:
            window.append(raw[tmp][1])
            tmp += 1
        # Advance ptr past this window for the next iteration.
        ptr = tmp

        feat_vec = _compute_hourly_stats(window, num_raw_features)
        features.append(feat_vec)
        if window:
            last_valid_hour = t

    return features, num_features, last_valid_hour


def _process_data(data_dir, split, max_hours, time_intensity):
    """Process one split (train or test) of decompensation data.

    Arguments:
        data_dir: Path to decompensation data directory.
        split: 'train' or 'test'.
        max_hours: Maximum number of hours to include.
        time_intensity: Whether to include intensity (observation mask cumsum).

    Returns:
        X_times: [N, max_hours, num_features] tensor with NaN for missing.
        y_seq: [N, max_hours] tensor with labels, -1 for invalid/padding.
        final_indices: [N] tensor with last valid time index per patient.
        num_features: Number of raw features per timestep.
    """
    data_dir = pathlib.Path(data_dir)
    listfile_path = data_dir / split / 'listfile.csv'
    ts_dir = data_dir / split

    print("Reading listfile: {}".format(listfile_path))
    stay_labels = _read_listfile(listfile_path)
    print("Found {} stays in {} split".format(len(stay_labels), split))

    # Pre-allocate numpy arrays to avoid Python list-of-lists memory explosion.
    # Python floats cost ~28 bytes each; numpy float32 costs 4 bytes.
    # 35K patients x 168 steps x 76 features: Python lists ~14GB, numpy ~1.7GB.
    max_n = len(stay_labels)
    X_arr = None   # allocated after first successful read (to learn num_features)
    y_arr = np.full((max_n, max_hours), -1.0, dtype=np.float32)
    fi_arr = np.zeros(max_n, dtype=np.int64)
    count = 0
    num_features = None
    skipped = 0

    for i, (stay_file, labels) in enumerate(stay_labels.items()):
        if (i + 1) % 5000 == 0 or i == 0:
            print("  Processing stay {}/{}...".format(i + 1, len(stay_labels)))

        csv_path = ts_dir / stay_file
        if not csv_path.exists():
            skipped += 1
            continue

        features, nf, last_valid_hour = _read_timeseries(str(csv_path), max_hours)

        if num_features is None and nf > 0:
            num_features = nf
            X_arr = np.full((max_n, max_hours, nf), np.nan, dtype=np.float32)

        if last_valid_hour == 0:
            skipped += 1
            continue

        # Build per-hour label sequence (period_length values are integers)
        y_row = np.full(max_hours, -1.0, dtype=np.float32)
        last_labeled_hour = 0
        for hour, label in labels:
            if 1 <= hour <= max_hours:
                y_row[hour - 1] = float(label)
                if hour > last_labeled_hour:
                    last_labeled_hour = hour

        if last_labeled_hour == 0:
            skipped += 1
            continue

        # final_index: last time point where we have a label (0-indexed).
        final_idx = min(last_labeled_hour, max_hours) - 1

        if X_arr is not None:
            X_arr[count] = features
        y_arr[count] = y_row
        fi_arr[count] = final_idx
        count += 1

    if skipped > 0:
        print("  Skipped {} stays (missing file or too short)".format(skipped))

    print("Converting to tensors ({} samples)...".format(count))
    if X_arr is None or count == 0:
        nf = num_features or 76
        X_times = torch.zeros(0, max_hours, nf, dtype=torch.float32)
    else:
        X_times = torch.from_numpy(X_arr[:count].copy())
    y_seq = torch.from_numpy(y_arr[:count].copy())
    final_indices = torch.from_numpy(fi_arr[:count].copy())

    return X_times, y_seq, final_indices, num_features


def _normalise_with_stats(X, train_mean, train_std):
    """Normalise X using precomputed mean and std from training set."""
    out = []
    for i, (Xi, m, s) in enumerate(zip(X.unbind(dim=-1), train_mean, train_std)):
        out.append((Xi - m) / (s + 1e-5))
    return torch.stack(out, dim=-1)


def _compute_train_stats(X_train):
    """Compute per-feature mean and std from training data, ignoring NaN."""
    means = []
    stds = []
    for Xi in X_train.unbind(dim=-1):
        valid = Xi.masked_select(~torch.isnan(Xi))
        if valid.numel() > 0:
            means.append(valid.mean())
            stds.append(valid.std())
        else:
            means.append(torch.tensor(0.0))
            stds.append(torch.tensor(1.0))
    return means, stds


def _augment_data(X, times_1d, time_intensity):
    """Augment data with time channel and optionally intensity channel."""
    batch_size = X.size(0)
    seq_len = X.size(1)

    augmented = []
    # Prepend time channel
    time_channel = times_1d.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
    augmented.append(time_channel)

    # Optionally prepend intensity (observation mask cumsum)
    if time_intensity:
        intensity = (~torch.isnan(X)).to(X.dtype).cumsum(dim=1)
        augmented.append(intensity)

    augmented.append(X)
    return torch.cat(augmented, dim=2)


def _compute_spline_coeffs_batched(times_1d, X, batch_size=1000):
    """Compute natural cubic spline coefficients in batches to save memory.

    Uses pre-allocation to avoid 2x peak memory from torch.cat.
    batch_size=1000 keeps transient overhead under ~200 MB per batch.
    """
    N = X.size(0)

    # Compute first small batch to learn coefficient shapes.
    init_end = min(batch_size, N)
    print("  Computing spline coefficients (batch 0-{}/{})...".format(init_end, N))
    first_coeffs = controldiffeq.natural_cubic_spline_coeffs(times_1d, X[:init_end])
    coeff_shapes = [tuple(c.shape[1:]) for c in first_coeffs]

    # Pre-allocate output tensors for all N samples at once.
    # This avoids the 2x memory spike from repeated torch.cat.
    all_coeffs = [torch.empty(N, *shape, dtype=torch.float32)
                  for shape in coeff_shapes]
    for k, c in enumerate(first_coeffs):
        all_coeffs[k][:init_end] = c
    del first_coeffs
    gc.collect()

    for start in range(init_end, N, batch_size):
        end = min(start + batch_size, N)
        print("  Spline batch {}-{}/{}...".format(start, end, N))

        batch_X = X[start:end]
        batch_coeffs = controldiffeq.natural_cubic_spline_coeffs(times_1d, batch_X)
        for k, c in enumerate(batch_coeffs):
            all_coeffs[k][start:end] = c
        del batch_X, batch_coeffs
        gc.collect()

    return tuple(all_coeffs)


def get_data(data_dir, time_intensity, batch_size, max_hours=168):
    """Load and preprocess MIMIC-III decompensation data.

    Arguments:
        data_dir: Path to mimic3-benchmarks decompensation output directory.
            Expected structure: data_dir/train_listfile.csv, data_dir/train/, etc.
        time_intensity: Whether to include observation intensity features.
        batch_size: Batch size for DataLoaders.
        max_hours: Maximum hours to include per patient (default 168 = 7 days).

    Returns:
        times, train_dataloader, val_dataloader, test_dataloader
    """
    data_dir = pathlib.Path(data_dir)
    # 默认缓存路径在当前代码所在目录下的 processed_data 文件夹
    # 当前位于 D 盘: d:\实验室\mamba-cde\NeuralCDE-master\experiments\datasets\processed_data
    # 如果您希望转移到其他位置（例如 D 盘根目录），可以修改并将下面这行取消注释：
    # cache_base = pathlib.Path('D:/AI_Cache/decompensation')
    cache_base = here / 'processed_data'
    # _76feat suffix marks engineered 76-feature vectors (14 numerical*4 + 3 GCS + 17 masks).
    # Old caches with _locf used raw 17-feature LOCF; they are incompatible with this version.
    cache_name = 'decompensation_h{}_76feat{}'.format(
        max_hours,
        '_timeintensity' if time_intensity else '_notimeintensity')
    cache_loc = cache_base / cache_name

    if os.path.exists(cache_loc):
        print("Loading cached decompensation data from {}...".format(cache_loc))
        tensors = common.load_data(cache_loc)
        times = tensors['times']
        train_coeffs = (tensors['train_a'], tensors['train_b'],
                       tensors['train_c'], tensors['train_d'])
        val_coeffs = (tensors['val_a'], tensors['val_b'],
                     tensors['val_c'], tensors['val_d'])
        test_coeffs = (tensors['test_a'], tensors['test_b'],
                      tensors['test_c'], tensors['test_d'])
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
        print("Cached data loaded successfully!")
    else:
        print("No cached data found, processing from raw files...")

        # Process train and test splits
        print("\n=== Processing TRAIN split ===")
        train_X, train_y_full, train_final_index, num_features = _process_data(
            data_dir, 'train', max_hours, time_intensity)

        print("\n=== Processing TEST split ===")
        test_X, test_y, test_final_index, _ = _process_data(
            data_dir, 'test', max_hours, time_intensity)

        # Split train into train/val (85/15, stratified by patient-level label)
        # Patient-level label: 1 if any hour has label=1, else 0
        print("\nSplitting train into train/val (85/15)...")
        patient_labels = (train_y_full.max(dim=1).values > 0).long()
        indices = list(range(len(train_X)))
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            indices, test_size=0.15, random_state=42,
            stratify=patient_labels.numpy())

        val_X = train_X[val_idx]
        val_y = train_y_full[val_idx]
        val_final_index = train_final_index[val_idx]
        train_X = train_X[train_idx]
        train_y = train_y_full[train_idx]
        train_final_index = train_final_index[train_idx]

        print("Train: {}, Val: {}, Test: {}".format(
            len(train_X), len(val_X), len(test_X)))

        # Compute normalization stats from training set only.
        print("\nNormalizing data (computing train stats)...")
        train_mean, train_std = _compute_train_stats(train_X)

        # Create time axis
        times = torch.linspace(1, max_hours, max_hours)

        # Process each split sequentially to minimise peak memory.
        # Pattern: normalize -> augment -> fillnan -> compute coeffs -> del X.
        # This avoids holding all three preprocessed X tensors in RAM
        # simultaneously alongside the growing coefficient tensors.
        print("\nProcessing train set (normalize / augment / spline)...")
        train_X = _normalise_with_stats(train_X, train_mean, train_std)
        train_X = _augment_data(train_X, times, time_intensity)
        train_X = torch.where(torch.isnan(train_X),
                              torch.zeros_like(train_X), train_X)
        print("  Train set:")
        train_coeffs = _compute_spline_coeffs_batched(times, train_X)
        del train_X
        gc.collect()

        print("Processing val set (normalize / augment / spline)...")
        val_X = _normalise_with_stats(val_X, train_mean, train_std)
        val_X = _augment_data(val_X, times, time_intensity)
        val_X = torch.where(torch.isnan(val_X),
                            torch.zeros_like(val_X), val_X)
        print("  Val set:")
        val_coeffs = _compute_spline_coeffs_batched(times, val_X)
        del val_X
        gc.collect()

        print("Processing test set (normalize / augment / spline)...")
        test_X = _normalise_with_stats(test_X, train_mean, train_std)
        test_X = _augment_data(test_X, times, time_intensity)
        test_X = torch.where(torch.isnan(test_X),
                             torch.zeros_like(test_X), test_X)
        print("  Test set:")
        test_coeffs = _compute_spline_coeffs_batched(times, test_X)
        del test_X
        gc.collect()

        # Cache to disk
        print("\nSaving processed data to cache...")
        if not os.path.exists(cache_base):
            os.mkdir(cache_base)
        if not os.path.exists(cache_loc):
            os.mkdir(cache_loc)

        common.save_data(cache_loc,
                        times=times,
                        train_a=train_coeffs[0], train_b=train_coeffs[1],
                        train_c=train_coeffs[2], train_d=train_coeffs[3],
                        val_a=val_coeffs[0], val_b=val_coeffs[1],
                        val_c=val_coeffs[2], val_d=val_coeffs[3],
                        test_a=test_coeffs[0], test_b=test_coeffs[1],
                        test_c=test_coeffs[2], test_d=test_coeffs[3],
                        train_y=train_y, val_y=val_y, test_y=test_y,
                        train_final_index=train_final_index,
                        val_final_index=val_final_index,
                        test_final_index=test_final_index)
        print("Data cached successfully!")

    # Wrap into DataLoaders
    print("Creating data loaders...")
    # For decompensation, y_seq replaces single y label
    # final_index is still needed for variable-length sequences
    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(
        times, train_coeffs, val_coeffs, test_coeffs,
        train_y, val_y, test_y,
        train_final_index, val_final_index, test_final_index,
        'cpu', batch_size=batch_size)

    print("Data loading complete!")
    return times, train_dataloader, val_dataloader, test_dataloader
