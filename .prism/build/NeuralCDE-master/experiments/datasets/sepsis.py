import csv
import math
import os
import pathlib
import torch
import urllib.request
import zipfile

from . import common


here = pathlib.Path(__file__).resolve().parent

base_base_loc = here / 'data'
base_loc = base_base_loc / 'sepsis'
loc_Azip = base_loc / 'training_setA.zip'
loc_Bzip = base_loc / 'training_setB.zip'


def _get_psv_files():
    """Get all .psv file paths, checking both base_loc and subdirectories."""
    psv_files = []
    if os.path.exists(base_loc):
        # Check directly in base_loc
        for f in os.listdir(base_loc):
            if f.endswith('.psv'):
                psv_files.append(base_loc / f)
        # Check in subdirectories (training_A, training_B, training, training_setB)
        for subdir in ('training_A', 'training_B', 'training', 'training_setB'):
            subdir_path = base_loc / subdir
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.endswith('.psv'):
                        psv_files.append(subdir_path / f)
    return psv_files


def download():
    # Check if .psv files already exist (data already downloaded and extracted)
    psv_files = _get_psv_files()
    if len(psv_files) > 0:
        print(f"Data already exists: found {len(psv_files)} .psv files")
        return

    if not os.path.exists(loc_Azip):
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                                   str(loc_Azip))
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
                                   str(loc_Bzip))

        with zipfile.ZipFile(loc_Azip, 'r') as f:
            f.extractall(str(base_loc))
        with zipfile.ZipFile(loc_Bzip, 'r') as f:
            f.extractall(str(base_loc))
        for folder in ('training', 'training_setB'):
            for filename in os.listdir(base_loc / folder):
                if os.path.exists(base_loc / filename):
                    raise RuntimeError
                os.rename(base_loc / folder / filename, base_loc / filename)


def _process_data(static_intensity, time_intensity):
    X_times = []
    X_static = []
    y = []
    psv_files = _get_psv_files()
    total_files = len(psv_files)
    print(f"Processing {total_files} .psv files...")
    for i, filepath in enumerate(psv_files):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"Processing file {i + 1}/{total_files}...")
        with open(filepath) as file:
            time = []
            label = 0.0
            reader = csv.reader(file, delimiter='|')
            reader = iter(reader)
            next(reader)  # first line is headings
            prev_iculos = 0
            for line in reader:
                assert len(line) == 41
                *time_values, age, gender, unit1, unit2, hospadmtime, iculos, sepsislabel = line
                iculos = int(iculos)
                if iculos > 72:  # keep at most the first three days
                    break
                for iculos_ in range(prev_iculos + 1, iculos):
                    time.append([float('nan') for value in time_values])
                prev_iculos = iculos
                time.append([float(value) for value in time_values])
                label = max(label, float(sepsislabel))
            unit1 = float(unit1)
            unit2 = float(unit2)
            unit1_obs = not math.isnan(unit1)
            unit2_obs = not math.isnan(unit2)
            if not unit1_obs:
                unit1 = 0.
            if not unit2_obs:
                unit2 = 0.
            hospadmtime = float(hospadmtime)
            if math.isnan(hospadmtime):
                hospadmtime = 0.  # this only happens for one record
            static = [float(age), float(gender), unit1, unit2, hospadmtime]
            if static_intensity:
                static += [unit1_obs, unit2_obs]
            if len(time) > 2:
                X_times.append(time)
                X_static.append(static)
                y.append(label)
    print(f"Loaded {len(X_times)} samples, computing indices...")
    final_indices = []
    for time in X_times:
        final_indices.append(len(time) - 1)
    maxlen = max(final_indices) + 1
    print(f"Padding sequences to length {maxlen}...")
    for time in X_times:
        for _ in range(maxlen - len(time)):
            time.append([float('nan') for value in time_values])

    print("Converting to tensors...")
    X_times = torch.tensor(X_times)
    X_static = torch.tensor(X_static)
    y = torch.tensor(y)
    final_indices = torch.tensor(final_indices)

    times = torch.linspace(1, X_times.size(1), X_times.size(1))

    print("Preprocessing data (computing spline coefficients, this may take a while)...")
    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X_times, y, final_indices, append_times=True,
                                                   append_intensity=time_intensity)
    print("Processing static features...")
    if static_intensity:
        X_static_ = X_static[:, :-2]
        X_static_ = common.normalise_data(X_static_, y)
        X_static = torch.cat([X_static_, X_static[:, -2:]], dim=1)
    else:
        X_static = common.normalise_data(X_static, y)
    print("Splitting static features...")
    train_X_static, val_X_static, test_X_static = common.split_data(X_static, y)
    train_coeffs = (*train_coeffs, train_X_static)
    val_coeffs = (*val_coeffs, val_X_static)
    test_coeffs = (*test_coeffs, test_X_static)
    print("Data processing complete!")

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data(static_intensity, time_intensity, batch_size):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / ('sepsis' + ('_staticintensity' if static_intensity else '_nostaticintensity') + ('_timeintensity' if time_intensity else '_notimeintensity'))
    if os.path.exists(loc):
        print("Loading cached processed data...")
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d'], tensors['train_static']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d'], tensors['val_static']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d'], tensors['test_static']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
        print("Cached data loaded successfully!")
    else:
        print("No cached data found, processing from raw files...")
        download()
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_data(static_intensity, time_intensity)
        print("Saving processed data to cache...")
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3], train_static=train_coeffs[4],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         val_static=val_coeffs[4],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         test_static=test_coeffs[4],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)
        print("Data cached successfully!")

    print("Creating data loaders...")
    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)
    print("Data loading complete!")
    return times, train_dataloader, val_dataloader, test_dataloader
