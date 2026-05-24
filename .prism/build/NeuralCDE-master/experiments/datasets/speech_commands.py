import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio
from tqdm import tqdm

from . import common

here = pathlib.Path(__file__).resolve().parent


def download():
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'SpeechCommands'
    loc = base_loc / 'speech_commands.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', loc)
    with tarfile.open(loc, 'r') as f:
        f.extractall(base_loc)


def _process_data(intensity_data):
    base_loc = here / 'data' / 'SpeechCommands'

    folders = ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go')
    all_files = []
    for foldername in folders:
        loc = base_loc / foldername
        for filename in os.listdir(loc):
            all_files.append((foldername, filename))

    y_map = {name: i for i, name in enumerate(folders)}

    # Load audio and compute MFCC in batches to avoid OOM
    mfcc_transform = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                                melkwargs=dict(n_fft=200, n_mels=64))
    mfcc_chunks = []
    y_list = []
    batch_size = 2000
    audio_batch = []

    for foldername, filename in tqdm(all_files, desc="Loading audio"):
        loc = base_loc / foldername
        audio, _ = torchaudio.load(loc / filename)
        audio = audio.squeeze(0)  # [channels, samples] -> [samples]
        if audio.dtype == torch.int16:
            audio = audio.float() / 2 ** 15
        elif audio.abs().max() > 1.0:
            audio = audio / 2 ** 15

        if len(audio) != 16000:
            continue

        audio_batch.append(audio)
        y_list.append(y_map[foldername])

        if len(audio_batch) >= batch_size:
            batch_tensor = torch.stack(audio_batch)
            mfcc_chunks.append(mfcc_transform(batch_tensor).transpose(1, 2).detach())
            audio_batch = []

    if audio_batch:
        batch_tensor = torch.stack(audio_batch)
        mfcc_chunks.append(mfcc_transform(batch_tensor).transpose(1, 2).detach())

    X = torch.cat(mfcc_chunks, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    batch_index = X.size(0)
    assert batch_index == 34975, "batch_index is {}".format(batch_index)
    # X is of shape (batch=34975, length=161, channels=20)

    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    final_index = torch.tensor(X.size(1) - 1).repeat(X.size(0))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X, y, final_index, append_times=True,
                                                   append_intensity=intensity_data)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data(intensity_data, batch_size):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / ('speech_commands_with_mels' + ('_intensity' if intensity_data else ''))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:
        download()
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_data(intensity_data)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader
