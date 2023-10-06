import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import librosa
import yaml


with open('make_dataset/dataset_settings.yaml') as file:
    config = yaml.safe_load(file)

# for dataset
length = config['length']
batchsize = config['batchsize']

# for trim
top_db = config['top_db']

# for melspectrogram
n_mels = config['n_mels']
n_fft = config['n_fft']
hop_length = config['hop_length']
window_fn = config['window_fn']


def data_to_dataset(data, sr, size=None):  # data.shape: (n)

    data, _ = librosa.effects.trim(data, top_db=top_db)
    spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=None, window=window_fn)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = spec / 100
    spec = spec.transpose()
    xs = []
    ys = []

    if size == None or size > spec.shape[-2] - 64:
        size = spec.shape[-2] - 64
        blank = 1
    else:
        blank = (spec.shape[-2] - 64) // size
    for i in range(size):
        xs.append(spec[i*blank:i*blank+64].copy())
        ys.append(spec[i*blank+64].copy())

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys


class MyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = torch.tensor(self.xs[idx], dtype=torch.float)
        y = torch.tensor(self.ys[idx], dtype=torch.float)
        return x, y


def make_dataset(file_path):

    wave, sampling_rate = librosa.load(file_path, sr=None, mono=False)

    a, b = wave

    xs1, ys1 = data_to_dataset(a, sampling_rate)
    xs2, ys2 = data_to_dataset(b, sampling_rate)
    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    dataset = MyDataset(xs, ys)

    ds_length = dataset.__len__()
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [ds_length - ds_length // 10, ds_length // 10])

    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=2)

    return train_loader, valid_loader
