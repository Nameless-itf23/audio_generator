import torch
import torch.nn as nn
import numpy as np
from scipy.io import wavfile

from models import model as mdl

import librosa
import yaml

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

with open('make_dataset/dataset_settings.yaml') as file:
    config = yaml.safe_load(file)

# for trim
top_db = config['top_db']

# for melspectrogram
n_mels = config['n_mels']
n_fft = config['n_fft']
hop_length = config['hop_length']
window_fn = config['window_fn']

file_path = "make_dataset/dataset/data.mp3"

wave, sampling_rate = librosa.load(file_path, sr=None, mono=False)

a, b = wave

data, _ = librosa.effects.trim(a, top_db=top_db)
spec = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=None, window=window_fn)
spec = librosa.power_to_db(spec, ref=np.max)
spec = spec / 100
spec = spec.transpose()

model = mdl.model()

model.load_state_dict(torch.load('weights/model_20230704_175111.pth', map_location=torch.device(device)))

ans = torch.from_numpy(spec[0:64].astype(np.float32)).clone()
ans = ans[None, ...]
for i in range(100):
    x = model(ans.squeeze()[0+i:64+i][None, ...])[None, ...]
    ans = torch.cat((ans, x), dim=1)

ans = ans.squeeze().detach().numpy()
audio = librosa.feature.inverse.mel_to_audio(ans, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, window=window_fn)

wavfile.write('test.wav', sampling_rate, audio)
