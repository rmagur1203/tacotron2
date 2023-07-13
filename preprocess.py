import pandas as pd
import numpy as np
import os, librosa, re, glob
from tqdm import tqdm
from utils.hparams import HParam
from utils.text import text_to_sequence

hp = HParam("./config/default.yaml")

sample_rate = hp.audio.sample_rate
n_fft = hp.audio.n_fft
hop_length = hp.audio.hop_length
win_length = hp.audio.win_length
preemphasis = hp.audio.preemphasis
max_db = hp.audio.max_db
ref_db = hp.audio.ref_db
mel_dim = hp.audio.n_mels
reduction = hp.model.reduction_factor

text_dir = glob.glob(os.path.join("./kss", "*.txt"))
filters = "([.,!?])"

metadata = pd.read_csv(text_dir[0], dtype="object", sep="|", header=None)
wav_dir = metadata[0].values
text = metadata[3].values

out_dir = "./data"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + "/text", exist_ok=True)
os.makedirs(out_dir + "/mel", exist_ok=True)
os.makedirs(out_dir + "/dec", exist_ok=True)
os.makedirs(out_dir + "/spec", exist_ok=True)

# text
print("Load Text")
text_len = []
for idx, s in enumerate(tqdm(text)):
    sentence = re.sub(re.compile(filters), "", s)
    sentence = text_to_sequence(sentence)
    text_len.append(len(sentence))
    text_name = "kss-text-%05d.npy" % idx
    np.save(os.path.join(out_dir + "/text", text_name), sentence, allow_pickle=False)
np.save(os.path.join(out_dir + "/text_len.npy"), np.array(text_len))
print("Text Done")

# audio
print("Load Audio")
mel_len_list = []
for idx, fn in enumerate(tqdm(wav_dir)):
    file_dir = "./kss/" + fn
    wav, _ = librosa.load(file_dir, sr=sample_rate)
    wav, _ = librosa.effects.trim(wav)
    wav = np.append(wav[0], wav[1:] - preemphasis * wav[:-1])
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft = np.abs(stft)
    mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=mel_dim)
    mel_spec = np.dot(mel_filter, stft)

    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec))
    stft = 20 * np.log10(np.maximum(1e-5, stft))

    mel_spec = np.clip((mel_spec - ref_db + max_db) / max_db, 1e-8, 1)
    stft = np.clip((stft - ref_db + max_db) / max_db, 1e-8, 1)

    mel_spec = mel_spec.T.astype(np.float32)
    stft = stft.T.astype(np.float32)
    mel_len_list.append([mel_spec.shape[0], idx])

    # padding
    remainder = mel_spec.shape[0] % reduction
    if remainder != 0:
        mel_spec = np.pad(
            mel_spec, [[0, reduction - remainder], [0, 0]], mode="constant"
        )
        stft = np.pad(stft, [[0, reduction - remainder], [0, 0]], mode="constant")

    mel_name = "kss-mel-%05d.npy" % idx
    np.save(os.path.join(out_dir + "/mel", mel_name), mel_spec, allow_pickle=False)

    stft_name = "kss-spec-%05d.npy" % idx
    np.save(os.path.join(out_dir + "/spec", stft_name), stft, allow_pickle=False)

    # Decoder Input
    mel_spec = mel_spec.reshape((-1, mel_dim * reduction))
    dec_input = np.concatenate(
        (np.zeros_like(mel_spec[:1, :]), mel_spec[:-1, :]), axis=0
    )
    dec_input = dec_input[:, -mel_dim:]
    dec_name = "kss-dec-%05d.npy" % idx
    np.save(os.path.join(out_dir + "/dec", dec_name), dec_input, allow_pickle=False)

mel_len = sorted(mel_len_list)
np.save(os.path.join(out_dir + "/mel_len.npy"), np.array(mel_len))
print("Audio Done")
