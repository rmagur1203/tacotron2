from copy import deepcopy
import glob
import os
import librosa
import numpy as np
import scipy
import tensorflow as tf
import soundfile as sf
from models.tacotron import PostNet
from utils.dataset import KSSDataset
from utils.hparams import HParam


hp = HParam("./config/default.yaml")

# load dataset
dataset = KSSDataset(hp.data.path, hp.data.batch_size)
dataset = dataset.get_dataset()

# load model
model = PostNet(
    mel_dim=hp.audio.n_mels,
    n_fft=hp.audio.n_fft,
)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=hp.train.lr)

# checkpoint
checkpoint_dir = hp.train.checkpoint_dir + "_post"
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=5
)

# restore checkpoint
checkpoint.restore(checkpoint_manager.latest_checkpoint)
if checkpoint_manager.latest_checkpoint:
    print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

def griffin_lim(spectrogram):
    spec = deepcopy(spectrogram)
    for i in range(50):
        est_wav = librosa.istft(spec, hop_length=hp.audio.hop_length, win_length=hp.audio.win_length)
        est_stft = librosa.stft(
            est_wav, n_fft=hp.audio.n_fft, hop_length=hp.audio.hop_length, win_length=hp.audio.win_length
        )
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spec = spectrogram * phase
    wav = librosa.istft(spec, hop_length=hp.audio.hop_length, win_length=hp.audio.win_length)
    return np.real(wav)

def generate(mel, idx=0):
    mel = np.expand_dims(mel, axis=0)
    pred = model(mel, training=False)

    pred = np.squeeze(pred, axis=0)
    pred = np.transpose(pred)

    pred = (np.clip(pred, 0, 1) * hp.audio.max_db) - hp.audio.max_db + hp.audio.ref_db
    pred = np.power(10.0, pred * 0.05)
    wav = griffin_lim(pred ** 1.5)
    wav = scipy.signal.lfilter([1], [1, -hp.audio.preemphasis], wav)
    wav = librosa.effects.trim(
        wav, frame_length=hp.audio.win_length, hop_length=hp.audio.hop_length)[0]
    wav = wav.astype(np.float32)
    sf.write(os.path.join('output', '{}.wav'.format(idx)), wav, hp.audio.sample_rate)

    return pred

for i, mel_path in enumerate(glob.glob(os.path.join("output", "*.npy"))):
    mel = np.load(mel_path)
    pred = generate(mel, i)
    print(pred.shape)

for (text, mel, dec, spec, text_len) in dataset:
    pred = generate(mel[0], -1)
    print(pred.shape)
    break