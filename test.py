from copy import deepcopy
import os
import librosa
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import scipy
import tensorflow as tf
import soundfile as sf
from tqdm import tqdm
from models.tacotron import Tacotron
from utils.dataset import KSSDataset
from utils.hparams import HParam
from utils.text import sequence_to_text, text_to_sequence


hp = HParam("./config/default.yaml")

# load dataset
dataset = KSSDataset(hp.data.path, hp.data.batch_size)
dataset = dataset.get_dataset()

# load model
model = Tacotron(
    vocab_size=hp.model.vocab_size,
    embedding_dim=hp.model.embedding_dim,
    enc_units=hp.model.enc_units,
    dec_units=hp.model.dec_units,
    batch_size=1,
    reduction=hp.model.reduction_factor,
)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=hp.train.lr)

# checkpoint
checkpoint_dir = hp.train.checkpoint_dir
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

matplotlib.rc("font", family="NanumBarunGothic", size=14)


def plot_alignment(alignment, path, text):
    text = text.rstrip("_").rstrip("~")
    alignment = alignment[: len(text)]
    _, ax = plt.subplots(figsize=(len(text) / 3, 5))
    ax.imshow(tf.transpose(alignment), aspect="auto", origin="lower")
    plt.xlabel("Encoder timestep")
    plt.ylabel("Decoder timestep")
    text = [x if x != " " else "" for x in list(text)]
    plt.xticks(range(len(text)), text)
    plt.tight_layout()
    plt.savefig(path, format="png")


def generate(text, idx=0):
    seq = text_to_sequence(text)
    seq_len = int(len(seq) * 1.5)
    reduction = hp.model.reduction_factor
    enc_input = np.asarray([seq], dtype=np.int32)
    dec_input = np.zeros((1, seq_len, hp.audio.n_mels), dtype=np.float32)

    pred = []
    for i in tqdm(range(1, seq_len + 1)):
        mel_out, alignment = model(enc_input, dec_input, training=False)
        if i < seq_len:
            dec_input[:, i, :] = mel_out[:, reduction * i - 1, :]
        pred.extend(mel_out[:, reduction * (i - 1) : reduction * i, :])

    pred = np.reshape(np.asarray(pred), [-1, hp.audio.n_mels])
    alignment = np.squeeze(alignment, axis=0)

    np.save(os.path.join("output", "mel-{0}".format(idx)), pred, allow_pickle=False)

    input_seq = sequence_to_text(seq)
    alignment_dir = os.path.join("output", "align-{}.png".format(idx))
    plot_alignment(alignment, alignment_dir, input_seq)


generate("""3학년 3반 박금혁입니다.""", idx=1)
