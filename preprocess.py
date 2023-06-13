# Preprocess LJSpeech dataset

import os
import json
import numpy as np
from tqdm import tqdm
from utils.audio import melspectrogram
from utils.text import text_to_sequence


# hyper parameters
class Hparams:
    def __init__(self):
        self.path = "LJSpeech-1.1"
        self.max_length = 128
        self.sampling_rate = 22050
        self.n_fft = 2048
        self.frame_shift = 0.0125
        self.frame_length = 0.05
        self.hop_length = int(self.sampling_rate * self.frame_shift)
        self.win_length = int(self.sampling_rate * self.frame_length)
        self.n_mels = 80
        self.power = 1.2
        self.max_db = 100
        self.ref_db = 20
        self.preemphasis = 0.97
        self.n_iter = 50
        self.max_grad_norm = 100
        self.embed_size = 256
        self.enc_units = 256
        self.dec_units = 256
        self.batch_size = 32
        self.epochs = 100
        self.lr = 0.001
        self.logdir = "logdir"
        self.sampledir = "samples"
        self.restore = False
        self.save_step = 5
        self.synthesis = True


hp = Hparams()

# load metadata
with open(os.path.join(hp.path, "metadata.csv"), "r", encoding="utf-8") as f:
    metadata = f.read().splitlines()
