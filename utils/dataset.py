import os
import time
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from glob import glob


class LJSpeechDataset:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        # LJSpeechDataset using tensorflow_datasets
        self.dataset, info = tfds.load("ljspeech", with_info=True)

    def get_dataset(self):
        def _map_fn(data):
            text = data["text"]
            mel = data["mel_spectrogram"]
            stop = data["stop_token"]
            text = tf.cast(text, tf.int32)
            mel = tf.cast(mel, tf.float32)
            stop = tf.cast(stop, tf.float32)
            return text, mel, stop

        dataset = self.dataset["train"]
        dataset = dataset.map(_map_fn)
        dataset = dataset.shuffle(10000)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=([None], [None, 80], []),
            padding_values=(0, 0.0, 1.0),
            drop_remainder=True,
        )
        return dataset


class KSSDataset:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

    def get_dataset(self):
        text_list = sorted(glob(os.path.join(self.path + "/text", "*.npy")))
        mel_list = sorted(glob(os.path.join(self.path + "/mel", "*.npy")))
        dec_list = sorted(glob(os.path.join(self.path + "/dec", "*.npy")))
        spec_list = sorted(glob(os.path.join(self.path + "/spec", "*.npy")))
        text_len = np.load(os.path.join(self.path + "/text_len.npy"))

        dataset = tf.data.Dataset.from_tensor_slices(
            (text_list, mel_list, dec_list, spec_list, text_len)
        )
        dataset = dataset.map(
            lambda text, mel, dec, spec, text_len: tuple(
                tf.py_function(
                    self._load_data,
                    [text, mel, dec, spec, text_len],
                    [tf.int32, tf.float32, tf.float32, tf.float32, tf.int32],
                )
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # dataset = dataset.shuffle(10000)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=([None], [None, 80], [None, 80], [None, None], []),
            padding_values=(0, 0.0, 0.0, 0.0, 0),
            drop_remainder=True,
        )
        return dataset

    def _load_data(self, text_path, mel_path, dec_path, spec_path, text_len):
        text = np.load(text_path.numpy())
        mel = np.load(mel_path.numpy())
        dec = np.load(dec_path.numpy())
        spec = np.load(spec_path.numpy())
        text_len = text_len.numpy()
        return text, mel, dec, spec, text_len
