import os
import time
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


# LJSpeechDataset using tensorflow_datasets
dataset, info = tfds.load("ljspeech", with_info=True)


class LJSpeechDataset:
    def __init__(self, path, batch_size, max_length):
        self.path = path
        self.batch_size = batch_size
        self.max_length = max_length

    def get_dataset(self):
        def _map_fn(data):
            text = data["text"]
            mel = data["mel_spectrogram"]
            stop = data["stop_token"]
            text = tf.cast(text, tf.int32)
            mel = tf.cast(mel, tf.float32)
            stop = tf.cast(stop, tf.float32)
            return text, mel, stop

        dataset = dataset["train"]
        dataset = dataset.map(_map_fn)
        dataset = dataset.shuffle(10000)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=([None], [None, 80], []),
            padding_values=(0, 0.0, 1.0),
            drop_remainder=True,
        )
        return dataset
