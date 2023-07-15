# train tacotron

import os
import time
import librosa
import numpy as np
import scipy
import tensorflow as tf
from tqdm import tqdm
from models.cbhg import CBHG
from models.tacotron import PostNet, Tacotron
from utils.dataset import KSSDataset
from utils.audio import griffin_lim
from utils.hparams import HParam
from scipy.io import wavfile
import datetime

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

# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = hp.train.log_dir + current_time + "/post-train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# restore checkpoint
checkpoint.restore(checkpoint_manager.latest_checkpoint)
if checkpoint_manager.latest_checkpoint:
    print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


# train
@tf.function(experimental_relax_shapes=True)
def train_step(mel, spec):
    with tf.GradientTape() as tape:
        pred = model(mel, training=True)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(spec, pred))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, pred


for epoch in range(hp.train.epochs):
    start = time.time()
    total_loss = 0.0
    # total_loss_post = 0.0
    for batch, (text, mel, dec, spec, text_len) in enumerate(dataset):
        loss, pred = train_step(mel, spec)
        total_loss += loss
        # if batch % 100 == 0:
        print(
            "Epoch {}/{} Batch {}/{} Loss {:.4f}".format(
                epoch + 1,
                hp.train.epochs,
                batch + 1,
                dataset.cardinality().numpy(),
                loss.numpy(),
            )
        )
    # save checkpoint
    checkpoint_manager.save()
    # save audio
    # audio = pred[0].numpy()
    # audio = np.squeeze(audio)
    # audio = np.transpose(audio)
    # audio = np.clip(audio, 0, 1) * hp.audio.max_db - hp.audio.max_db + hp.audio.ref_db
    # audio = griffin_lim(audio, hp.audio.n_fft, hp.audio.hop_length, hp.audio.win_length, hp.audio.num_iters)
    # audio = scipy.signal.lfilter([1], [1, -hp.audio.preemphasis], audio)
    # audio = librosa.effects.trim(
    #     audio, frame_length=hp.audio.win_length, hop_length=hp.audio.hop_length)[0]
    # audio = audio.astype(np.float32)
    
    # wavfile.write(
    #     "./audio/epoch{}_loss{:.4f}.wav".format(epoch + 1, total_loss / (batch + 1)),
    #     22050,
    #     audio,
    # )
    # tensorboard
    with train_summary_writer.as_default():
        # tf.summary.audio(
        #     "audio", audio, sample_rate=22050, step=epoch, max_outputs=1
        # )
        tf.summary.scalar("loss", total_loss / (batch + 1), step=epoch)
    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / (batch + 1)))
    print("Time taken for 1 epoch {} sec\n".format(time.time() - start))