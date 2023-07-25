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
model = Tacotron(
    vocab_size=hp.model.vocab_size,
    embedding_dim=hp.model.embedding_dim,
    enc_units=hp.model.enc_units,
    dec_units=hp.model.dec_units,
    batch_size=hp.data.batch_size,
    reduction=hp.model.reduction_factor,
)

# post_model = PostNet(
#     mel_dim=hp.audio.n_mels,
#     n_fft=hp.audio.n_fft,
# )

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=hp.train.lr)

# checkpoint
checkpoint_dir = hp.train.checkpoint_dir
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=5
)

# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = hp.train.log_dir + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# restore checkpoint
checkpoint.restore(checkpoint_manager.latest_checkpoint)
if checkpoint_manager.latest_checkpoint:
    print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


# train
@tf.function(experimental_relax_shapes=True)
def train_step(text, mel, dec):
    with tf.GradientTape() as tape:
        predictions, alignment = model(text, dec, training=True)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(mel, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions, alignment

for epoch in range(hp.train.epochs):
    start = time.time()
    total_loss = 0.0
    for batch, (text, mel, dec, spec, text_len) in enumerate(dataset):
        loss, pred, alignment = train_step(text, mel, dec)
        total_loss += loss
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
    # audio = audio[0].numpy()
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
    # plot alignment
    alignment = alignment[0].numpy()
    alignment = np.squeeze(alignment)
    alignment = alignment[::-1]
    alignment = tf.expand_dims(alignment, axis=-1)
    alignment = (alignment - np.min(alignment)) / (
        np.max(alignment) - np.min(alignment)
    )
    # tensorboard
    with train_summary_writer.as_default():
        tf.summary.image("alignment", [alignment], step=epoch)
        tf.summary.scalar("loss", total_loss / (batch + 1), step=epoch)
    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / (batch + 1)))
    print("Time taken for 1 epoch {} sec\n".format(time.time() - start))


# # synthesis
# def synthesis(text):
#     inputs = tf.convert_to_tensor([dataset.text_to_sequence(text)])
#     enc_output = model.encoder(inputs)
#     dec_hidden = enc_output[:, -1, :]
#     dec_input = tf.expand_dims([1], 0)
#     predictions, dec_hidden, alignments = model.decoder(
#         dec_input, dec_hidden, enc_output
#     )
#     predictions = tf.nn.softmax(predictions, axis=-1)
#     audio = griffin_lim(predictions)
#     return audio


# # save audio
# def save_audio(audio, path):
#     audio = audio.astype(np.float32)
#     audio = audio * 32767 / max(0.01, np.max(np.abs(audio)))
#     audio = audio.astype(np.int16)
#     wavfile.write(path, 22050, audio)


# synthesis
# text = [
#     "Scientists at the CERN laboratory say they have discovered a new particle.",
#     "There's a way to measure the acute emotional intelligence that has never gone out of style.",
#     "President Trump met with other leaders at the Group of 20 conference.",
#     "Generative adversarial network or variational auto-encoder.",
# ]
# for i, t in enumerate(text):
#     audio = synthesis(t)
#     save_audio(audio, "./audio/synthesis{}.wav".format(i))

# # synthesis
# text = "Scientists at the CERN laboratory say they have discovered a new particle."
# audio = synthesis(text)
# save_audio(audio, "./audio/synthesis.wav")

# # synthesis
# text = "There's a way to measure the acute emotional intelligence that has never gone out of style."
# audio = synthesis(text)
# save_audio(audio, "./audio/synthesis2.wav")

# # synthesis
# text = "President Trump met with other leaders at the Group of 20 conference."
# audio = synthesis(text)
# save_audio(audio, "./audio/synthesis3.wav")

# # synthesis
# text = "Generative adversarial network or variational auto-encoder."
# audio = synthesis(text)
# save_audio(audio, "./audio/synthesis4.wav")
