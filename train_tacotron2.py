import os
import random
import argparse
import tensorflow as tf
from datetime import datetime
from hparams import create_hparams, load_hparams, save_hparams
from tensorboard.plugins.hparams import api as hp


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='logs/tacotron2',
                        help='Directory to save checkpoints and Tensorboard log')

    parser.add_argument('--data_dir', type=str, default='data/moon', nargs='+',
                        help='Directory containing dataset')

    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to model to load')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model to load with out global step')

    # parser.add_argument('--hparams', type=str, default='',
    #                     help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    # parser.add_argument('--hparams_path', type=str, default=None,
    #                     help='Path to a YAML file containing hyperparameter overrides')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_test_per_speaker', type=int, default=2,
                        help='Number of test samples per speaker')
    parser.add_argument('--random_seed', type=int, default=random.randint(0, 100000),
                        help='Random seed')

    config = parser.parse_args()
    tf.random.set_seed(config.random_seed)

    log_path = os.path.join(
        config.log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_path, exist_ok=True)

    hparams = create_hparams()
    setattr(hparams, 'tacotron_batch_size', config.batch_size)
    setattr(hparams, 'num_speakers', len(config.data_dir))
    if config.load_path is not None:
        # overwrite hparams with saved ones
        load_hparams(hparams, os.path.join(config.load_path, 'hparams.json'))
    save_hparams(hparams, os.path.join(log_path, 'hparams.json'))

    print(hparams)
    # hp.hparams_config(hparams=hparams, metrics=[
    #     hp.Metric('loss', display_name='Loss')])


if __name__ == '__main__':
    main()
