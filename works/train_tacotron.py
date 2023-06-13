import argparse
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/LJSpeech-1.1',
                        help='Directory containing the LJSpeech dataset')

    config = parser.parse_args()

    setattr(config, 'output_directory', 'output')


if __name__ == '__main__':
    main()
