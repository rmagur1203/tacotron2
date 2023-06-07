import json
from tensorboard.plugins.hparams import api as hp


class Hyperparams(dict):
    def __init__(self, hparams):
        self.hparams = hparams

    def __getitem__(self, __key):
        if __key in self.hparams:
            return self.hparams[__key]
        else:
            raise KeyError(f'{__key} is not in hparams')

    def __getattr__(self, name):
        if name in self.hparams:
            return self.hparams[name]
        else:
            raise AttributeError(f'{name} is not in hparams')

    def __setattr__(self, name, value):
        if name == 'hparams':
            super().__setattr__(name, value)
        else:
            if name not in self.hparams:
                print(f'Warning: {name} is not in hparams')
            self.hparams[name] = value

    def __delattr__(self, name):
        if name in self.hparams:
            del self.hparams[name]
        else:
            raise AttributeError(f'{name} is not in hparams')

    def __iter__(self):
        print('iter')
        for key in self.hparams:
            yield key

    def __dir__(self):
        print('dir')
        return super().__dir__() + list(self.hparams.keys())

    def __str__(self) -> str:
        return self.hparams.__str__()

    def keys(self):
        return self.hparams.keys()

    def toJSON(self, **kwargs):
        return json.dumps(self.hparams, **kwargs)


def create_hparams(hparams_string=None, hparams_path=None):
    hparams = Hyperparams({
        # Basic model parameters
        'tacotron_batch_size': 32,

        # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
        # text, you may want to use "basic_cleaners" or "transliteration_cleaners".
        # 학습 및 평가 전에 텍스트에서 실행할 클리너의 쉼표로 구분된 목록입니다.
        # 영어가 아닌 텍스트의 경우 "basic_cleaners" 또는 "transliteration_cleaners"를 사용할 수 있습니다.
        'cleaners': 'korean_cleaners',

        # npz파일에서 불필요한 것을 거르는 작업을 할지 말지 결정. receptive_field 보다 짧은 data를 걸러야 하기 때문에 해 줘야 한다.
        'skip_path_filter': False,
        'use_lws': False,

        # Audio
        'sample_rate': 24000,  #

        # shift can be specified by either hop_size(우선) or frame_shift_ms
        'hop_size': 300,             # frame_shift_ms = 12.5ms
        'fft_size': 2048,   # n_fft. 주로 1024로 되어있는데, tacotron에서 2048사용
        'win_size': 1200,   # 50ms
        'num_mels': 80,
        'num_freq': 1025,   # fft_size//2 + 1

        # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
        'preemphasize': True,  # whether to apply filter
        'preemphasis': 0.97,
        'min_level_db': -100,
        'ref_level_db': 20,
        # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        'signal_normalization': True,
        # Only relevant if mel_normalization = True
        'allow_clipping_in_normalization': True,
        # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
        'symmetric_mels': True,
        # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)
        'max_abs_value': 4.,

        'rescaling': True,
        'rescaling_max': 0.999,

        # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
        'trim_silence': True,
        # M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
        'trim_fft_size': 512,
        'trim_hop_size': 128,
        'trim_top_db': 23,

        # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
        'clip_mels_length': True,
        # Only relevant if clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
        'max_mel_frames': 900,

        # Coefficient in the L2 regularization.
        'l2_regularization_strength': 0,
        'sample_size': 9000,  # Concatenate and cut audio samples to this many samples
        # Volume threshold below which to trim the start and the end from the training set samples. e.g. 2
        'silence_threshold': 0,

        'filter_width': 3,
        # global_condition_vector의 차원. 이것 지정함으로써, global conditioning을 모델에 반영하라는 의미가 된다.
        'gc_channels': 32,

        'input_type': "raw",    # 'mulaw-quantize', 'mulaw', 'raw',   mulaw, raw 2가지는 scalar input
        'scalar_input': True,   # input_type과 맞아야 함.

        'dilations': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                      1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        'residual_channels': 128,
        'dilation_channels': 256,
        'quantization_channels': 256,
        'out_channels': 30,  # discretized_mix_logistic_loss를 적용하기 때문에, 3의 배수
        'skip_channels': 128,
        'use_biases': True,
        'upsample_type': 'SubPixel',  # 'SubPixel', None
        # np.prod(upsample_factor) must equal to hop_size
        'upsample_factor': [12, 25],

        # wavenet training hp
        # 16--> OOM. wavenet은 batch_size가 고정되어야 한다.
        'wavenet_batch_size': 2,
        'store_metadata': False,
        'num_steps': 1000000,                # Number of training steps

        # Learning rate schedule
        'wavenet_learning_rate': 1e-3,  # wavenet initial learning rate
        # Only used with 'exponential' scheme. Defines the decay rate.
        'wavenet_decay_rate': 0.5,
        # Only used with 'exponential' scheme. Defines the decay steps.
        'wavenet_decay_steps': 300000,

        # Regularization parameters
        # Whether the clip the gradients during wavenet training.
        'wavenet_clip_gradients': True,

        # residual 결과를 sum할 때,
        # Whether to use legacy mode: Multiply all skip outputs but the first one with sqrt(0.5) (True for more early training stability, especially for large models)
        'legacy': True,

        # residual block내에서  x = (x + residual) * np.sqrt(0.5)
        # Whether to scale residual blocks outputs by a factor of sqrt(0.5) (True for input variance preservation early in training and better overall stability)
        'residual_legacy': True,

        'wavenet_dropout': 0.05,

        'optimizer': 'adam',
        # 'Specify the momentum to be used by sgd or rmsprop optimizer. Ignored by the adam optimizer.
        'momentum': 0.9,
        # 'Maximum amount of checkpoints that will be kept alive. Default: '
        'max_checkpoints': 3,

        ####################################
        ####################################
        ####################################
        # TACOTRON HYPERPARAMETERS

        # Training
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,

        # Learning rate schedule
        # boolean, determines if the learning rate will follow an exponential decay
        'tacotron_decay_learning_rate': True,
        'tacotron_start_decay': 40000,  # Step at which learning decay starts
        # Determines the learning rate decay slope (UNDER TEST)
        'tacotron_decay_steps': 18000,
        'tacotron_decay_rate': 0.5,  # learning rate decay rate (UNDER TEST)
        'tacotron_initial_learning_rate': 1e-3,  # starting learning rate
        'tacotron_final_learning_rate': 1e-4,  # minimal learning rate

        'initial_data_greedy': True,
        # 여기서 지정한 step 이전에는 data_dirs의 각각의 디렉토리에 대하여 같은 수의 example을 만들고, 이후, weght 비듈에 따라 ... 즉, 아래의 'main_data_greedy_factor'의 영향을 받는다.
        'initial_phase_step': 8000,
        'main_data_greedy_factor': 0,
        # 이곳에 있는 directory 속에 있는 data는 가중치를 'main_data_greedy_factor' 만큼 더 준다.
        'main_data': [''],
        'prioritize_loss': False,

        # Model
        'model_type': 'multi-speaker',  # [single, multi-speaker]
        'speaker_embedding_size': 16,

        'embedding_size': 512,    # 'ᄀ', 'ᄂ', 'ᅡ' 에 대한 embedding dim
        'dropout_prob': 0.5,

        # reduction_factor가 적으면 더 많은 iteration이 필요하므로, 더 많은 메모리가 필요하다.
        'reduction_factor': 2,

        # Encoder
        'enc_conv_num_layers': 3,
        'enc_conv_kernel_size': 5,
        'enc_conv_channels': 512,
        'tacotron_zoneout_rate': 0.1,
        'encoder_lstm_units': 256,

        'attention_type': 'bah_mon_norm',    # 'loc_sen', 'bah_mon_norm'
        'attention_size': 128,

        # Attention mechanism
        'smoothing': False,  # Whether to smooth the attention normalization function
        'attention_dim': 128,  # dimension of attention space
        'attention_filters': 32,  # number of attention convolution filters
        'attention_kernel': (31, ),  # kernel size of attention convolution
        # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)
        'cumulative_weights': True,

        # Attention synthesis constraints
        # "Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
        # "Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
        # Whether to use attention windows constraints in synthesis
        'synthesis_constraint': False,
        # Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
        'synthesis_constraint_type': 'window',
        # Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.
        'attention_win_size': 7,

        # Loss params
        # whether to mask encoder padding while computing location sensitive attention. Set to True for better prosody but slower convergence.
        'mask_encoder': True,

        # Decoder
        # number of layers and number of units of prenet
        'prenet_layers': [256, 256],
        'decoder_layers': 2,  # number of decoder lstm layers
        'decoder_lstm_units': 1024,  # number of decoder lstm units on each layer

        # number of layers and number of units of prenet
        'dec_prenet_sizes': [256, 256],

        # Residual postnet
        'postnet_num_layers': 5,  # number of postnet convolutional layers
        # size of postnet convolution filters for each layer
        'postnet_kernel_size': (5, ),
        'postnet_channels': 512,  # number of postnet convolution filters for each layer

        # for linear mel spectrogrma
        'post_bank_size': 8,
        'post_bank_channel_size': 128,
        'post_maxpool_width': 2,
        'post_highway_depth': 4,
        'post_rnn_size': 128,
        'post_proj_sizes': [256, 80],  # num_mels=80
        'post_proj_width': 3,

        # regularization weight (for L2 regularization)
        'tacotron_reg_weight': 1e-6,
        'inference_prenet_dropout': True,

        # Eval
        # originally 50, 30 is good for korean,  text를 token으로 쪼갰을 때, 최소 길이 이상되어야 train에 사용
        'min_tokens': 30,
        # min_n_frame = reduction_factor * min_iters, reduction_factor와 곱해서 min_n_frame을 설정한다.
        'min_n_frame': 30*5,
        'max_n_frame': 200*5,
        'skip_inadequate': False,

        'griffin_lim_iters': 60,
        'power': 1.5,
    })

    if hparams.use_lws:
        # Does not work if fft_size is not multiple of hop_size!!
        # sample size = 20480, hop_size=256=12.5ms. fft_size는 window_size를 결정하는데, 2048을 시간으로 환산하면 2048/20480 = 0.1초=100ms
        hparams.sample_rate = 20480  #

        # shift can be specified by either hop_size(우선) or frame_shift_ms
        hparams.hop_size = 256             # frame_shift_ms = 12.5ms
        hparams.frame_shift_ms = None      # hop_size=  sample_rate *  frame_shift_ms / 1000
        hparams.fft_size = 2048   # 주로 1024로 되어있는데, tacotron에서 2048사용==> output size = 1025
        hparams.win_size = None  # 256x4 --> 50ms
    else:
        # 미리 정의되 parameter들로 부터 consistant하게 정의해 준다.
        hparams.num_freq = int(hparams.fft_size/2 + 1)
        # hop_size=  sample_rate *  frame_shift_ms / 1000
        hparams.frame_shift_ms = hparams.hop_size * 1000.0 / hparams.sample_rate
        hparams.frame_length_ms = hparams.win_size * 1000.0 / hparams.sample_rate

    return hparams


def load_hparams(hparams, hparams_path, except_keys=[]):
    print('Loading hparams from {}'.format(hparams_path))
    with open(hparams_path) as f:
        for key, value in json.load(f).items():
            if key in except_keys or key not in hparams.keys():
                print('Skip key: {}'.format(key))
                continue
            print('Load key: {} = {}'.format(key, value))
            setattr(hparams, key, value)
    return hparams


def save_hparams(hparams, hparams_path):
    print('Saving hparams to {}'.format(hparams_path))
    with open(hparams_path, 'w') as f:
        f.write(hparams.toJSON(indent=4, sort_keys=True))

# -*- coding: utf-8 -*-

#     tacotron_reg_weight = 1e-6, #regularization weight (for L2 regularization)
#     inference_prenet_dropout = True,


#     # Eval
#     min_tokens = 30,  #originally 50, 30 is good for korean,  text를 token으로 쪼갰을 때, 최소 길이 이상되어야 train에 사용
#     min_n_frame = 30*5,  # min_n_frame = reduction_factor * min_iters, reduction_factor와 곱해서 min_n_frame을 설정한다.
#     max_n_frame = 200*5,
#     skip_inadequate = False,

#     griffin_lim_iters = 60,
#     power = 1.5,

# )

# if hparams.use_lws:
#     # Does not work if fft_size is not multiple of hop_size!!
#     # sample size = 20480, hop_size=256=12.5ms. fft_size는 window_size를 결정하는데, 2048을 시간으로 환산하면 2048/20480 = 0.1초=100ms
#     hparams.sample_rate = 20480  #

#     # shift can be specified by either hop_size(우선) or frame_shift_ms
#     hparams.hop_size = 256             # frame_shift_ms = 12.5ms
#     hparams.frame_shift_ms=None      # hop_size=  sample_rate *  frame_shift_ms / 1000
#     hparams.fft_size=2048   # 주로 1024로 되어있는데, tacotron에서 2048사용==> output size = 1025
#     hparams.win_size = None # 256x4 --> 50ms


# else:
#     # 미리 정의되 parameter들로 부터 consistant하게 정의해 준다.
#     hparams.num_freq = int(hparams.fft_size/2 + 1)
#     hparams.frame_shift_ms = hparams.hop_size * 1000.0/ hparams.sample_rate      # hop_size=  sample_rate *  frame_shift_ms / 1000
#     hparams.frame_length_ms = hparams.win_size * 1000.0/ hparams.sample_rate


# def hparams_debug_string():
#     values = hparams.values()
#     hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
#     return 'Hyperparameters:\n' + '\n'.join(hp)


# Path: hparams.py
# Compare this snippet from train_tacotron.py:
# hparams = {
#     'batch_size': 32,
#     'epochs': 1000,
#     'learning_rate': 0.001,
#     'r': 5,
#     'alpha': 1.0,
#     'beta': 1.0,
#     'embedding_dim': 512,
#     'encoder_conv_filters': 512,
#     'encoder_conv_kernel_size': 5,
#     'encoder_conv_stride': 1,
#     'decoder_rnn_units': 1024,
#     'prenet_layers': [256, 128],
#     'max_iters': 200,
#     'griffin_lim_iters': 60,
#     'power': 1.5,
#     'outputs_per_step': 3,
#     'mu': 0.5,
#     'n_fft': 2048,
#     'num_mels': 80,
#     'frame_shift_ms': None,
#     'frame_length_ms': None,
#     'sample_rate': 22050,
#     'min_level_db': -100,
#     'ref_level_db': 20,
#     'preemphasis': 0.97,
#     'max_abs_value': 4.0,
#     'clip_norm': 1.0,
#     'mask_padding': True,
#     'random_seed': 42,
#     'use_lws': False,
#     'use_mixed_precision': False,
#     'checkpoint_interval': 1000,
#     'eval_interval': 1000,
#     'summary_interval': 1000,
#     'clear_Timeout': 1000,
#     'log_directory': 'logs',
#     'checkpoint_dir': 'checkpoints',
#     'samples_dir': 'samples',
#     'eval_dir': 'eval',
#     'tacotron_train_steps': 100000,
#     'tacotron_initial_learning_rate': 1e-3,
#     'tacotron_final_learning_rate': 1e-5,
#     'tacotron_decay_learning_rate': True,
#     'tacotron_start_decay': 50000,
#     'tacotron_decay_steps': 50000,
#     'tacotron_decay_rate': 0.5,
#     'tacotron_zoneout_rate': 0.1,
#     'tacotron_dropout_rate': 0.5,
#     'tacotron_clip_gradients': True,
#     'tacotron_positive_weight': 1.0,
#     'tacotron_reg_weight': 1e-6,
#     'tacotron_scale_regularization': True,
#     'tacotron_batch_size': 32,
#     'tacotron_random_seed': 42,
#     'tacotron_data_random_state': 1234,
#     'tacotron_natural_eval': False,
#     'tacotron_teacher_forcing_mode': 'constant',
#     'tacotron_teacher_forcing_ratio': 1.0,
#     'tacotron_teacher_forcing_init_ratio': 1.0,
#     'tacotron_teacher_forcing_final_ratio': 0.0,
#     'tacotron_teacher_forcing_start_decay': 50000,
#     'tacotron_teacher_forcing_decay_steps': 50000,
#     'tacotron_teacher_forcing_decay_alpha': 0.0,
#     'tacotron_teacher_forcing_schedule': 'constant',
#     'tacotron_initial_output_factor': 1.0,
#     'tacotron_final_output_factor': 1.0,
#     'tacotron_output_factor_start_decay': 50000,
#     'tacotron_output_factor_decay_steps': 50000,
#     'tacotron_output_factor_decay_alpha': 0.0,
#     'tacotron_output_factor_schedule': 'constant',
#     'tacotron_add_linear_outputs': True,
#     'tacotron_stop_at_any': True,
#     'tacotron_cleaners': 'english_cleaners',
#     'tacotron_symbol_embedding_dim': 512,
#     'tacotron_encoder_conv_layers': 3,
#     'tacotron_encoder_conv_filter_size': 5,
#     'tacotron_encoder_conv_channels': 512,
#     'tacotron_encoder_lstm_units': 256,
#     'tacotron_encoder_lstm_unidirectional': False,
#     'tacotron_encoder_lstm_layers': 1,
#     'tacotron_encoder_lstm_use_skip_connections': False,
#     'tacotron_encoder_lstm_type': 'bi',
#     'tacotron_encoder_lstm_residual': False,
#     'tacotron_encoder_lstm_projection': 0,
#     'tacotron_encoder_lstm_projection_bias': False,
#     'tacotron_encoder_lstm_zoneout': 0.0,
#     'tacotron_encoder_lstm_dropout': 0.0,
#     'tacotron_encoder_lstm_use_cudnn': True,
#     'tacotron_encoder_lstm_time_major': False,
#     'tacotron_encoder_lstm_impl': 'lstm',
#     'tacotron_encoder_lstm_skip_connections': False,
#     'tacotron_encoder_lstm_concat_layers': False,
#     'tacotron_encoder_lstm_concat_input': False,
#     'tacotron_encoder_lstm_residual_connections': False,
#     'tacotron_encoder_lstm_residual_dense': False,
#     'tacotron_encoder_lstm_residual_conv_filters': 0,
#     'tacotron_encoder_lstm_residual_conv_kernel': 1,
#     'tacotron_encoder_lstm_residual_conv_stride': 1,
#     'tacotron_encoder_lstm_residual_conv_dilation': 1,
#     'tacotron_encoder_lstm_residual_conv_groups': 1,
#     'tacotron_encoder_lstm_residual_conv_causal': False,
#     'tacotron_encoder_lstm_residual_conv_use_bias': True,
#     'tacotron_encoder_lstm_residual_conv_data_format': 'channels_last',
#     'tacotron_encoder_lstm_residual_conv_activation': 'linear',
#     'tacotron_encoder_lstm_residual_conv_recurrent_activation': 'sigmoid',
#     'tacotron_encoder_lstm_residual_conv_use_cudnn': True,
#     'tacotron_encoder_lstm_residual_conv_time_major': False,
#     'tacotron_encoder_lstm_residual_conv_impl': 'lstm',
#     'tacotron_encoder_lstm_residual_conv_skip_connections': False,
#     'tacotron_encoder_lstm_residual_conv_concat_layers': False,
#     'tacotron_encoder_lstm_residual_conv_concat_input': False,
#     'tacotron_encoder_lstm_residual_conv_residual_connections': False,
#     'tacotron_encoder_lstm_residual_conv_residual_dense': False,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_filters': 0,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_kernel': 1,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_stride': 1,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_dilation': 1,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_groups': 1,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_causal': False,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_use_bias': True,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_data_format': 'channels_last',
#     'tacotron_encoder_lstm_residual_conv_residual_conv_activation': 'linear',
#     'tacotron_encoder_lstm_residual_conv_residual_conv_recurrent_activation': 'sigmoid',
#     'tacotron_encoder_lstm_residual_conv_residual_conv_use_cudnn': True,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_time_major': False,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_impl': 'lstm',
#     'tacotron_encoder_lstm_residual_conv_residual_conv_skip_connections': False,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_concat_layers': False,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_concat_input': False,
#     'tacotron_encoder_lstm_residual_conv_residual_conv_residual_connections': False,
