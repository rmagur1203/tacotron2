from utils.infolog import log
from text.symbols import symbols

import tensorflow as tf


class Tacotron2():
    def __init__(self, hparams) -> None:
        self.hparams

    def initialize(self, inputs, input_lengths, num_speakers, speaker_id=None, mel_targets=None, linear_targets=None, is_training=False, loss_coeff=None, stop_token_targets=None):

        with tf.variable_scope('Embedding') as scope:
            hp = self._hparams
            batch_size = tf.shape(inputs)[0]

            # Embeddings(256)
            char_embed_table = tf.get_variable('inputs_embedding', [len(
                symbols), hp.embedding_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))

            # transformer에 구현되어 있는 거 보고, 가져온 로직.
            # <PAD> 0 은 embedding이 0으로 고정되고, train으로 변하지 않는다. 즉, 위의 get_variable에서 잡았던 변수의 첫번째 행(<PAD>)에 대응되는 것은 사용되지 않는 것이다)
            char_embed_table = tf.concat(
                (tf.zeros(shape=[1, hp.embedding_size]), char_embed_table[1:, :]), 0)

            # [N, T_in, embedding_size]
            char_embedded_inputs = tf.nn.embedding_lookup(
                char_embed_table, inputs)

            self.num_speakers = num_speakers
            if self.num_speakers > 1:
                speaker_embed_table = tf.get_variable('speaker_embedding', [
                                                      self.num_speakers, hp.speaker_embedding_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
                # [N, T_in, speaker_embedding_size]
                speaker_embed = tf.nn.embedding_lookup(
                    speaker_embed_table, speaker_id)

                def deep_dense(x, dim, name): return tf.layers.dense(
                    x, dim, activation=tf.nn.softsign, name=name)   # softsign: x / (abs(x) + 1)

                encoder_rnn_init_state = deep_dense(
                    speaker_embed, hp.encoder_lstm_units * 4, 'encoder_init_dense')  # hp.encoder_lstm_units = 256

                decoder_rnn_init_states = [deep_dense(speaker_embed, hp.decoder_lstm_units*2, 'decoder_init_dense_{}'.format(
                    i)) for i in range(hp.decoder_layers)]  # hp.decoder_lstm_units = 1024

                speaker_embed = None
            else:
                # self.num_speakers =1인 경우
                speaker_embed = None
                encoder_rnn_init_state = None   # bidirectional GRU의 init state
                attention_rnn_init_state = None
                decoder_rnn_init_states = None

        with tf.variable_scope('Encoder') as scope:
            ##############
            # Encoder
            ##############
            x = char_embedded_inputs
            for i in range(hp.enc_conv_num_layers):
                x = tf.layers.conv1d(x, filters=hp.enc_conv_channels, kernel_size=hp.enc_conv_kernel_size,
                                     padding='same', activation=tf.nn.relu, name='Encoder_{}'.format(i))
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.layers.dropout(
                    x, rate=hp.dropout_prob, training=is_training, name='dropout_{}'.format(i))

            if encoder_rnn_init_state is not None:
                initial_state_fw_c, initial_state_fw_h, initial_state_bw_c, initial_state_bw_h = tf.split(
                    encoder_rnn_init_state, 4, 1)
                initial_state_fw = LSTMStateTuple(
                    initial_state_fw_c, initial_state_fw_h)
                initial_state_bw = LSTMStateTuple(
                    initial_state_bw_c, initial_state_bw_h)
            else:  # single mode
                initial_state_fw, initial_state_bw = None, None

            cell_fw = ZoneoutLSTMCell(hp.encoder_lstm_units, is_training, zoneout_factor_cell=hp.tacotron_zoneout_rate,
                                      zoneout_factor_output=hp.tacotron_zoneout_rate, name='encoder_fw_LSTM')
            cell_bw = ZoneoutLSTMCell(hp.encoder_lstm_units, is_training, zoneout_factor_cell=hp.tacotron_zoneout_rate,
                                      zoneout_factor_output=hp.tacotron_zoneout_rate, name='encoder_fw_LSTM')
            encoder_conv_output = x
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_conv_output, sequence_length=input_lengths,
                                                              initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype=tf.float32)

            # envoder_outpust = [N,T,2*encoder_lstm_units] = [N,T,512]
            # Concat and return forward + backward outputs
            encoder_outputs = tf.concat(outputs, axis=2)

        with tf.variable_scope('Decoder') as scope:

            ##############
            # Attention
            ##############
            if hp.attention_type == 'bah_mon':
                attention_mechanism = BahdanauMonotonicAttention(
                    hp.attention_size, encoder_outputs, memory_sequence_length=input_lengths, normalize=False)
            elif hp.attention_type == 'bah_mon_norm':  # hccho 추가
                attention_mechanism = BahdanauMonotonicAttention(
                    hp.attention_size, encoder_outputs, memory_sequence_length=input_lengths, normalize=True)
            elif hp.attention_type == 'loc_sen':  # Location Sensitivity Attention
                attention_mechanism = LocationSensitiveAttention(hp.attention_size, encoder_outputs, hparams=hp, is_training=is_training,
                                                                 mask_encoder=hp.mask_encoder, memory_sequence_length=input_lengths, smoothing=hp.smoothing, cumulate_weights=hp.cumulative_weights)
            elif hp.attention_type == 'gmm':  # GMM Attention
                attention_mechanism = GmmAttention(
                    hp.attention_size, memory=encoder_outputs, memory_sequence_length=input_lengths)
            elif hp.attention_type == 'bah_norm':
                attention_mechanism = BahdanauAttention(
                    hp.attention_size, encoder_outputs, memory_sequence_length=input_lengths, normalize=True)
            elif hp.attention_type == 'luong_scaled':
                attention_mechanism = LuongAttention(
                    hp.attention_size, encoder_outputs, memory_sequence_length=input_lengths, scale=True)
            elif hp.attention_type == 'luong':
                attention_mechanism = LuongAttention(
                    hp.attention_size, encoder_outputs, memory_sequence_length=input_lengths)
            elif hp.attention_type == 'bah':
                attention_mechanism = BahdanauAttention(
                    hp.attention_size, encoder_outputs, memory_sequence_length=input_lengths)
            else:
                raise Exception(
                    " [!] Unkown attention type: {}".format(hp.attention_type))

            decoder_lstm = [ZoneoutLSTMCell(hp.decoder_lstm_units, is_training, zoneout_factor_cell=hp.tacotron_zoneout_rate,
                                            zoneout_factor_output=hp.tacotron_zoneout_rate, name='decoder_LSTM_{}'.format(i+1)) for i in range(hp.decoder_layers)]

            decoder_lstm = tf.contrib.rnn.MultiRNNCell(
                decoder_lstm, state_is_tuple=True)
            # 여기서 zero_state를 부르면, 위의 AttentionWrapper에서 이미 넣은 준 값도 포함되어 있다.
            decoder_init_state = decoder_lstm.zero_state(
                batch_size=batch_size, dtype=tf.float32)

            if hp.model_type == "multi-speaker":

                decoder_init_state = list(decoder_init_state)

                for idx, cell in enumerate(decoder_rnn_init_states):
                    shape1 = decoder_init_state[idx][0].get_shape().as_list()
                    shape2 = cell.get_shape().as_list()
                    if shape1[1]*2 != shape2[1]:
                        raise Exception(
                            " [!] Shape {} and {} should be equal".format(shape1, shape2))
                    c, h = tf.split(cell, 2, 1)
                    decoder_init_state[idx] = LSTMStateTuple(c, h)

                decoder_init_state = tuple(decoder_init_state)

            attention_cell = AttentionWrapper(decoder_lstm, attention_mechanism, initial_cell_state=decoder_init_state,
                                              alignment_history=True, output_attention=False)  # output_attention=False 에 주목, attention_layer_size에 값을 넣지 않았다. 그래서 attention = contex vector가 된다.

            # attention_state_size = 256
            # Decoder input -> prenet -> decoder_lstm -> concat[output, attention]
            dec_prenet_outputs = DecoderWrapper(
                attention_cell, is_training, hp.dec_prenet_sizes, hp.dropout_prob, hp.inference_prenet_dropout)

            dec_outputs_cell = OutputProjectionWrapper(
                dec_prenet_outputs, (hp.num_mels+1) * hp.reduction_factor)

            if is_training:
                # inputs은 batch_size 계산에만 사용됨
                helper = TacoTrainingHelper(
                    mel_targets, hp.num_mels, hp.reduction_factor)
            else:
                helper = TacoTestHelper(
                    batch_size, hp.num_mels, hp.reduction_factor)

            decoder_init_state = dec_outputs_cell.zero_state(
                batch_size=batch_size, dtype=tf.float32)
            (decoder_outputs, _), final_decoder_state, _ = \
                tf.contrib.seq2seq.dynamic_decode(BasicDecoder(dec_outputs_cell, helper, decoder_init_state), maximum_iterations=int(
                    hp.max_n_frame/hp.reduction_factor))  # max_iters=200

            decoder_mel_outputs = tf.reshape(decoder_outputs[:, :, :hp.num_mels * hp.reduction_factor], [
                                             batch_size, -1, hp.num_mels])   # [N,iters,400] -> [N,5*iters,80]
            stop_token_outputs = tf.reshape(
                decoder_outputs[:, :, hp.num_mels * hp.reduction_factor:], [batch_size, -1])  # [N,iters]

            # Postnet
            x = decoder_mel_outputs
            for i in range(hp.postnet_num_layers):
                activation = tf.nn.tanh if i != (
                    hp.postnet_num_layers-1) else None
                x = tf.layers.conv1d(x, filters=hp.postnet_channels, kernel_size=hp.postnet_kernel_size,
                                     padding='same', activation=activation, name='Postnet_{}'.format(i))
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.layers.dropout(
                    x, rate=hp.dropout_prob, training=is_training, name='Postnet_dropout_{}'.format(i))

            residual = tf.layers.dense(
                x, hp.num_mels, name='residual_projection')
            mel_outputs = decoder_mel_outputs + residual

            # Add post-processing CBHG:
            # mel_outputs: (N,T,num_mels)
            post_outputs = cbhg(mel_outputs, None, is_training, hp.post_bank_size, hp.post_bank_channel_size, hp.post_maxpool_width, hp.post_highway_depth, hp.post_rnn_size,
                                hp.post_proj_sizes, hp.post_proj_width, scope='post_cbhg')

            linear_outputs = tf.layers.dense(
                post_outputs, hp.num_freq, name='linear_spectogram_projection')    # [N, T_out, F(1025)]

            # Grab alignments from the final decoder state:
            # batch_size, text length(encoder), target length(decoder)
            alignments = tf.transpose(
                final_decoder_state.alignment_history.stack(), [1, 2, 0])

            self.inputs = inputs
            self.speaker_id = speaker_id
            self.input_lengths = input_lengths
            self.loss_coeff = loss_coeff
            self.decoder_mel_outputs = decoder_mel_outputs
            self.mel_outputs = mel_outputs
            self.linear_outputs = linear_outputs
            self.alignments = alignments
            self.mel_targets = mel_targets
            self.linear_targets = linear_targets
            self.final_decoder_state = final_decoder_state
            self.stop_token_targets = stop_token_targets
            self.stop_token_outputs = stop_token_outputs
            self.all_vars = tf.trainable_variables()
            log('='*40)
            log(' model_type: %s' % hp.model_type)
            log('='*40)

            log('Initialized Tacotron model. Dimensions: ')
            log('    embedding:                %d' %
                char_embedded_inputs.shape[-1])
            log('    encoder conv out:               %d' %
                encoder_conv_output.shape[-1])
            log('    encoder out:              %d' % encoder_outputs.shape[-1])
            log('    attention out:            %d' %
                attention_cell.output_size)
            log('    decoder prenet lstm concat out :        %d' %
                dec_prenet_outputs.output_size)
            log('    decoder cell out:         %d' %
                dec_outputs_cell.output_size)
            log('    decoder out (%d frames):  %d' %
                (hp.reduction_factor, decoder_outputs.shape[-1]))
            log('    decoder mel out:    %d' % decoder_mel_outputs.shape[-1])
            log('    mel out:    %d' % mel_outputs.shape[-1])
            log('    postnet out:              %d' % post_outputs.shape[-1])
            log('    linear out:               %d' % linear_outputs.shape[-1])
            log('  Tacotron Parameters       {:.3f} Million.'.format(
                np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))
