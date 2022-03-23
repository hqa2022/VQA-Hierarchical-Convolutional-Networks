import sys

sys.path.append('..')
import tensorflow as tf
from tools import layers, utils
import random
import pickle as pkl


def load_file(filename):
    with open(filename, 'rb') as f1:
        return pkl.load(f1)


class Model(object):

    def __init__(self, params):
        # self.batch_size = params['batch_size']
        self.params = params
        print(params)
        self.input_frame_dim = params['input_video_dim']
        self.input_n_frames = params['max_n_frames']
        self.input_ques_dim = params['input_ques_dim']
        self.max_n_q_words = params['max_n_q_words']
        self.max_n_a_words = params['max_n_a_words']
        self.n_words = params['n_words']

        self.ref_dim = params['ref_dim']
        self.lstm_dim = params['lstm_dim']
        self.second_lstm_dim = params['second_lstm_dim']
        self.attention_dim = params['attention_dim']
        self.regularization_beta = params['regularization_beta']
        self.dropout_prob = params['dropout_prob']

        self.decode_dim = params['decode_dim']

    def build_train_proc(self):
        # input layer (batch_size, n_steps, input_dim)
        self.input_q = tf.placeholder(tf.float32, [None, self.max_n_q_words, self.input_ques_dim])
        self.input_q_len = tf.placeholder(tf.int32, [None])
        self.input_x = tf.placeholder(tf.float32, [None, self.input_n_frames, self.input_frame_dim])
        self.input_x_len = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, self.max_n_a_words])
        self.y_mask = tf.placeholder(tf.float32, [None, self.max_n_a_words])
        self.ans_vec = tf.placeholder(tf.float32, [None, self.max_n_a_words, self.input_ques_dim])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.is_training = tf.placeholder(tf.bool)
        self.reward = tf.placeholder(tf.float32, [None])

        # video LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
        input_x = tf.contrib.layers.dropout(self.input_x, self.dropout_prob, is_training=self.is_training)
        v_lstm_output, _ = layers.dynamic_origin_lstm_layer(input_x, self.lstm_dim, 'v_lstm', input_len = self.input_x_len)
        v_lstm_output = tf.contrib.layers.dropout(v_lstm_output, self.dropout_prob, is_training=self.is_training)

        # Question LSTM layer, [n_steps * (batch_size, input_dim)] -> [n_steps * (batch_size, 2*lstm_dim)]
        input_q = tf.contrib.layers.dropout(self.input_q, self.dropout_prob, is_training=self.is_training)
        q_lstm_output, q_lstm_state = layers.dynamic_origin_lstm_layer(input_q, self.lstm_dim, 'q_lstm', input_len=self.input_q_len)
        q_lstm_output = tf.contrib.layers.dropout(q_lstm_output, self.dropout_prob, is_training=self.is_training)
        q_last_state = tf.contrib.layers.dropout(q_lstm_state[1], self.dropout_prob, is_training=self.is_training)

        # local attention layer (batch_size, max_q_n_words, q_dim) , [n_steps * (batch_size, 2*lstm_dim)] -> [batch_size, 2*lstm_dim]
        v_first_attention_output, first_attention_score_list = layers.collective_matrix_attention_layer(v_lstm_output, q_lstm_output, self.attention_dim, 'v_first_local_attention', context_len=self.input_q_len, use_maxpooling=False)
        v_global_attention_output, first_attention_score = layers.matrix_attention_layer(v_lstm_output, q_last_state, self.attention_dim, 'v_global_attention')

        # video attention lstm
        v_input_att = tf.contrib.layers.dropout(v_first_attention_output, self.dropout_prob, is_training=self.is_training)
        v_att_lstm_output, _ = layers.dynamic_origin_lstm_layer(v_input_att, self.lstm_dim, 'v_att_lstm', input_len=self.input_q_len)
        v_att_lstm_output = tf.contrib.layers.dropout(v_att_lstm_output, self.dropout_prob, is_training=self.is_training)

        #att_last_state = tf.contrib.layers.dropout(att_lstm_state[1], self.dropout_prob, is_training=self.is_training)

        # second attention (batch_size, input_video_dim)
        v_second_attention_output, second_attention_score = layers.matrix_attention_layer(v_att_lstm_output, q_last_state, self.attention_dim, 'v_second_local_attention')

        self.attention = tf.reduce_sum(tf.multiply(first_attention_score_list, tf.expand_dims(second_attention_score, 2)),1)

        # dot product
        #qv_dot = tf.multiply(q_last_state, v_last_state)

        # concatenation
        concat_output = tf.concat([q_last_state, v_global_attention_output, v_second_attention_output], axis=1)
        self.v_first_lstm_output = v_lstm_output
        self.q_last_state = q_last_state
        print(self.v_first_lstm_output.shape)


        # decoder

        # output -> first_atten
        # self.decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.decode_dim)
        self.decoder_cell = tf.contrib.rnn.GRUCell(self.decode_dim)

        with tf.variable_scope('linear'):
            decoder_input_W = tf.get_variable('w', shape=[concat_output.shape[1], self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())  # initializer=tf.random_normal_initializer(stddev=0.03))
            decoder_input_b = tf.get_variable('b', shape=[self.decode_dim], dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())  # initializer=tf.random_normal_initializer(stddev=0.03))
            self.decoder_input = tf.matmul(concat_output, decoder_input_W) + decoder_input_b  # [None, decode_dim]

        # answer->word predict
        self.embed_word_W = tf.Variable(tf.random_uniform([self.decode_dim, self.n_words], -0.1, 0.1),
                                        name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.random_uniform([self.n_words], -0.1, 0.1), name='embed_word_b')

        # word dim -> decode_dim
        self.word_to_lstm_w = tf.Variable(tf.random_uniform([self.input_ques_dim, self.decode_dim], -0.1, 0.1),
                                          name='word_to_lstm_W')
        self.word_to_lstm_b = tf.Variable(tf.random_uniform([self.decode_dim], -0.1, 0.1), name='word_to_lstm_b')

        # decoder attention layer
        with tf.variable_scope('decoder_attention'):
            self.attention_w_q = tf.get_variable('attention_w_q', shape=[self.lstm_dim, self.attention_dim],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.attention_w_x = tf.get_variable('attention_w_x', shape=[self.lstm_dim, self.attention_dim],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.attention_w_h = tf.get_variable('attention_w_h', shape=[self.decode_dim, self.attention_dim],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.attention_b = tf.get_variable('attention_b', shape=[self.attention_dim], dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.attention_a = tf.get_variable('attention_a', shape=[self.attention_dim, 1], dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.attention_to_decoder = tf.get_variable('attention_to_decoder', shape=[self.lstm_dim, self.decode_dim],
                                                        dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer())
        # decoder
        with tf.variable_scope('decoder'):
            self.decoder_r = tf.get_variable('decoder_r', shape=[self.decode_dim * 3, self.decode_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_z = tf.get_variable('decoder_z', shape=[self.decode_dim * 3, self.decode_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.decoder_w = tf.get_variable('decoder_w', shape=[self.decode_dim * 3, self.decode_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())

        # embedding layer
        embeddings = load_file(self.params['word_embedding'])
        self.Wemb = tf.constant(embeddings, dtype=tf.float32)

        # generate training
        answer_train, train_loss = self.generate_answer_on_training()
        answer_test, test_loss = self.generate_answer_on_testing()

        # final
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
        self.answer_word_train = answer_train
        self.train_loss = train_loss + self.regularization_beta * regularization_cost

        self.answer_word_test = answer_test
        self.test_loss = test_loss + self.regularization_beta * regularization_cost
        tf.summary.scalar('training cross entropy', self.train_loss)

    def generate_answer_on_training(self):
        with tf.variable_scope("decoder"):
            answer_train = []
            decoder_state = self.decoder_cell.zero_state(self.batch_size, tf.float32)
            loss = 0.0

            with tf.variable_scope("lstm") as scope:
                for i in range(self.max_n_a_words):
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        scope.reuse_variables()
                        # next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        # current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)
                        current_emb = tf.nn.xw_plus_b(self.ans_vec[:, i - 1, :], self.word_to_lstm_w,
                                                      self.word_to_lstm_b)

                    # decoder_state
                    tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1),
                                                    tf.stack([1, self.input_n_frames, 1]))
                    tiled_q_last_state = tf.tile(tf.expand_dims(self.q_last_state, 1),
                                                 tf.stack([1, self.input_n_frames, 1]))
                    attention_input = tf.tanh(utils.tensor_matmul(self.v_first_lstm_output, self.attention_w_x)
                                              + utils.tensor_matmul(tiled_q_last_state, self.attention_w_q)
                                              + utils.tensor_matmul(tiled_decoder_state_h, self.attention_w_h)
                                              + self.attention_b)
                    attention_score = tf.nn.softmax(
                        tf.squeeze(utils.tensor_matmul(attention_input, self.attention_a), axis=[2]))
                    attention_output = tf.reduce_sum(
                        tf.multiply(self.v_first_lstm_output, tf.expand_dims(attention_score, 2)), 1)
                    attention_decoder = tf.matmul(attention_output, self.attention_to_decoder)

                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, attention_decoder, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_z))
                    decoder_middle = tf.concat(
                        [tf.multiply(decoder_r_t, decoder_state), tf.multiply(decoder_r_t, attention_decoder),
                         current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t,
                                                                                                decoder_state_)

                    output = decoder_state

                    # ground truth
                    labels = tf.expand_dims(self.y[:, i], 1)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat([indices, labels], 1)
                    onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    max_prob_word = tf.argmax(logit_words, 1)
                    answer_train.append(max_prob_word)

                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask[:, 1:])
            return answer_train, loss

    def generate_answer_on_testing(self):
        with tf.variable_scope("decoder"):
            answer_test = []
            decoder_state = self.decoder_cell.zero_state(self.batch_size, tf.float32)
            loss = 0.0

            with tf.variable_scope("lstm") as scope:
                for i in range(self.max_n_a_words):
                    scope.reuse_variables()
                    if i == 0:
                        current_emb = self.decoder_input
                    else:
                        next_word_vec = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                        current_emb = tf.nn.xw_plus_b(next_word_vec, self.word_to_lstm_w, self.word_to_lstm_b)

                    # decoder_state
                    tiled_decoder_state_h = tf.tile(tf.expand_dims(decoder_state, 1),
                                                    tf.stack([1, self.input_n_frames, 1]))
                    tiled_q_last_state = tf.tile(tf.expand_dims(self.q_last_state, 1),
                                                 tf.stack([1, self.input_n_frames, 1]))
                    attention_input = tf.tanh(utils.tensor_matmul(self.v_first_lstm_output, self.attention_w_x)
                                              + utils.tensor_matmul(tiled_q_last_state, self.attention_w_q)
                                              + utils.tensor_matmul(tiled_decoder_state_h, self.attention_w_h)
                                              + self.attention_b)
                    attention_score = tf.nn.softmax(
                        tf.squeeze(utils.tensor_matmul(attention_input, self.attention_a), axis=[2]))
                    attention_output = tf.reduce_sum(
                        tf.multiply(self.v_first_lstm_output, tf.expand_dims(attention_score, 2)), 1)
                    attention_decoder = tf.matmul(attention_output, self.attention_to_decoder)

                    # decoder : GRU with attention
                    decoder_input = tf.concat([decoder_state, attention_decoder, current_emb], axis=1)
                    decoder_r_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_r))
                    decoder_z_t = tf.nn.sigmoid(tf.matmul(decoder_input, self.decoder_z))
                    decoder_middle = tf.concat(
                        [tf.multiply(decoder_r_t, decoder_state), tf.multiply(decoder_r_t, attention_decoder),
                         current_emb], axis=1)
                    decoder_state_ = tf.tanh(tf.matmul(decoder_middle, self.decoder_w))
                    decoder_state = tf.multiply((1 - decoder_z_t), decoder_state) + tf.multiply(decoder_z_t,
                                                                                                decoder_state_)

                    output = decoder_state

                    # ground truth
                    labels = tf.expand_dims(self.y[:, i], 1)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat([indices, labels], 1)
                    onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    logit_words = tf.nn.xw_plus_b(output, self.embed_word_W, self.embed_word_b)
                    max_prob_word = tf.argmax(logit_words, 1)
                    answer_test.append(max_prob_word)

                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
                    # cross_entropy = cross_entropy * self.reward
                    cross_entropy = cross_entropy * self.y_mask[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(self.y_mask[:, 1:])
            return answer_test, loss

    def build_model(self):
        # build all graphes
        self.build_train_proc()
