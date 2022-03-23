import tensorflow as tf
from models.conv_utils import *
import pickle as pkl
from collections import namedtuple

def load_file(filename):
    with open(filename,'rb') as f1:
        return pkl.load(f1)

EncoderOutput = namedtuple(
    "EncoderOutput",
    "frame_final_output frame_value_output frame_final_state frame_length ques_final_output ques_value_output ques_final_state ques_length")

class ConvBeamSearchModel(object):

    def __init__(self, params):

        self.params = params
        print(params)
        self.input_frame_dim = params['input_video_dim']
        self.input_n_frames = params['max_n_frames']
        self.input_ques_dim = params['input_ques_dim']
        self.input_ans_dim = params['input_ques_dim']
        self.max_n_q_words = params['max_n_q_words']
        self.max_n_a_words = params['max_n_a_words']
        self.vocab_size = params['n_words']
        self.regularization_beta = params['regularization_beta']
        self.beam_width = params['beam_width']



        if self.params["position_embeddings_enable"]:
            self.encoder_frame_pos_embed = tf.get_variable('en_frame_pos_emb', shape=[self.input_n_frames,
                                            self.input_frame_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
            self.encoder_ques_pos_embed = tf.get_variable('en_ques_pos_emb', shape=[self.max_n_q_words,
                                            self.input_ques_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
            self.decoder_pos_embed = tf.get_variable('de_pos_emb', shape=[self.max_n_a_words,
                                            self.input_ans_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(), trainable=False)

        vocab_embeddings = load_file(self.params['word_embedding'])
        self.vocab_embeddings = tf.constant(vocab_embeddings, dtype=tf.float32)

        self.build_encoder()
        self.build_decoder()


    def build_encoder(self):
        # input layer (batch_size, n_steps, input_dim)
        self.ques_vecs = tf.placeholder(tf.float32, [None, self.max_n_q_words, self.input_ques_dim])
        self.ques_len = tf.placeholder(tf.int32, [None])
        self.frame_vecs = tf.placeholder(tf.float32, [None, self.input_n_frames, self.input_frame_dim])
        self.frame_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.is_training = tf.placeholder(tf.bool)




        if self.params["position_embeddings_enable"]:
            frame_positions_embed = tf.tile([self.encoder_frame_pos_embed],[self.batch_size, 1, 1])
            self.frame_vecs = tf.add(self.frame_vecs, frame_positions_embed)
            ques_positions_embed = tf.tile([self.encoder_ques_pos_embed], [self.batch_size, 1, 1])
            self.ques_vecs = tf.add(self.ques_vecs, ques_positions_embed)

        # Apply dropout to embeddings
        self.frame_vecs = tf.contrib.layers.dropout(
            inputs=self.frame_vecs,
            keep_prob=self.params["embedding_dropout_keep_prob"],
            is_training=self.is_training
        )

        self.ques_vecs = tf.contrib.layers.dropout(
            inputs=self.ques_vecs,
            keep_prob=self.params["embedding_dropout_keep_prob"],
            is_training=self.is_training

        )



        with tf.variable_scope("encoder_frame_cnn"):
            frame_next_layer = self.frame_vecs

            frame_nhids_list = parse_list_or_default(self.params["encoder_nhids"], self.params["encoder_layers"],
                                                   self.params["encoder_nhid_default"])
            frame_kwidths_list = parse_list_or_default(self.params["encoder_kwidths"], self.params["encoder_layers"],
                                                     self.params["encoder_kwidth_default"])

            # mapping emb dim to hid dim
            frame_next_layer = linear_mapping_weightnorm(frame_next_layer, frame_nhids_list[0],
                                                       dropout=self.params["embedding_dropout_keep_prob"],
                                                       var_scope_name="linear_mapping_before_cnn")
            frame_next_layer = conv_encoder_stack(frame_next_layer, frame_nhids_list, frame_kwidths_list,
                                                {'src': self.params["embedding_dropout_keep_prob"],
                                                 'hid': self.params["nhid_dropout_keep_prob"]}, mode=self.is_training)

            frame_next_layer = linear_mapping_weightnorm(frame_next_layer, self.input_frame_dim,
                                                       var_scope_name="linear_mapping_after_cnn")

            ## The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
            ## cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))
            self.frame_final_output = frame_next_layer
            self.frame_value_output = (frame_next_layer + self.frame_vecs) * tf.sqrt(0.5)
            self.frame_final_state = tf.reduce_mean(self.frame_value_output, 1)

        with tf.variable_scope("encoder_ques_cnn"):
            ques_next_layer = self.ques_vecs

            ques_nhids_list = parse_list_or_default(self.params["ques_nhids"], self.params["ques_layers"],
                                                   self.params["ques_nhid_default"])
            ques_kwidths_list = parse_list_or_default(self.params["ques_kwidths"], self.params["ques_layers"],
                                                     self.params["ques_kwidth_default"])

            # mapping emb dim to hid dim
            ques_next_layer = linear_mapping_weightnorm(ques_next_layer, ques_nhids_list[0],
                                                             dropout=self.params["embedding_dropout_keep_prob"],
                                                             var_scope_name="linear_mapping_before_cnn")
            ques_next_layer = conv_encoder_stack(ques_next_layer, ques_nhids_list, ques_kwidths_list,
                                                      {'src': self.params["embedding_dropout_keep_prob"],
                                                       'hid': self.params["nhid_dropout_keep_prob"]},
                                                      mode=self.is_training)

            ques_next_layer = linear_mapping_weightnorm(ques_next_layer, self.params["input_ques_dim"],
                                                             var_scope_name="linear_mapping_after_cnn")

            ## The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
            ## cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))
            self.ques_final_output = ques_next_layer
            self.ques_value_output = (ques_next_layer + self.ques_vecs) * tf.sqrt(0.5)

            ques_mask = tf.sequence_mask(lengths=self.ques_len, maxlen=self.params["max_n_q_words"], dtype=tf.float32)
            ques_value_mask = tf.multiply(self.ques_value_output, tf.expand_dims(ques_mask, 2))
            self.ques_final_state = tf.reduce_mean(ques_value_mask, 1)

        self.frame_final_state = linear_mapping(self.frame_final_state, self.input_ans_dim,
                                                     var_scope_name="linear_mapping_frame_final_state")
        self.frame_value_output = linear_mapping_weightnorm(self.frame_value_output, self.input_ans_dim,
                                                     var_scope_name="linear_mapping_frame_value_output")
        self.frame_final_output = linear_mapping_weightnorm(self.frame_final_output, self.input_ans_dim,
                                                    var_scope_name="linear_mapping_frame_final_output")

        self.encoder_output = EncoderOutput(
            frame_final_output=self.frame_final_output,
            frame_value_output=self.frame_value_output,
            frame_final_state=self.frame_final_state,
            frame_length=self.frame_len,
            ques_final_output=self.ques_final_output,
            ques_value_output=self.ques_value_output,
            ques_final_state=self.ques_final_state,
            ques_length=self.ques_len
        )




    def build_decoder(self):

        self.target = tf.placeholder(tf.int32, [None, self.max_n_a_words])
        self.target_mask = tf.placeholder(tf.float32, [None, self.max_n_a_words])
        self.answer_vecs = tf.placeholder(tf.float32, [None, self.max_n_a_words, self.input_ans_dim], name="ans_vecs")
        self.reward = tf.placeholder(tf.float32, [None])



        if self.params["position_embeddings_enable"]:
            ans_positions_embed = tf.tile([self.decoder_pos_embed], [self.batch_size, 1, 1])
            self.ans_vecs = tf.add(self.answer_vecs, ans_positions_embed)

        # Apply dropout to embeddings
        self.ans_vecs = tf.contrib.layers.dropout(
            inputs=self.ans_vecs,
            keep_prob=self.params["embedding_dropout_keep_prob"],
            is_training=self.is_training
        )

        # start_vecs = linear_mapping(self.ques_final_state, self.input_ans_dim,
        #                                        dropout=self.params["embedding_dropout_keep_prob"],
        #                                        var_scope_name="linear_mapping_start_vecs")
        self.start_vecs = tf.expand_dims(self.encoder_output.ques_final_state,1)
        self.ans_vecs = tf.concat([self.start_vecs, self.ans_vecs[:,:-1,:]], axis=1)
        assert self.ans_vecs.get_shape().as_list()[1] == self.max_n_a_words

        with tf.variable_scope("decoder_cnn") as scope:
            answer_train, train_loss = self.build_train_decoder()
            scope.reuse_variables()
            answer_test, test_loss = self.build_test_decoder()

        # final
        variables = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in variables ])
        self.answer_word_train = answer_train
        self.train_loss = train_loss + self.regularization_beta * regularization_cost

        self.answer_word_test = answer_test
        self.test_loss = test_loss  + self.regularization_beta * regularization_cost
        tf.summary.scalar('training cross entropy', self.train_loss)

    def build_train_decoder(self):

        decoder_next_layer = self.ans_vecs

        decoder_nhids_list = parse_list_or_default(self.params["decoder_nhids"], self.params["decoder_layers"],
                                                       self.params["decoder_nhid_default"])
        decoder_kwidths_list = parse_list_or_default(self.params["decoder_kwidths"], self.params["decoder_layers"],
                                                         self.params["decoder_kwidth_default"])

        # mapping emb dim to hid dim
        decoder_next_layer = linear_mapping_weightnorm(decoder_next_layer, decoder_nhids_list[0],
                                                           dropout=self.params["embedding_dropout_keep_prob"],
                                                           var_scope_name="linear_mapping_before_cnn")

        decoder_next_layer = conv_decoder_stack(self.ans_vecs, self.encoder_output, decoder_next_layer,
                                                    decoder_nhids_list, decoder_kwidths_list,
                                                    {'src': self.params["embedding_dropout_keep_prob"],
                                                     'hid': self.params["nhid_dropout_keep_prob"]},
                                                    mode=self.is_training)

        # argmax

        decoder_next_layer = linear_mapping_weightnorm(decoder_next_layer, self.params["nout_embed"],
                                                           var_scope_name="linear_mapping_after_cnn")

        # decoder_next_layer = linear_mapping_weightnorm(decoder_next_layer[:, -1:, :], self.params["nout_embed"],
        #                                       var_scope_name="linear_mapping_after_cnn")

        decoder_next_layer = tf.contrib.layers.dropout(
                inputs=decoder_next_layer,
                keep_prob=self.params["out_dropout_keep_prob"],
                is_training=self.is_training)

        logits = linear_mapping_weightnorm(decoder_next_layer, self.vocab_size, in_dim=self.params["nout_embed"],
                                               dropout=self.params["out_dropout_keep_prob"],
                                               var_scope_name="logits_before_softmax")


        max_prob_words = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        logit_list = tf.unstack(logits, num=self.max_n_a_words, axis=1)
        max_prob_word_list = tf.unstack(max_prob_words, num=self.max_n_a_words, axis=1)

        answer_train = list()
        loss = 0.0
        for i in range(self.max_n_a_words):
            answer_train.append(max_prob_word_list[i])

            # ground truth
            labels = tf.expand_dims(self.target[:, i], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab_size]), 1.0, 0.0)


            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_list[i])
            # cross_entropy = cross_entropy * self.reward
            cross_entropy = cross_entropy * self.target_mask[:, i]
            current_loss = tf.reduce_sum(cross_entropy)
            loss = loss + current_loss

        loss = loss / tf.reduce_sum(self.target_mask[:, 1:])
        return answer_train, loss


    def build_test_decoder(self):

        decoder_next_layer = self.ans_vecs

        decoder_nhids_list = parse_list_or_default(self.params["decoder_nhids"], self.params["decoder_layers"],
                                                   self.params["decoder_nhid_default"])
        decoder_kwidths_list = parse_list_or_default(self.params["decoder_kwidths"], self.params["decoder_layers"],
                                                     self.params["decoder_kwidth_default"])

        # mapping emb dim to hid dim
        decoder_next_layer = linear_mapping_weightnorm(decoder_next_layer, decoder_nhids_list[0],
                                                       dropout=self.params["embedding_dropout_keep_prob"],
                                                       var_scope_name="linear_mapping_before_cnn")

        decoder_next_layer = conv_decoder_stack(self.ans_vecs, self.encoder_output, decoder_next_layer,
                                                decoder_nhids_list, decoder_kwidths_list,
                                                {'src': self.params["embedding_dropout_keep_prob"],
                                                 'hid': self.params["nhid_dropout_keep_prob"]},
                                                mode=self.is_training)

        # argmax

        decoder_next_layer = linear_mapping_weightnorm(decoder_next_layer[:, :, :], self.params["nout_embed"],
                                                       var_scope_name="linear_mapping_after_cnn")

        decoder_next_layer = tf.contrib.layers.dropout(
            inputs=decoder_next_layer,
            keep_prob=self.params["out_dropout_keep_prob"],
            is_training=self.is_training)

        logits = linear_mapping_weightnorm(decoder_next_layer, self.vocab_size, in_dim=self.params["nout_embed"],
                                           dropout=self.params["out_dropout_keep_prob"],
                                           var_scope_name="logits_before_softmax")

        top_prods, top_prod_idxs, = tf.nn.top_k(logits, k=self.beam_width)


        answer_test = [top_prods, top_prod_idxs]

        loss = 0.0

        return answer_test, loss
