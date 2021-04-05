from collections import defaultdict
import tensorflow as tf
from utils.inits import *
import numpy as np
import math
from tqdm import tqdm

# from tensorflow.keras.activations import relu
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Masking, Dense, Input, LSTM, GRU, Conv1D, MaxPool1D, AveragePooling1D
#
from tensorflow.contrib.keras.python.keras.activations import relu
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Masking, Dense, Input, LSTM, GRU, Conv1D, MaxPool1D, AveragePooling1D


# from tensorflow.contrib.cudnn_rnn import CuDNNLSTM, CuDNNGRU

def gather_cols(params, indices, name=None, numpy=False):
    """Gather columns of a 2D tensor.

	Args:
		params: A 2D tensor.
		indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
		name: A name for the operation (optional).

	Returns:
		A 2D Tensor. Has the same type as ``params``.
	"""

    if not numpy:
        with tf.op_scope([params, indices], name, "gather_cols") as scope:
            # Check input
            params = tf.convert_to_tensor(params, name="params")
            indices = tf.convert_to_tensor(indices, name="indices")
            try:
                params.get_shape().assert_has_rank(2)
            except ValueError:
                raise ValueError('\'params\' must be 2D.')
            try:
                indices.get_shape().assert_has_rank(1)
            except ValueError:
                raise ValueError('\'params\' must be 1D.')

            # Define op
            p_shape = tf.shape(params)
            p_flat = tf.reshape(params, [-1])
            i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                           [-1, 1]) + indices, [-1])

            answer = tf.reshape(tf.gather(p_flat, i_flat), [p_shape[0], -1])

            # j = tf.Print(answer, [tf.shape(answer)])

            return answer

    if numpy:
        p_shape = params.shape
        p_flat = np.reshape(params, (-1))
        i_flat = np.reshape(np.reshape(np.arange(0, p_shape[0]) * p_shape[1], [-1,1]) + indices, [-1])
        a = np.reshape(np.take(p_flat, i_flat), [p_shape[0], -1])
        return a

class Model(object):
    def __init__(self, **kwargs):

        # TODO uncomment this
        # allowed_kwargs = {'name', 'logging'}
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        #
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')

        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.name: var for var in variables})

    def complex_formulation_numpy(self, src_r, src_i, rel_r, rel_i, tar_r, tar_i):

        p1 = np.sum(rel_r * src_r * tar_r, axis=-1)
        p2 = np.sum(rel_r * src_i * tar_i, axis=-1)
        p3 = np.sum(rel_i * src_r * tar_i, axis=-1)
        p4 = np.sum(rel_i * src_i * tar_r, axis=-1)

        phi = p1 + p2 + p3 - p4

        return phi

    def score(self, head, rel, tail, predict_tails=True):

        if self.config.decoder == 'complex':

            # self.node_all_r = self.placeholders['node_all_r']
            # self.node_all_i = self.placeholders['node_all_i']

            if predict_tails:

                src_r = tf.gather(self.node_all_r, head)
                src_i = tf.gather(self.node_all_i, head)

                rel_r = tf.gather(self.node_all_r, rel)
                rel_i = tf.gather(self.node_all_i, rel)

                # src_r = self.node_all_r[head]
                # src_i = self.node_all_i[head]
                #
                # rel_r = self.node_all_r[rel]
                # rel_i = self.node_all_i[rel]

                return self.complex_formulation(src_r, src_i, rel_r, rel_i, self.node_all_r, self.node_all_i)


    def evaluate(self, node_all_r, node_all_i, sess, predict='tails'):

        # self.mrr_tfilt = tf.summary.scalar('lstm_continuous_loss', loss)

        self.node_all_r = node_all_r
        self.node_all_i = node_all_i

        # if self.config.decoder == 'complex':
        #     self.all_embeddings_r = self.placeholders['all_embeddings_r']
        #     self.all_embeddings_i = self.placeholders['all_embeddings_i']

        for triples in tqdm(self.dataset.batch_generator_for_eval()):

            heads, rels, tails = triples[:, 0], triples[:, 1], triples[:, 2]

            scores = np.zeros((len(triples), self.config.n_nodes_total), dtype=np.float32)

            for index, (head, rel, tail) in enumerate(triples):

                print(index)

                score = self.score(head, rel, tail, predict_tails=True)

                print(sess.run(score))
                # scores[index] = score
                #
                # scores[tail_labels] = -np.inf


    # def evaluate(self):

class CombinationModel(Model):
    def __init__(self, config, placeholders, **kwargs):
        super(CombinationModel, self).__init__(**kwargs)
        self.config = config

        self.placeholders = placeholders

        self.pos_samples = placeholders['batch']
        self.neg_samples = placeholders['negative_samples']
        self.dropout = placeholders['dropout']
        self.ss_trainable = self.config.ss_trainable
        self.dataset = kwargs['dataset']

        if not config.featureless:
            self.features = placeholders['features']

        if self.config.pretrained_node is None:
            self.node_embeddings = tf.get_variable(
                shape=[config.n_nodes, config.embedding_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=config.seed),
                name='entity_embeddings',)
        else:
            self.node_embeddings = tf.Variable(np.load(self.config.pretrained_node, allow_pickle=True),
                                               trainable=self.ss_trainable,
                                               name='entity_embeddings')


        if self.config.pretrained_rel == None:
            self.relation_embeddings = tf.get_variable(
                shape=[config.n_relations, config.embedding_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=config.seed),
                name='relation_embeddings',)
        else:
            self.relation_embeddings = tf.Variable(np.load(self.config.pretrained_rel, allow_pickle=True),
                                                   trainable=self.ss_trainable,
                                                   name='relation_embeddings')

        self.projection_embeddings = tf.zeros(shape=[1])
        self.zeroshot_node_embeddings = tf.zeros(shape=[1])

        if config.zeroshot:
            self.projection_matrix = weight_variable_glorot(config.features_dim, config.embedding_dim,
                                                            name='projection_matrix')
            self.projection_bias = weight_variable_glorot(1, config.embedding_dim, name='projection_bias')
            self.features_unseen_nodes = placeholders['features_unseen_nodes']

        if self.config.reduce_func == "mean":
            self.reduce_func = tf.reduce_mean
        else:
            self.reduce_func = tf.reduce_sum

        if self.config.project_batchwise:
            self.nodes_to_project = placeholders['nodes_to_project']

        if not self.config.use_static_features:

            self.vocab_size = kwargs['vocab_size']
            self.word_embeddings_dimension = kwargs['word_embeddings_dimension']

            self.word_embeddings_tf = tf.Variable(tf.constant(0.0, shape=(self.vocab_size, self.word_embeddings_dimension))
                                                  , dtype=tf.float32, trainable=True, name='word_embeddings')

            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.word_embeddings_dimension])
            self.word_embeddings_init = self.word_embeddings_tf.assign(self.word_embeddings_placeholder)

            # self.word_embeddings_tf = kwargs['word_embeddings_tf']
            self.entityid2tokenids = kwargs['entityid2tokenids']
            self.node_dict = kwargs['node_dict']
            self.node_dict_zeroshot = kwargs['node_dict_zeroshot']
            self.n_nodes_total = len(self.entityid2tokenids)


            self.entityid2len_tokens = np.array([len(a) for a in self.entityid2tokenids])

            self.entityid2tokenids = self.normalise_tokens(self.entityid2tokenids)


            self.masks = np.zeros_like(self.entityid2tokenids, dtype=np.float32)

            self.entityid2tokenids_tf = tf.constant(self.entityid2tokenids)

            for i in range(self.n_nodes_total):
                self.masks[i, :self.entityid2len_tokens[i]] = 1

            self.masks_tf = tf.constant(self.masks)

        self.build()
        self.optimizer()

    @staticmethod
    def normalise_tokens(entityid2tokenids):

        longest = max([len(x) for x in entityid2tokenids])

        for i in range(len(entityid2tokenids)):
            while len(entityid2tokenids[i]) < longest:
                entityid2tokenids[i].append(0)

        tokens_new = np.array(entityid2tokenids, dtype=np.int32)

        return tokens_new

    def project(self, features):

        projected_features = None

        if self.config.project == 'affine':
            projected_features = tf.matmul(features, self.projection_matrix) + self.projection_bias

        if self.config.project == 'linear':
            projected_features = tf.matmul(features, self.projection_matrix)

        return projected_features

    def _build(self):
        if self.config.normalize_embed:
            self.node_embeddings = tf.nn.l2_normalize(self.node_embeddings, dim=1)
            self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, dim=1)

        if self.config.zeroshot:
            # TODO: add non-linearity to projection embeddings
            # projection embeddings given by: XW
            # self.projection_embeddings = tf.matmul(self.features, self.projection_matrix)

            if self.config.use_static_features:
                if not self.config.project_batchwise:
                    self.projection_embeddings = self.project(self.features)

                    print('\n\nAAAAH\n\n', self.projection_embeddings)

                    self.zeroshot_node_embeddings = self.project(self.features_unseen_nodes)

                else:
                    self.zeroshot_node_embeddings = None
                    self.projection_embeddings = None

                    simply_gathered = tf.gather(self.features, self.nodes_to_project)
                    self.simply_projected = self.project(simply_gathered)

                    if self.config.normalize_embed:
                        self.simply_projected = tf.nn.l2_normalize(self.simply_projected, dim=1)

            else:
                # TODO make this a function

                if not self.config.project_batchwise:
                    unseen_node_embeddings = self.gather_and_aggregate_words(np.array(sorted(list(self.node_dict_zeroshot.values()))))
                    # PROJECTED UNSEEN NODE EMBEDDINGS
                    self.zeroshot_node_embeddings = self.project(unseen_node_embeddings)

                    seen_nodes = np.array(sorted(list(self.node_dict.values())))

                    seen_node_embeddings = self.gather_and_aggregate_words(seen_nodes)
                    # PROJECTED SEEN EMBEDDINGS
                    self.projection_embeddings = self.project(self.gather_and_aggregate_words(seen_nodes))


                else:
                    self.zeroshot_node_embeddings = None
                    self.projection_embeddings = None

                    self.simply_gathered = self.gather_and_aggregate_words(self.nodes_to_project)
                    self.simply_projected = self.project(self.simply_gathered)

                    if self.config.normalize_embed:
                        self.simply_projected = tf.nn.l2_normalize(self.simply_projected, dim=1)

        if self.config.normalize_embed:
            if not self.config.project_batchwise:
                self.projection_embeddings = tf.nn.l2_normalize(self.projection_embeddings, dim=1)

                if self.config.zeroshot:
                    self.zeroshot_node_embeddings = tf.nn.l2_normalize(self.zeroshot_node_embeddings, dim=1)

                #TODO normalise batch wise also

        self.pos_ss, self.pos_sd, self.pos_ds, self.pos_dd = self.predict(self.pos_samples, pos=True)

        self.pos_src_struc_embed = self.src_struc_embed
        self.pos_src_desc_embed = self.src_desc_embed

        #======================================================

        self.pos_tar_struc_embed = self.tar_struc_embed
        self.pos_tar_desc_embed = self.tar_desc_embed

        self.neg_ss, self.neg_sd, self.neg_ds, self.neg_dd = self.predict(self.neg_samples)

    # def lstm(self):

    def sum(self, node_token_embeddings, node_masks):

        masked = node_token_embeddings * tf.expand_dims(node_masks, 2)

        reduced = tf.reduce_sum(masked, axis=1)

        return reduced

    def average(self, node_token_embeddings, node_masks):

        masked = node_token_embeddings * tf.expand_dims(node_masks, 2)

        reduced = tf.reduce_sum(masked, axis=1)

        n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

        # n_tokens_print = tf.Print(n_tokens, [tf.shape(tf.gather(node_masks, [0], axis=0))])

        averaged = reduced / n_tokens

        return averaged

    def lstm(self, node_token_embeddings, node_masks):

        node_token_embeddings = tf.zeros((self.config.batch_size, self.config.embedding_dim))

        node_token_embeddings = tf.Print(node_token_embeddings, ['node token embeddings', node_token_embeddings, 'shape', tf.shape(node_token_embeddings)], summarize=100)

        with tf.variable_scope('LSTM_encodeing'):
            print('cool')

        return node_token_embeddings

    def aggregate(self, node_token_embeddings, node_masks):

        if self.config.aggregation_function == 'average':
            return self.average(node_token_embeddings, node_masks)

        if self.config.aggregation_function == 'sum':
            return self.sum(node_token_embeddings, node_masks)

        if self.config.aggregation_function == 'lstm':
            return self.lstm(node_token_embeddings, node_masks)

        # if self.config.aggregation_function == 'sum':
        #     return self.average(node_token_embeddings, node_masks)



    def gather_and_aggregate_words(self, indices):
        # node_tokens = self.entityid2tokenids_tf[indices]
        # node_masks = self.masks_tf[indices]

        node_tokens = tf.gather(self.entityid2tokenids_tf, indices, axis=0)
        node_masks = tf.gather(self.masks_tf, indices, axis=0)

        # node_tokens_print = tf.Print(node_tokens, ['NODE TOKENS',node_tokens, tf.shape(node_tokens)])
        # node_masks_print = tf.Print(node_masks, ['NODE MASKS',node_masks, tf.shape(node_masks)])
        #
        # node_tokens = node_tokens_print
        # node_masks = node_masks_print

        node_token_embeddings = tf.gather(self.word_embeddings_tf, node_tokens, axis=0)

        node_embeddings = self.aggregate(node_token_embeddings, node_masks)

        # node_embeddings = self.project(node_embeddings)

        return node_embeddings

    def predict(self, inputs, pos=False):
        src_indices = tf.squeeze(gather_cols(inputs, [0]))
        rel_indices = tf.squeeze(gather_cols(inputs, [1]))
        tar_indices = tf.squeeze(gather_cols(inputs, [2]))

        self.src_struc_embed = tf.gather(self.node_embeddings, src_indices)
        self.tar_struc_embed = tf.gather(self.node_embeddings, tar_indices)
        rel_struc_embed = tf.gather(self.relation_embeddings, rel_indices)

        preds_ss = None
        preds_sd = None
        preds_ds = None
        preds_dd = None

        self.src_desc_embed = None
        self.tar_desc_embed = None

        if pos:
            self.src_batch_features = tf.gather(self.features, src_indices)
            self.src_indices = src_indices

        if self.config.zeroshot:
            if self.config.use_static_features:
                if not self.config.project_batchwise:
                    self.src_desc_embed = tf.gather(self.projection_embeddings, src_indices)
                    self.tar_desc_embed = tf.gather(self.projection_embeddings, tar_indices)
                else:
                    self.src_desc_embed = self.project(tf.gather(self.features, src_indices))
                    self.tar_desc_embed = self.project(tf.gather(self.features, tar_indices))
            else:
                #vkcomp
                self.src_desc_embed = self.project(self.gather_and_aggregate_words(src_indices))
                self.tar_desc_embed = self.project(self.gather_and_aggregate_words(tar_indices))

        # if self.config.decoder == 'distmult':
        #     if self.config.loss1:
        #         preds_ss = tf.reduce_sum(self.src_struc_embed * rel_struc_embed * self.tar_struc_embed, axis=1)
        #     if self.config.loss2:
        #         preds_dd = tf.reduce_sum(self.src_desc_embed * rel_struc_embed * self.tar_desc_embed, axis=1)
        #     if self.config.loss4:
        #         preds_sd = tf.reduce_sum(self.src_struc_embed * rel_struc_embed * self.tar_desc_embed, axis=1)
        #     if self.config.loss5:
        #         preds_ds = tf.reduce_sum(self.src_desc_embed * rel_struc_embed * self.tar_struc_embed, axis=1)


        # TODO: remove normalisation for description embeds
        if self.config.decoder == 'transe':

            if self.config.weird_transe:
                self.src_struc_embed = tf.nn.l2_normalize(self.src_struc_embed, 1)
                rel_struc_embed = tf.nn.l2_normalize(rel_struc_embed, 1)
                self.tar_struc_embed = tf.nn.l2_normalize(self.tar_struc_embed, 1)

                self.src_desc_embed = tf.nn.l2_normalize(self.src_desc_embed, 1)
                self.tar_desc_embed = tf.nn.l2_normalize(self.tar_desc_embed, 1)

            if self.config.transe_scoring == 'L1':

                if self.config.loss1:
                    preds_ss = tf.reduce_sum(tf.abs(self.src_struc_embed + rel_struc_embed - self.tar_struc_embed), axis=1)
                if self.config.loss2:
                    preds_dd = tf.reduce_sum(tf.abs(self.src_desc_embed + rel_struc_embed - self.tar_desc_embed), axis=1)
                if self.config.loss4:
                    preds_sd = tf.reduce_sum(tf.abs(self.src_struc_embed + rel_struc_embed - self.tar_desc_embed), axis=1)
                if self.config.loss5:
                    preds_ds = tf.reduce_sum(tf.abs(self.src_desc_embed + rel_struc_embed - self.tar_struc_embed), axis=1)

            else:
                if self.config.loss1:
                    preds_ss = tf.reduce_sum(tf.square(self.src_struc_embed + rel_struc_embed - self.tar_struc_embed), axis=1)
                if self.config.loss2:
                    preds_dd = tf.reduce_sum(tf.square(self.src_desc_embed + rel_struc_embed - self.tar_desc_embed), axis=1)
                if self.config.loss4:
                    preds_sd = tf.reduce_sum(tf.square(self.src_struc_embed + rel_struc_embed - self.tar_desc_embed), axis=1)
                if self.config.loss5:
                    preds_ds = tf.reduce_sum(tf.square(self.src_desc_embed + rel_struc_embed - self.tar_struc_embed), axis=1)

        return preds_ss, preds_sd, preds_ds, preds_dd

    def optimizer(self):
        self.loss_ss = tf.zeros(shape=[1])
        self.loss_dd = tf.zeros(shape=[1])
        self.loss_sd = tf.zeros(shape=[1])
        self.loss_ds = tf.zeros(shape=[1])
        self.projection_loss = tf.zeros(shape=[1])
        self.regularizer = tf.zeros(shape=[1])

        if self.config.decoder == 'distmult':
            self.cost = self.distmult_loss()

        elif self.config.decoder == 'transe':
            self.cost = self.transe_loss()

        # self.opt = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.opt_op = self.opt.minimize(self.cost)

    def distmult_loss(self):
        # structure-structure cross-entropy loss
        if self.config.loss1:
            self.loss_ss = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_ss), tf.zeros_like(self.neg_ss)], axis=0),
                logits=tf.concat([self.pos_ss, self.neg_ss], axis=0)
            ))

        # description-description cross-entropy loss
        if self.config.loss2:
            self.loss_dd = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_dd), tf.zeros_like(self.neg_dd)], axis=0),
                logits=tf.concat([self.pos_dd, self.neg_dd], axis=0)
            ))

        # structure-description cross-entropy loss
        if self.config.loss4:
            self.loss_sd = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_sd), tf.zeros_like(self.neg_sd)], axis=0),
                logits=tf.concat([self.pos_sd, self.neg_sd], axis=0)
            ))

        # description-structure cross-entropy loss
        if self.config.loss5:
            self.loss_ds = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_ds), tf.zeros_like(self.neg_ds)], axis=0),
                logits=tf.concat([self.pos_ds, self.neg_ds], axis=0)
            ))

        # projection l2-loss
        if self.config.loss3:
            self.projection_loss = self.reduce_func(
                tf.reduce_sum(tf.square(self.projection_embeddings - self.node_embeddings), axis=1)
            )

        if self.config.regularize:
            self.regularizer = tf.reduce_sum(tf.square(self.projection_matrix))


        loss = self.loss_ss + self.loss_dd + self.loss_ds + self.loss_sd + \
               self.config.projection_weight * self.projection_loss + \
               self.config.regularization_weight * self.regularizer
        return loss

    def regularize(self):

        if self.config.project == 'linear':
            return tf.reduce_sum(tf.square(self.projection_matrix))

        if self.config.project == 'affine':
            return tf.reduce_sum(tf.square(self.projection_matrix) + tf.square(self.projection_bias))

    def transe_loss(self):
        margin = self.config.max_margin

        # structure-structure max-margin loss
        if self.config.loss1:
            self.loss_ss = self.reduce_func(tf.nn.relu(self.pos_ss - self.neg_ss + margin))

        # description-description max-margin loss
        if self.config.loss2:
            self.loss_dd = self.reduce_func(tf.nn.relu(self.pos_dd - self.neg_dd + margin))

        # structure-description max-margin loss
        if self.config.loss4:
            self.loss_sd = self.reduce_func(tf.nn.relu(self.pos_sd - self.neg_sd + margin))

        # description-structure max-margin loss
        if self.config.loss5:
            self.loss_ds = self.reduce_func(tf.nn.relu(self.pos_ds - self.neg_ds + margin))

        # projection l2-loss
        if self.config.loss3:
            #CHANGED THIS

            #TODO make this sqrt
            if not self.config.project_batchwise:
                self.projection_loss = self.reduce_func(
                    tf.reduce_sum(tf.square(self.projection_embeddings - self.node_embeddings), axis=1)
                )
            else:
                # self.pos_src_desc_embed = tf.Print(self.pos_src_desc_embed, ['positive', tf.shape(self.pos_src_desc_embed)])
                # self.src_desc_embed = tf.Print(self.src_desc_embed, ['negative', tf.shape(self.src_desc_embed)])

                projection_loss_source = self.reduce_func(tf.sqrt(tf.reduce_sum(tf.square(self.pos_src_struc_embed - self.pos_src_desc_embed), axis=1)))
                # projection_loss_source += self.reduce_func(tf.reduce_sum(tf.square(self.src_struc_embed - self.src_desc_embed), axis=1))

                projection_loss_target = self.reduce_func(tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed - self.pos_tar_desc_embed), axis=1)))
                # projection_loss_target += self.reduce_func(tf.reduce_sum(tf.square(self.tar_struc_embed - self.tar_desc_embed), axis=1))

                self.projection_loss = (projection_loss_source + projection_loss_target) / 2

        if self.config.regularize:
            self.regularizer = self.regularize()

        loss = self.loss_ss + self.loss_dd + self.loss_ds + self.loss_sd + \
               self.config.projection_weight * self.projection_loss + \
               self.config.regularization_weight * self.regularizer
        return loss

##################################################################################################################################

class ComplexModel(Model):

    def __init__(self, config, placeholders, **kwargs):
        super(ComplexModel, self).__init__(**kwargs)
        self.config = config

        self.placeholders = placeholders

        self.pos_samples = placeholders['batch']
        self.neg_samples = placeholders['negative_samples']
        self.dropout = placeholders['dropout']
        self.ss_trainable = self.config.ss_trainable
        self.dataset = kwargs['dataset']

        if not config.featureless:
            self.features = placeholders['features']

        if self.config.pretrained_node is None:
            self.node_embeddings_r = tf.get_variable(
                shape=[config.n_nodes, config.embedding_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=config.seed),
                name='entity_embeddings_real')

            self.node_embeddings_i = tf.get_variable(
                shape=[config.n_nodes, config.embedding_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=config.seed),
                name='entity_embeddings_imaginary')

        else:

            a = np.load(self.config.pretrained_node[:-2] + '_r.p', allow_pickle=True)


            if self.config.decoder == 'complex':
                self.node_embeddings_r = tf.Variable(np.load(self.config.pretrained_node[:-2] + '_r.p', allow_pickle=True), name='entity_embeddings_real', trainable=self.ss_trainable)
                self.node_embeddings_i = tf.Variable(np.load(self.config.pretrained_node[:-2] + '_i.p', allow_pickle=True), name='entity_embeddings_imaginary', trainable=self.ss_trainable)

            else:
                self.node_embeddings = tf.Variable(np.load(self.config.pretrained_node, allow_pickle=True), name='entity_embeddings')

        if self.config.pretrained_rel == None:
            self.relation_embeddings_r = tf.get_variable(
                shape=[config.n_relations, config.embedding_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=config.seed),
                name='relation_embeddings_real')

            self.relation_embeddings_i = tf.get_variable(
                shape=[config.n_relations, config.embedding_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=config.seed),
                name='relation_embeddings_imaginary')

        else:
            if self.config.decoder == 'complex':
                self.relation_embeddings_r = tf.Variable(np.load(self.config.pretrained_rel[:-2] + '_r.p', allow_pickle=True), name='relation_embeddings_real', trainable=self.ss_trainable)
                self.relation_embeddings_i = tf.Variable(np.load(self.config.pretrained_rel[:-2] + '_i.p', allow_pickle=True), name='relation_embeddings_imaginary', trainable=self.ss_trainable)

            else:
                self.relation_embeddings = tf.Variable(np.load(self.config.pretrained_rel, allow_pickle=True), name='relation_embeddings')

        # self.node_embeddings_r = tf.Print(self.node_embeddings_r, ['NODE_EMBEDZ', self.node_embeddings_r])

        self.projection_embeddings_r = tf.zeros(shape=[1])
        self.projection_embeddings_i = tf.zeros(shape=[1])

        self.zeroshot_node_embeddings_r = tf.zeros(shape=[1])
        self.zeroshot_node_embeddings_i = tf.zeros(shape=[1])

        if config.zeroshot:
            self.projection_matrix_r = weight_variable_glorot(config.features_dim, config.embedding_dim,
                                                            name='projection_matrix_real')
            self.projection_bias_r = weight_variable_glorot(1, config.embedding_dim, name='projection_bias_real')

            # self.projection_matrix_r = tf.Variable(tf.ones([config.features_dim, config.embedding_dim]), trainable=True)
            #
            # self.projection_bias_r = tf.Variable(tf.ones([1, config.embedding_dim]), trainable=True)
            #
            # self.projection_matrix_i = tf.Variable(tf.ones([config.features_dim, config.embedding_dim]), trainable=True)
            #
            # self.projection_bias_i = tf.Variable(tf.ones([1, config.embedding_dim]), trainable=True)

            self.projection_matrix_i = weight_variable_glorot(config.features_dim, config.embedding_dim,
                                                              name='projection_matrix_imaginary')

            self.projection_bias_i = weight_variable_glorot(1, config.embedding_dim, name='projection_bias_imaginary')

            #vk changed this
            # self.projection_matrix_i = self.projection_matrix_r
            # self.projection_bias_i = self.projection_bias_r

            self.features_unseen_nodes = placeholders['features_unseen_nodes']

        if self.config.reduce_func == "mean":
            self.reduce_func = tf.reduce_mean
        else:
            self.reduce_func = tf.reduce_sum

        if self.config.project_batchwise:
            self.nodes_to_project = placeholders['nodes_to_project']

        if not self.config.use_static_features:

            self.vocab_size = kwargs['vocab_size']
            self.word_embeddings_dimension = kwargs['word_embeddings_dimension']

            self.word_embeddings_tf = tf.Variable(
                tf.constant(0.0, shape=(self.vocab_size, self.word_embeddings_dimension))
                , dtype=tf.float32, trainable=True, name='word_embeddings')

            self.word_embeddings_placeholder = tf.placeholder(tf.float32,
                                                              [self.vocab_size, self.word_embeddings_dimension])
            self.word_embeddings_init = self.word_embeddings_tf.assign(self.word_embeddings_placeholder)

            # self.word_embeddings_tf = kwargs['word_embeddings_tf']
            self.entityid2tokenids = kwargs['entityid2tokenids']
            self.node_dict = kwargs['node_dict']
            self.node_dict_zeroshot = kwargs['node_dict_zeroshot']
            self.n_nodes_total = len(self.entityid2tokenids)

            self.entityid2len_tokens = np.array([len(a) for a in self.entityid2tokenids])

            self.entityid2tokenids = self.normalise_tokens(self.entityid2tokenids)


            self.masks = np.zeros_like(self.entityid2tokenids, dtype=np.float32)

            self.entityid2tokenids_tf = tf.constant(self.entityid2tokenids)

            for i in range(self.n_nodes_total):
                self.masks[i, :self.entityid2len_tokens[i]] = 1

            self.masks_tf = tf.constant(self.masks)

        self.build()
        self.optimizer()

    def normalise_tokens(self, entityid2tokenids):

        longest = max([len(x) for x in entityid2tokenids])

        self.max_seq_length = longest

        for i in range(len(entityid2tokenids)):
            while len(entityid2tokenids[i]) < longest:
                entityid2tokenids[i].append(0)

        tokens_new = np.array(entityid2tokenids, dtype=np.int32)

        return tokens_new

    def project(self, features_r, features_i):

        projected_features_r, projected_features_i = None, None

        if self.config.project == 'affine':
            projected_features_r = tf.matmul(features_r, self.projection_matrix_r) + self.projection_bias_r
            projected_features_i = tf.matmul(features_i, self.projection_matrix_i) + self.projection_bias_i

        if self.config.project == 'linear':
            projected_features_r = tf.matmul(features_r, self.projection_matrix_r)
            projected_features_i = tf.matmul(features_i, self.projection_matrix_i)

        return projected_features_r, projected_features_i

    def _build(self):
        if self.config.normalize_embed:
            self.node_embeddings_r = tf.nn.l2_normalize(self.node_embeddings_r, dim=1)
            self.node_embeddings_i = tf.nn.l2_normalize(self.node_embeddings_i, dim=1)

            self.relation_embeddings_r = tf.nn.l2_normalize(self.relation_embeddings_r, dim=1)
            self.relation_embeddings_i = tf.nn.l2_normalize(self.relation_embeddings_i, dim=1)

        if self.config.zeroshot:
            # TODO: add non-linearity to projection embeddings
            # projection embeddings given by: XW
            # self.projection_embeddings = tf.matmul(self.features, self.projection_matrix)

            if self.config.use_static_features:

                if not self.config.project_batchwise:
                    print(self.features)
                    self.projection_embeddings_r, self.projection_embeddings_i = self.project(self.features, self.features)
                    print(self.projection_embeddings_r)
                    print(self.projection_matrix_i)
                    self.zeroshot_node_embeddings_r, self.zeroshot_node_embeddings_i = self.project(self.features_unseen_nodes, self.features_unseen_nodes)

                else:
                    self.zeroshot_node_embeddings = None
                    self.projection_embeddings = None

                    simply_gathered = tf.gather(self.features, self.nodes_to_project)
                    self.simply_projected_r, self.simply_projected_i = self.project(simply_gathered, simply_gathered)

                    if self.config.normalize_embed:
                        self.simply_projected_r = tf.nn.l2_normalize(self.simply_projected_r, dim=1)
                        self.simply_projected_i = tf.nn.l2_normalize(self.simply_projected_i, dim=1)

            else:
                # TODO make this a function

                if not self.config.project_batchwise:
                    unseen_node_embeddings = self.gather_and_aggregate_words(np.array(sorted(list(self.node_dict_zeroshot.values()))))
                    # PROJECTED UNSEEN NODE EMBEDDINGS
                    self.zeroshot_node_embeddings_r, self.zeroshot_node_embeddings_i = self.project(unseen_node_embeddings, unseen_node_embeddings)

                    seen_nodes = np.array(sorted(list(self.node_dict.values())))

                    seen_node_embeddings = self.gather_and_aggregate_words(seen_nodes)

                    # PROJECTED SEEN EMBEDDINGS
                    self.projection_embeddings_r, self.projection_embeddings_i = self.project(seen_node_embeddings, seen_node_embeddings)

                else:
                    self.zeroshot_node_embeddings = None
                    self.projection_embeddings = None

                    self.simply_gathered = self.gather_and_aggregate_words(self.nodes_to_project, train=False)
                    # self.simply_gathered = tf.Print(self.simply_gathered, ['jj', tf.shape(self.simply_gathered)])
                    self.simply_projected_r, self.simply_projected_i = self.project(self.simply_gathered, self.simply_gathered)

                    if self.config.normalize_embed:
                        self.simply_projected_r = tf.nn.l2_normalize(self.simply_projected_r, dim=1)
                        self.simply_projected_i = tf.nn.l2_normalize(self.simply_projected_i, dim=1)

        if self.config.normalize_embed:

            if not self.config.project_batchwise:
                self.projection_embeddings_r = tf.nn.l2_normalize(self.projection_embeddings_r, dim=1)
                self.projection_embeddings_i = tf.nn.l2_normalize(self.projection_embeddings_i, dim=1)

                if self.config.zeroshot:
                    self.zeroshot_node_embeddings_r = tf.nn.l2_normalize(self.zeroshot_node_embeddings_r, dim=1)
                    self.zeroshot_node_embeddings_i = tf.nn.l2_normalize(self.zeroshot_node_embeddings_i, dim=1)

                #TODO normalise batch wise also

        self.pos_ss, self.pos_sd, self.pos_ds, self.pos_dd = self.predict(self.pos_samples, pos=True)

        self.pos_src_struc_embed_r = self.src_struc_embed_r
        self.pos_src_struc_embed_i = self.src_struc_embed_i

        self.pos_tar_struc_embed_r = self.tar_struc_embed_r
        self.pos_tar_struc_embed_i = self.tar_struc_embed_i

        #================================================================================

        self.pos_src_desc_embed_r = self.src_desc_embed_r
        self.pos_src_desc_embed_i = self.src_desc_embed_i

        self.pos_tar_desc_embed_r = self.tar_desc_embed_r
        self.pos_tar_desc_embed_i = self.tar_desc_embed_i

        self.neg_ss, self.neg_sd, self.neg_ds, self.neg_dd = self.predict(self.neg_samples)

    # def lstm(self):

    def sum(self, node_token_embeddings, node_masks, divide_by=1):

        masked = node_token_embeddings * tf.expand_dims(node_masks, 2)

        reduced = tf.reduce_sum(masked, axis=1)

        return reduced / divide_by

    def average(self, node_token_embeddings, node_masks):

        masked = node_token_embeddings * tf.expand_dims(node_masks, 2)

        reduced = tf.reduce_sum(masked, axis=1)

        n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

        # n_tokens_print = tf.Print(n_tokens, [tf.shape(tf.gather(node_masks, [0], axis=0))])

        averaged = reduced / n_tokens

        return averaged

    def lstm(self, node_token_embeddings, node_masks):

        # answer = tf.zeros((self.config.batch_size, self.config.embedding_dim))
        #
        # answer = tf.Print(answer, ['node token embeddings', node_token_embeddings, 'shape', tf.shape(node_token_embeddings)], summarize=100)

        print('once')

        node_token_embeddings = tf.reshape(node_token_embeddings, (self.config.batch_size, self.max_seq_length, self.config.embedding_dim))

        # with tf.variable_scope('LSTM_encoding', reuse=True):

        embs_masked = Masking()(node_token_embeddings)

        lstm_out = LSTM(310, activation='relu')(embs_masked)

        lstm_out = Dense(300)(lstm_out)

        return lstm_out

    def cnn(self, node_token_embeddings, node_masks):

        node_token_embeddings = tf.reshape(node_token_embeddings,
                                           (self.config.batch_size, self.max_seq_length, self.config.embedding_dim))

        out1 = Conv1D(300, 3, 1, padding='same', activation=relu)(node_token_embeddings)
        out2 = MaxPool1D(padding='same')(out1)
        out3 = Conv1D(300, 2, 1, padding='same', activation=relu)(out2)
        out4 = MaxPool1D(padding='same')(out3)
        out5 = Conv1D(300, 3, 2, padding='same', activation=relu)(out4)
        out6 = MaxPool1D(4, padding='same')(out5)

        return tf.squeeze(out6)

    def aggregate(self, node_token_embeddings, node_masks, train):

        if self.config.toggle:
            if train:
                return self.sum(node_token_embeddings, node_masks, divide_by=6.19)

            else:
                return self.average(node_token_embeddings, node_masks)

        else:
            if self.config.aggregation_function == 'average':
                return self.average(node_token_embeddings, node_masks)

            if self.config.aggregation_function == 'sum':
                return self.sum(node_token_embeddings, node_masks)

            if self.config.aggregation_function == 'lstm':
                return self.lstm( node_token_embeddings, node_masks)

            if self.config.aggregation_function == 'cnn':
                return self.cnn( node_token_embeddings, node_masks)

    def gather_and_aggregate_words(self, indices, train=True):
        # node_tokens = self.entityid2tokenids_tf[indices]
        # node_masks = self.masks_tf[indices]

        node_tokens = tf.gather(self.entityid2tokenids_tf, indices, axis=0)
        node_masks = tf.gather(self.masks_tf, indices, axis=0)

        # node_tokens_print = tf.Print(node_tokens, ['NODE TOKENS',node_tokens, tf.shape(node_tokens)])
        # node_masks_print = tf.Print(node_masks, ['NODE MASKS',node_masks, tf.shape(node_masks)])
        #
        # node_tokens = node_tokens_print
        # node_masks = node_masks_print

        node_token_embeddings = tf.gather(self.word_embeddings_tf, node_tokens, axis=0)

        node_embeddings = self.aggregate(node_token_embeddings, node_masks, train)

        # node_embeddings_r, node_embeddings_i = self.project(node_embeddings, node_embeddings)

        return node_embeddings

    def predict(self, inputs, pos=False):
        src_indices = tf.squeeze(gather_cols(inputs, [0]))
        rel_indices = tf.squeeze(gather_cols(inputs, [1]))
        tar_indices = tf.squeeze(gather_cols(inputs, [2]))


        # src_indices = tf.Print(src_indices, ['gg', src_indices])
        # src_indices = tf.Print(src_indices, ['hh', src_indices[147]])

        self.src_struc_embed_r = tf.nn.embedding_lookup(self.node_embeddings_r, src_indices)
        self.src_struc_embed_i = tf.nn.embedding_lookup(self.node_embeddings_i, src_indices)


        self.tar_struc_embed_r = tf.nn.embedding_lookup(self.node_embeddings_r, tar_indices)
        self.tar_struc_embed_i = tf.nn.embedding_lookup(self.node_embeddings_i, tar_indices)

        rel_struc_embed_r = tf.nn.embedding_lookup(self.relation_embeddings_r, rel_indices)
        rel_struc_embed_i = tf.nn.embedding_lookup(self.relation_embeddings_i, rel_indices)

        preds_ss = None
        preds_sd = None
        preds_ds = None
        preds_dd = None

        self.src_desc_embed_r = None
        self.src_desc_embed_i = None

        self.tar_desc_embed_r = None
        self.tar_desc_embed_i = None

        if pos:
            self.src_batch_features = tf.nn.embedding_lookup(self.features, src_indices)
            self.src_indices = src_indices

        if self.config.zeroshot:

            if self.config.use_static_features:
                if not self.config.project_batchwise:

                    self.src_desc_embed_r = tf.nn.embedding_lookup(self.projection_embeddings_r, src_indices)
                    self.src_desc_embed_i = tf.nn.embedding_lookup(self.projection_embeddings_i, src_indices)

                    self.tar_desc_embed_r = tf.nn.embedding_lookup(self.projection_embeddings_r, tar_indices)
                    self.tar_desc_embed_i = tf.nn.embedding_lookup(self.projection_embeddings_i, tar_indices)

                else:
                    src_feats = tf.nn.embedding_lookup(self.features, src_indices)
                    tar_feats = tf.nn.embedding_lookup(self.features, tar_indices)

                    self.src_desc_embed_r, self.src_desc_embed_i = self.project(src_feats, src_feats)
                    self.tar_desc_embed_r, self.tar_desc_embed_i = self.project(tar_feats, tar_feats)

            else:
                src_gathered = self.gather_and_aggregate_words(src_indices)
                self.src_desc_embed_r, self.src_desc_embed_i = self.project(src_gathered, src_gathered)

                tar_gathered = self.gather_and_aggregate_words(tar_indices)
                self.tar_desc_embed_r, self.tar_desc_embed_i = self.project(tar_gathered, tar_gathered)


        if self.config.decoder == 'complex':
            if self.config.loss1:
                preds_ss = self.complex_formulation(self.src_struc_embed_r, self.src_struc_embed_i, rel_struc_embed_r, rel_struc_embed_i,
                                                   self.tar_struc_embed_r, self.tar_struc_embed_i)

            if self.config.loss2:
                preds_dd = self.complex_formulation(self.src_desc_embed_r, self.src_desc_embed_i, rel_struc_embed_r, rel_struc_embed_i,
                                                   self.tar_desc_embed_r, self.tar_desc_embed_i)

            if self.config.loss4:
                preds_sd = self.complex_formulation(self.src_struc_embed_r, self.src_struc_embed_i, rel_struc_embed_r, rel_struc_embed_i,
                                                   self.tar_desc_embed_r, self.tar_desc_embed_i)

            if self.config.loss5:
                preds_ds = self.complex_formulation(self.src_desc_embed_r, self.src_desc_embed_i, rel_struc_embed_r, rel_struc_embed_i,
                                                   self.tar_struc_embed_r, self.tar_struc_embed_i)

        return preds_ss, preds_sd, preds_ds, preds_dd

    def complex_formulation(self, src_r, src_i, rel_r, rel_i, tar_r, tar_i):

        p1 = tf.reduce_sum(rel_r * src_r * tar_r, axis=-1)
        p2 = tf.reduce_sum(rel_r * src_i * tar_i, axis=-1)
        p3 = tf.reduce_sum(rel_i * src_r * tar_i, axis=-1)
        p4 = tf.reduce_sum(rel_i * src_i * tar_r, axis=-1)

        phi = p1 + p2 + p3 - p4

        return phi

    def optimizer(self):
        self.loss_ss = tf.zeros(shape=[1])
        self.loss_dd = tf.zeros(shape=[1])
        self.loss_sd = tf.zeros(shape=[1])
        self.loss_ds = tf.zeros(shape=[1])
        self.projection_loss = tf.zeros(shape=[1])
        self.regularizer = tf.zeros(shape=[1])

        if self.config.decoder == 'complex':
            self.cost = self.complex_loss()

        # self.cost = tf.Print(self.cost, ['cost: ', self.cost])
        # self.opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.opt = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate, initial_accumulator_value=1e-20)
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        self.opt_op = self.opt.minimize(self.cost)


    def complex_reduce(self, pos, neg):

        # coeffs = tf.concat([-tf.ones_like(pos), tf.ones_like(neg)], axis=0)
        # phis = tf.concat([pos, neg], axis=0)
        #
        # losses = tf.nn.softplus(coeffs * phis)
        #
        # print('cool')
        #
        # losses = tf.Print(losses, ['losses: ', losses], first_n=100)

        # pos = tf.Print(pos, ['pos', pos])
        # neg = tf.Print(neg, ['neg', neg])

        # loss = self.reduce_func(tf.nn.softplus(tf.concat([-pos, neg], axis=0)))
        loss = self.reduce_func(tf.nn.softplus(-pos) + tf.nn.softplus(neg))

        print(loss)

        g = tf.gradients(loss, self.src_struc_embed_r)[0]

        print(g)
        #
        # loss = tf.Print(loss, ['g', g])

        return loss

    def complex_loss(self):
        if self.config.loss1:
            self.loss_ss = self.complex_reduce(self.pos_ss, self.neg_ss)

        if self.config.loss2:
            self.loss_dd = self.complex_reduce(self.pos_dd, self.neg_dd)

        if self.config.loss4:
            self.loss_sd = self.complex_reduce(self.pos_sd, self.neg_sd)

        if self.config.loss5:
            self.loss_ds = self.complex_reduce(self.pos_ds, self.neg_ds)

        if self.config.loss3:

            if not self.config.project_batchwise:
                self.projection_loss = self.reduce_func(
                    tf.sqrt(tf.reduce_sum(tf.square(self.projection_embeddings_r - self.node_embeddings_r), axis=1))
                                  + tf.sqrt(tf.reduce_sum(tf.square(self.projection_embeddings_i - self.node_embeddings_i), axis=1)),

                )

            else:

                # self.pos_src_desc_embed_i = tf.Print(self.pos_src_desc_embed_i, ['positive projn shape', tf.shape(self.pos_src_desc_embed_i)])
                # self.src_desc_embed_i = tf.Print(self.src_desc_embed_i, ['negative projn shape', tf.shape(self.src_desc_embed_i)])

                # projection_loss_source = self.reduce_func(tf.reduce_sum(tf.square(self.src_struc_embed_r - self.src_desc_embed_r)
                #                                                         + tf.square(self.src_struc_embed_i - self.src_desc_embed_i),
                #                                                         axis=1))
                #
                # projection_loss_source += self.reduce_func(tf.reduce_sum(tf.square(self.pos_src_struc_embed_r - self.pos_src_desc_embed_r)
                #                                                         + tf.square(self.pos_src_struc_embed_i - self.pos_src_desc_embed_i),
                #                                                         axis=1))
                #
                # projection_loss_target = self.reduce_func(tf.reduce_sum(tf.square(self.tar_struc_embed_r - self.tar_desc_embed_r),
                #                                                         + tf.square(self.tar_struc_embed_i - self.tar_desc_embed_i),
                #                                                         axis=1))
                #
                # projection_loss_target += self.reduce_func(tf.reduce_sum(tf.square(self.pos_tar_struc_embed_r - self.pos_tar_desc_embed_r),
                #                                                          + tf.square(self.pos_tar_struc_embed_i - self.pos_tar_desc_embed_i),
                #                                                          axis=1))

                # projection_loss_source = self.reduce_func(
                #     tf.sqrt(tf.reduce_sum(tf.square(self.src_struc_embed_r - self.src_desc_embed_r), axis=1))
                #                   + tf.sqrt(tf.reduce_sum(tf.square(self.src_struc_embed_i - self.src_desc_embed_i),
                #                   axis=1)))

                #TODO include target projection also
                self.loss_pre = tf.sqrt(tf.reduce_sum(tf.square(self.pos_src_struc_embed_r - self.pos_src_desc_embed_r), axis=1))\
                                  + tf.sqrt(tf.reduce_sum(tf.square(self.pos_src_struc_embed_i - self.pos_src_desc_embed_i),
                                  axis=1))

                projection_loss_target = self.reduce_func(tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed_r - self.pos_tar_desc_embed_r), axis=1))\
                                  + tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed_i - self.pos_tar_desc_embed_i),
                                  axis=1)))

                projection_loss_source = self.reduce_func(self.loss_pre)

                # projection_loss_target = self.reduce_func(
                #     tf.sqrt(tf.reduce_sum(tf.square(self.tar_struc_embed_r - self.tar_desc_embed_r), axis=1))
                #                   + tf.sqrt(tf.reduce_sum(tf.square(self.tar_struc_embed_i - self.tar_desc_embed_i),
                #                   axis=1)))

                # projection_loss_target = self.reduce_func(
                #     tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed_r - self.pos_tar_desc_embed_r), axis=1))
                #                   + tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed_i - self.pos_tar_desc_embed_i),
                #                   axis=1)))

                # self.projection_loss = (projection_loss_source + projection_loss_target) / 2

                self.projection_loss = projection_loss_source + projection_loss_target
                # self.projection_loss = projection_loss_target

        # self.emb_regularizer = self.reduce_func(self.pos_src_struc_embed_r ** 2) + \
        #                        self.reduce_func(self.pos_src_struc_embed_i ** 2) + \
        #                        self.reduce_func(self.pos_tar_struc_embed_r ** 2) + \
        #                        self.reduce_func(self.pos_tar_struc_embed_i ** 2) + \
        #                        self.reduce_func(self.pos_src_struc_embed_r ** 2) + \
        #                        self.reduce_func(self.pos_src_struc_embed_r ** 2) + \

        if self.config.regularize:
            self.regularizer = self.regularize()

        loss = self.loss_ss + self.loss_dd + self.loss_ds + self.loss_sd + \
               self.config.projection_weight * self.projection_loss + \
               self.config.regularization_weight * self.regularizer

        return loss

    def distmult_loss(self):
        # structure-structure cross-entropy loss
        if self.config.loss1:
            self.loss_ss = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_ss), tf.zeros_like(self.neg_ss)], axis=0),
                logits=tf.concat([self.pos_ss, self.neg_ss], axis=0)
            ))

        # description-description cross-entropy loss
        if self.config.loss2:
            self.loss_dd = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_dd), tf.zeros_like(self.neg_dd)], axis=0),
                logits=tf.concat([self.pos_dd, self.neg_dd], axis=0)
            ))

        # structure-description cross-entropy loss
        if self.config.loss4:
            self.loss_sd = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_sd), tf.zeros_like(self.neg_sd)], axis=0),
                logits=tf.concat([self.pos_sd, self.neg_sd], axis=0)
            ))

        # description-structure cross-entropy loss
        if self.config.loss5:
            self.loss_ds = self.reduce_func(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([tf.ones_like(self.pos_ds), tf.zeros_like(self.neg_ds)], axis=0),
                logits=tf.concat([self.pos_ds, self.neg_ds], axis=0)
            ))

        # projection l2-loss
        if self.config.loss3:
            self.projection_loss = self.reduce_func(
                tf.reduce_sum(tf.square(self.projection_embeddings - self.node_embeddings), axis=1)
            )



        if self.config.regularize:
            self.regularizer = tf.reduce_sum(tf.square(self.projection_matrix))


        loss = self.loss_ss + self.loss_dd + self.loss_ds + self.loss_sd + \
               self.config.projection_weight * self.projection_loss + \
               self.config.regularization_weight * self.regularizer
        return loss

    def regularize(self):

        #MEAN ISN'T CORRECT HERE

        if self.config.project == 'linear':
            return tf.reduce_mean(tf.square(self.projection_matrix_r) + tf.square(self.projection_matrix_i))

        if self.config.project == 'affine':
            return tf.reduce_mean(tf.square(self.projection_matrix_r) + tf.square(self.projection_matrix_i)) \
                   + tf.reduce_mean(tf.square(self.projection_bias_r) + tf.square(self.projection_bias_i))



