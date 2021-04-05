from utils.inits import *
import numpy as np
import math
from tqdm import tqdm
from tensorflow.keras.layers import Masking, Dense, Input,  LSTM, GRU, Conv1D, MaxPool1D, AveragePooling1D, Embedding, Softmax
from tensorflow.keras.activations import softplus
import sys
sys.path.append("..")

from parser import Parser
from config import Config

a = Parser().get_parser().parse_args()
c = Config(a)

from models.model import KGModel
from models.model import ProjectTranse as Project
from models.model import Aggregate
from models.model import Roberta
from models.model import conditional_tfunction
from models.model import gather_cols

class TranseModel(KGModel):

    def __init__(self, config, **kwargs):
        super(TranseModel, self).__init__(**kwargs)

        self.global_flag = None
        self.global_flag_lstm = None
        self.global_flag_projection = None
        self.node_embeddings = None
        self.word_embeddings_tf = None

        self.config = config

        self.dropout = None
        self.features_unseen_nodes = None

        self.ss_trainable = self.config.ss_trainable
        self.dataset = kwargs['dataset']

        self.projection_embeddings = tf.zeros(shape=[1])

        self.zeroshot_node_embeddings = tf.zeros(shape=[1])

        if self.config.reduce_func == "mean":
            self.reduce_func = tf.reduce_mean
        else:
            self.reduce_func = tf.reduce_sum

        if self.config.project_batchwise:
            self.nodes_to_project = None

        if not self.config.use_static_features:

            self.entityid2tokenids = kwargs['entityid2tokenids']
            self.n_nodes_total = len(self.entityid2tokenids)
            self.entityid2len_tokens = np.array([len(a) for a in self.entityid2tokenids])
            self.entityid2tokenids = self.normalise_tokens(self.entityid2tokenids)
            self.masks = np.zeros_like(self.entityid2tokenids, dtype=np.float32)
            self.entityid2tokenids_tf = tf.convert_to_tensor(self.entityid2tokenids)
            self.node_dict = kwargs['node_dict']
            self.node_dict_zeroshot = kwargs['node_dict_zeroshot']

            for i in range(self.n_nodes_total):
                self.masks[i, :self.entityid2len_tokens[i]] = 1

            self.masks_tf = tf.convert_to_tensor(self.masks)

            if self.config.aggregation_function != 'roberta':
                self.vocab_size = kwargs['vocab_size']
                self.word_embeddings_dimension = kwargs['word_embeddings_dimension']

        self.project = Project(self.config)
        self.aggregate = Aggregate(self.config)

        if self.config.aggregation_function == 'roberta':
            self.roberta = Roberta(self.config)

        self.word_embeddings = kwargs.get('word_embeddings', None)
        self.entity_we_flag = True
        self.entity_proj_flag = True
        self.init_embeddings()

        if self.config.aggregation_function != 'roberta':
            self.init_word_embeddings()

        if self.config.initialize_entities_with_we:
            self.init_entity_embeddings_we()

        if self.config.pretrained_projn:
            if self.config.projection_trainable:
                self.aggregated_embeddings = tf.zeros((10, 10), dtype=tf.float32)
            else:
                self.projected_embeddings = tf.zeros((self.config.n_nodes_total, self.config.embedding_dim), dtype=tf.float32)
        ##############################################

    @conditional_tfunction
    def init_word_embeddings(self):


        if self.word_embeddings_tf is None and self.config.aggregation_function != 'roberta':
            with tf.device(self.config.emb_device):
                self.word_embeddings_tf = Embedding(self.vocab_size, self.word_embeddings_dimension,
                                                    embeddings_initializer=tf.keras.initializers.Constant(self.word_embeddings),
                                                    trainable=True)

    @conditional_tfunction
    def init_embeddings(self):

        config = self.config

        if config.folder_suffix == 'pre_mapping':
            addendum = ''
        else:
            addendum = ''

        if self.node_embeddings is None or self.relation_embeddings is None:
            with tf.device(self.config.emb_device):

                if self.config.pretrained_node is None:
                    self.node_embeddings = tf.Variable(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                        shape=[config.n_nodes, config.embedding_dim],
                        ), name='entity_embeddings' + addendum, trainable=self.ss_trainable)

                else:

                    if self.config.pretrained_node[-3:] == 'npy':
                        self.node_embeddings = tf.Variable(
                            np.load(self.config.pretrained_node[:-4] + '.npy', allow_pickle=True)[:config.n_nodes],
                            name='entity_embeddings' + addendum, trainable=self.ss_trainable)

                    else:
                        self.node_embeddings = tf.Variable(
                            np.load(self.config.pretrained_node[:-2] + '.p', allow_pickle=True)[:config.n_nodes],
                            name='entity_embeddings'+ addendum,
                            trainable=self.ss_trainable)

                if self.config.pretrained_rel == None:
                    self.relation_embeddings = tf.Variable(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                        shape=[config.n_relations, config.embedding_dim],
                        ), name='relation_embeddings' + addendum, trainable=self.ss_trainable)

                else:
                    if self.config.pretrained_rel[-3:] == 'npy':
                        self.relation_embeddings = tf.Variable(
                            np.load(self.config.pretrained_rel[:-4] + '.npy', allow_pickle=True),
                            name='relation_embeddings' + addendum, trainable=self.ss_trainable)

                    else:
                        self.relation_embeddings = tf.Variable(
                            np.load(self.config.pretrained_rel[:-2] + '.p', allow_pickle=True),
                            name='relation_embeddings' + addendum, trainable=self.ss_trainable)

            print(tf.reduce_sum(self.node_embeddings))
            print(tf.reduce_sum(self.relation_embeddings))

        elif config.folder_suffix == 'pre_mapping':
            self.node_embeddings.assign(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                shape=[config.n_nodes, config.embedding_dim]))

            self.relation_embeddings.assign(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                shape=[config.n_relations, config.embedding_dim],),)


        # config.batch_size = temp

    def init_projected_embeddings(self):     ###start here

        if self.entity_proj_flag:
            a = self.dataset.batch_generator_for_save(0, self.config.n_nodes_total)

            description_embeddings = None

            for feed_nodes, ignore in tqdm(a):
                description_temp = self.gather_and_aggregate_words(feed_nodes, initialization=True)
                if not self.config.projection_trainable:
                    description_temp = self.project(description_temp)

                description_temp = tf.stop_gradient(description_temp).numpy()[:ignore]

                if description_embeddings is None:
                    description_embeddings = description_temp
                else:
                    description_embeddings = np.concatenate((description_embeddings, description_temp))

            with tf.device(self.config.emb_device):
                if self.config.projection_trainable:
                    self.aggregated_embeddings = tf.Variable(description_embeddings, dtype=tf.float32)
                else:
                    self.projected_embeddings = tf.Variable(description_embeddings, dtype=tf.float32)

            self.entity_proj_flag = False
            del self.project



    def normalise_tokens(self, entityid2tokenids):

        longest = max([len(x) for x in entityid2tokenids])

        self.max_seq_length = longest
        self.config.max_seq_length = longest

        for i in tqdm(range(len(entityid2tokenids))):
            while len(entityid2tokenids[i]) < longest:
                if self.config.aggregation_function == 'roberta':
                    entityid2tokenids[i].append(1)

                else:
                    entityid2tokenids[i].append(0)

        tokens_new = np.array(entityid2tokenids, dtype=np.int32)

        return tokens_new


    def simply_project(self, nodes_to_project):
        self.zeroshot_node_embeddings = None
        self.projection_embeddings = None

        # nodes_to_project = tf.Tensor(nodes_to_project)

        self.simply_gathered = self.gather_and_aggregate_words(nodes_to_project, train=False)

        if self.config.project == 'combine':

            if nodes_to_project[0] >= self.config.n_nodes:
                return tf.convert_to_tensor(np.random.normal(size=(self.config.batch_size, self.config.embedding_dim)).astype(np.float32)), \
            tf.convert_to_tensor(np.random.normal(size=(self.config.batch_size, self.config.embedding_dim)).astype(np.float32))

            n = tf.gather(self.node_embeddings, nodes_to_project)

            d = self.simply_gathered

            self.simply_projected = self.project(d, n)

        else:
            if not self.config.pretrained_projn:
                self.simply_projected = self.project(self.simply_gathered)
            else:
                self.simply_projected = self.simply_gathered

            if self.config.normalize_embed:
                self.simply_projected = tf.nn.l2_normalize(self.simply_projected, dim=1)

        return self.simply_projected

    @conditional_tfunction
    def call(self, positive_samples, negative_samples):

        self.pos_samples = positive_samples
        self.neg_samples = negative_samples

        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

        if self.config.normalize_embed:
            self.node_embeddings = tf.nn.l2_normalize(self.node_embeddings, dim=1)

            self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, dim=1)
                #TODO normalise batch wise also

        self.pos_ss, self.pos_sd, self.pos_ds, self.pos_dd = self.predict(self.pos_samples, pos=True)

        self.pos_src_struc_embed = self.src_struc_embed

        self.pos_tar_struc_embed = self.tar_struc_embed

        self.pos_rel_struc_embed = self.rel_struc_embed
        #================================================================================

        self.pos_src_desc_embed = self.src_desc_embed

        self.pos_tar_desc_embed = self.tar_desc_embed

        self.neg_ss, self.neg_sd, self.neg_ds, self.neg_dd = self.predict(self.neg_samples)

        self.neg_src_struc_embed = self.src_struc_embed

        self.neg_tar_struc_embed = self.tar_struc_embed

        # ================================================================================

        self.neg_src_desc_embed = self.src_desc_embed

        self.neg_tar_desc_embed = self.tar_desc_embed

        self.optimizer()

        if not self.config.corr:
            return self.cost, self.projection_loss, self.loss_ss, self.loss_sd, self.loss_ds, self.loss_dd
        else:
            return self.cost, self.projection_loss, self.loss_ss, self.loss_sd, self.loss_ds, self.loss_dd, \
                   self.pos_src_struc_embed, self.pos_rel_struc_embed, self.pos_tar_struc_embed

    def sum(self, node_token_embeddings, node_masks, divide_by=1):

        masked = node_token_embeddings * tf.expand_dims(node_masks, 2)

        reduced = tf.reduce_sum(masked, axis=1)

        return reduced / divide_by

    def average(self, node_token_embeddings, node_masks):

        masked = node_token_embeddings * tf.expand_dims(node_masks, 2)

        reduced = tf.reduce_sum(masked, axis=1)

        n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

        averaged = reduced / n_tokens

        return averaged

    def bilstm(self, node_token_embeddings, node_masks, reuse=None, name_to_append=''):

        node_token_embeddings = tf.reshape(node_token_embeddings,
                                           (self.config.batch_size, self.max_seq_length, self.config.embedding_dim))

        node_masks = tf.reshape(node_masks, (self.config.batch_size, -1))
        n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

        embs_masked = Masking()(node_token_embeddings)

        with tf.variable_scope('description_lstm' + name_to_append, reuse=reuse):
            print('once')

            lstm_forward = LSTM(450, activation='tanh', return_sequences=True)(embs_masked)

            lstm_out = tf.reduce_sum(lstm_out, axis=1) / n_tokens

            lstm_out = Dense(300)(lstm_out)

            return lstm_out

    def lstm(self, node_token_embeddings, node_masks, reuse=None, name_to_append=''):

        node_token_embeddings = tf.reshape(node_token_embeddings,
                                           (self.config.batch_size, self.max_seq_length, self.config.embedding_dim))

        node_masks = tf.reshape(node_masks, (self.config.batch_size, -1))
        n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

        embs_masked = Masking()(node_token_embeddings)

        with tf.variable_scope('description_lstm' + name_to_append, reuse=reuse):
            print('once')

            lstm_out = LSTM(450, activation='tanh', return_sequences=True)(embs_masked)

            lstm_out = tf.reduce_sum(lstm_out, axis=1) / n_tokens

            lstm_out = Dense(300)(lstm_out)

            return lstm_out

    def cnn(self, node_token_embeddings, node_masks):

        node_token_embeddings = tf.reshape(node_token_embeddings,
                                           (self.config.batch_size, self.max_seq_length, self.config.embedding_dim))

        out1 = Conv1D(300, 3, 1, padding='same', activation='relu')(node_token_embeddings)
        out2 = MaxPool1D(padding='same')(out1)
        out3 = Conv1D(300, 2, 1, padding='same', activation='relu')(out2)
        out4 = MaxPool1D(padding='same')(out3)
        out5 = Conv1D(300, 3, 2, padding='same', activation='relu')(out4)
        out6 = MaxPool1D(4, padding='same')(out5)

        return tf.squeeze(out6)

    def attention(self, node_token_embeddings, node_masks, reuse=None):

        with tf.variable_scope('attentive_aggregation', reuse=reuse):
            node_token_embeddings = tf.reshape(node_token_embeddings, (self.config.batch_size, -1, self.config.embedding_dim))

            n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

            attention_weights = Dense(1)(node_token_embeddings)

            weighted = node_token_embeddings * attention_weights

            reduced = tf.reduce_sum(weighted, axis=1)

            return reduced / n_tokens

    def gather_and_aggregate_words(self, indices, train=True, initialization=False):

        indices = tf.convert_to_tensor(indices)

        node_tokens = tf.gather(self.entityid2tokenids_tf, indices, axis=0)
        node_masks = tf.gather(self.masks_tf, indices, axis=0)

        if initialization:
            if self.config.aggregation_function != 'roberta':
                node_token_embeddings = self.word_embeddings_tf(node_tokens)

                node_embeddings, _ = self.aggregate(node_token_embeddings, node_masks, train)

            else:
                node_embeddings, _ = self.roberta(node_tokens, node_masks, training=train)

        else:
            if self.config.aggregation_function != 'roberta' and not self.config.pretrained_projn:
                node_token_embeddings = self.word_embeddings_tf(node_tokens)

                node_embeddings, _ = self.aggregate(node_token_embeddings, node_masks, train)

            elif self.config.pretrained_projn:
                if self.config.projection_trainable:
                    node_embeddings = tf.gather(self.aggregated_embeddings, indices, axis=0)
                else:
                    node_embeddings = tf.gather(self.projected_embeddings, indices, axis=0)

            else:
                node_embeddings, _ = self.roberta(node_tokens, node_masks, training=train)

        return node_embeddings

    def predict(self, inputs, pos=False):
        src_indices = tf.squeeze(gather_cols(tf.convert_to_tensor(inputs), tf.convert_to_tensor([0])))
        rel_indices = tf.squeeze(gather_cols(tf.convert_to_tensor(inputs), tf.convert_to_tensor([1])))
        tar_indices = tf.squeeze(gather_cols(tf.convert_to_tensor(inputs), tf.convert_to_tensor([2])))

        self.src_struc_embed = tf.nn.embedding_lookup(self.node_embeddings, src_indices)

        self.tar_struc_embed = tf.nn.embedding_lookup(self.node_embeddings, tar_indices)

        rel_struc_embed = tf.nn.embedding_lookup(self.relation_embeddings, rel_indices)

        self.rel_struc_embed = rel_struc_embed

        preds_ss = None
        preds_sd = None
        preds_ds = None
        preds_dd = None

        self.src_desc_embed = None

        self.tar_desc_embed = None

        if self.config.zeroshot:

            if self.config.project == 'combine':
                src_gathered = self.gather_and_aggregate_words(src_indices, train=True)

                tar_gathered = self.gather_and_aggregate_words(tar_indices, train=True)

                if not self.config.pretrained_projn or self.config.projection_trainable:
                    self.src_desc_embed = \
                        self.project(src_gathered, self.src_struc_embed, train=True)
                    self.tar_desc_embed = \
                        self.project(tar_gathered , self.tar_struc_embed, train=True)

                else:
                    raise ValueError

            else:

                src_gathered = self.gather_and_aggregate_words(src_indices, train=True)

                tar_gathered = self.gather_and_aggregate_words(tar_indices, train=True)

                if not self.config.pretrained_projn or self.config.projection_trainable:
                    self.src_desc_embed = self.project(src_gathered, train=True)
                    self.tar_desc_embed = self.project(tar_gathered, train=True)

                else:
                    self.src_desc_embed = src_gathered
                    self.tar_desc_embed = tar_gathered

        if self.config.decoder == 'dismult':
            formulation = self.dismult_formulation
        if self.config.decoder == 'transe':
            formulation = self.transe_formulation

        if self.config.loss1:
            if self.config.project != 'combine':
                preds_ss = formulation(self.src_struc_embed, rel_struc_embed,
                                                    self.tar_struc_embed)

            else:
                preds_ss = formulation(self.src_desc_embed, rel_struc_embed,
                                                   self.tar_desc_embed)

        if self.config.loss2:
            preds_dd = formulation(self.src_desc_embed, rel_struc_embed, self.tar_desc_embed)

        if self.config.loss4:
            preds_sd = formulation(self.src_struc_embed, rel_struc_embed,
                                               self.tar_desc_embed)

        if self.config.loss5:
            preds_ds = formulation(self.src_desc_embed, rel_struc_embed,  self.tar_struc_embed)

        return preds_ss, preds_sd, preds_ds, preds_dd

    def dismult_formulation(self, src, rel, tar):

        p1 = tf.reduce_sum(src * rel * tar, axis=-1)

        phi = p1

        return phi

    def transe_formulation(self, src, rel, tar):

        src = tf.nn.l2_normalize(src, -1)
        rel = tf.nn.l2_normalize(rel, -1)
        tar = tf.nn.l2_normalize(tar, -1)

        phi = tf.math.abs(src + rel - tar)
        phi = tf.reduce_sum(-phi, axis=-1)

        return phi

    def optimizer(self):
        self.loss_ss = tf.zeros(shape=[1])
        self.loss_dd = tf.zeros(shape=[1])
        self.loss_sd = tf.zeros(shape=[1])
        self.loss_ds = tf.zeros(shape=[1])
        self.projection_loss = tf.zeros(shape=[1])
        self.regularizer = tf.zeros(shape=[1])

        self.cost = self.loss()

    def dismult_reduce(self, pos, neg, reduce=True):
        if reduce:
            loss = ( self.reduce_func(softplus(-pos)) + self.reduce_func(softplus(neg)) ) / 2
        else:
            loss =  ( softplus(-pos) + softplus(neg) ) / 2

        return loss

    def transe_reduce(self, pos, neg, reduce=True):

        pos = -tf.reshape(pos, (pos.shape[0], 1))
        neg = -tf.reshape(neg, (pos.shape[0], -1))

        if reduce:
            #shape of loss before meaning is batch size * n_neg_samples
            loss = tf.reduce_mean(tf.maximum(pos - neg + self.config.max_margin, 0))
        else:
            loss = tf.reduce_mean(tf.maximum(pos - neg + self.config.max_margin, 0), 1)

        return loss

    def loss(self):

        if self.config.decoder == 'dismult':
            self.reduce = self.dismult_reduce
        if self.config.decoder == 'transe':
            self.reduce = self.transe_reduce

        if self.config.loss1:
            self.loss_ss = self.reduce(self.pos_ss, self.neg_ss)

        if self.config.loss2:
            self.loss_dd = self.reduce(self.pos_dd, self.neg_dd)

        if self.config.loss4:
            self.loss_sd = self.reduce(self.pos_sd, self.neg_sd)

        if self.config.loss5:
            self.loss_ds = self.reduce(self.pos_ds, self.neg_ds)

        if self.config.loss3:

            if not self.config.project_batchwise:
                # if self.config.train_intermediate:
                self.projection_loss = self.reduce_func(
                    tf.sqrt(tf.reduce_sum(tf.square(self.projection_embeddings - self.node_embeddings), axis=1)) )

            else:
                #TODO include target projection also
                self.ploss_pre_src = tf.sqrt(tf.reduce_sum(tf.square(self.pos_src_struc_embed - self.pos_src_desc_embed), axis=1))

                self.ploss_pre_tar = tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed - self.pos_tar_desc_embed), axis=1))

                if self.config.weight_projections:
                    copy = tf.identity(1 / self.pos_ss)

                    self.ploss_pre_src = self.ploss_pre_src * tf.nn.sigmoid(copy)
                    self.ploss_pre_tar = self.ploss_pre_tar * tf.nn.sigmoid(copy)

                projection_loss_source = self.reduce_func(self.ploss_pre_src)
                projection_loss_target = self.reduce_func(self.ploss_pre_tar)

                self.projection_loss = projection_loss_source + projection_loss_target

        all_src_embeds = tf.stack((self.pos_src_struc_embed, self.pos_src_desc_embed, self.neg_src_struc_embed, self.neg_src_desc_embed))
        all_tar_embeds = tf.stack((self.pos_tar_struc_embed, self.pos_tar_desc_embed, self.neg_tar_struc_embed, self.neg_tar_desc_embed))

        self.embed_regularization_loss = tf.reduce_mean(all_src_embeds ** 2 + all_tar_embeds ** 2)

        print('        ***** embed reg loss shape ************* \n\n\n')

        print(self.embed_regularization_loss.shape)

        print('        ***** embed reg loss shape ************* \n\n\n')

        if self.config.regularize:
            self.regularizer = self.regularize()



        if self.config.separate_projection:
            loss = self.loss_ss + self.loss_dd + self.loss_ds + self.loss_sd + \
                   self.config.regularization_weight * self.regularizer

            self.projection_loss *= self.config.projection_weight

            return loss

        else:
            loss = self.loss_ss + self.loss_dd + self.loss_ds + self.loss_sd + \
                   self.config.projection_weight * self.projection_loss + \
                   self.config.regularization_weight * self.regularizer + \
                   self.config.regularization_weight * self.embed_regularization_loss

        return loss

    def regularize(self):

        #MEAN ISN'T CORRECT HERE

        if self.config.project == 'linear':
            return tf.reduce_mean(tf.square(self.projection_matrix_r) + tf.square(self.projection_matrix_i))

        if self.config.project == 'affine':
            return tf.reduce_mean(tf.square(self.projection_matrix_r) + tf.square(self.projection_matrix_i)) \
                   + tf.reduce_mean(tf.square(self.projection_bias_r) + tf.square(self.projection_bias_i))



