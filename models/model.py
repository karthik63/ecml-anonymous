
import comet_ml
from collections import defaultdict
import tensorflow as tf
from utils.inits import *
import numpy as np
import math
from tqdm import tqdm




# from main import conditional_tfunction

# from tensorflow.keras.activations import relu
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Masking, Dense, Input, LSTM, GRU, Conv1D, MaxPool1D, AveragePooling1D
#
# from tensorflow.contrib.keras.python.keras.activations import relu
# from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
# from tensorflow.contrib.keras.python.keras.models import Sequential
# from tensorflow.contrib.keras.python.keras.layers import Masking, Dense, Input,  LSTM, GRU, Conv1D, MaxPool1D, AveragePooling1D, Masking
# from tensorflow.python.ops.rnn import static_bidirectional_rnn

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Masking, Dense, Input,  LSTM, GRU, Conv1D, MaxPool1D, AveragePooling1D, Embedding, Softmax
from tensorflow.keras.layers import Bidirectional
from transformers import TFRobertaModel, RobertaTokenizer
from tensorflow.keras.activations import softplus
import sys
sys.path.append("..")

from parser import Parser
from config import Config
# from tensorflow.contrib.cudnn_rnn import CuDNNLSTM, CuDNNGRU

a = Parser().get_parser().parse_args()
c = Config(a)

def conditional_tfunction(func):
    if c.graph_mode:
        return tf.function(func)

    else:
        return func

# @conditional_tfunction
def gather_cols(params, indices, name=None, numpy=False):
    """Gather columns of a 2D tensor.

	Args:
		params: A 2D tensor.
		indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
		name: A name for the operation (optional).

	Returns:
		A 2D Tensor. Has the same type as ``params``.
	"""

    # if not numpy:
        # Check input
        # print(params)
        # print(indices)

        # params = tf.convert_to_tensor(params)
        # indices = tf.convert_to_tensor(indices)
        # try:
        #     params.get_shape().assert_has_rank(2)
        # except ValueError:
        #     raise ValueError('\'params\' must be 2D.')
        # try:
        #     indices.get_shape().assert_has_rank(1)
        # except ValueError:
        #     raise ValueError('\'params\' must be 1D.')

        # Define op
    p_shape = tf.shape(params)
    p_flat = tf.reshape(params, [-1])
    i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                   [-1, 1]) + indices, [-1])

    answer = tf.reshape(tf.gather(p_flat, i_flat), [p_shape[0], -1])

    # j = tf.Print(answer, [tf.shape(answer)])

    return answer

    # if numpy:
    #     p_shape = params.shape
    #     p_flat = np.reshape(params, (-1))
    #     i_flat = np.reshape(np.reshape(np.arange(0, p_shape[0]) * p_shape[1], [-1,1]) + indices, [-1])
    #     a = np.reshape(np.take(p_flat, i_flat), [p_shape[0], -1])
    #     return a

class KGModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(KGModel, self).__init__()
        # TODO uncomment this
        # allowed_kwargs = {'name', 'logging'}
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        #
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')

        logging = kwargs.get('logging', False)
        self.logging = logging


    def complex_formulation_numpy(self, src_r, src_i, rel_r, rel_i, tar_r, tar_i):

        p1 = np.sum(rel_r * src_r * tar_r, axis=-1)
        p2 = np.sum(rel_r * src_i * tar_i, axis=-1)
        p3 = np.sum(rel_i * src_r * tar_i, axis=-1)
        p4 = np.sum(rel_i * src_i * tar_r, axis=-1)

        phi = p1 + p2 + p3 - p4

        return phi

    def score(self, head, rel, tail, predict_tails=True):

        if self.config.decoder == 'complex':

            if predict_tails:

                src_r = tf.gather(self.node_all_r, head)
                src_i = tf.gather(self.node_all_i, head)

                rel_r = tf.gather(self.node_all_r, rel)
                rel_i = tf.gather(self.node_all_i, rel)

                return self.complex_formulation(src_r, src_i, rel_r, rel_i, self.node_all_r, self.node_all_i)

            else:
                tar_r = tf.gather(self.node_all_r, tail)
                tar_i = tf.gather(self.node_all_i, tail)

                rel_r = tf.gather(self.node_all_r, rel)
                rel_i = tf.gather(self.node_all_i, rel)

                return self.complex_formulation(self.node_all_r, self.node_all_i, rel_r, rel_i, tar_r, tar_i)

    def gather_and_reshape(self, indices, special=False, relation=False):

        if relation:
            gathered_r, gathered_i = tf.gather(self.all_relations_r, indices), tf.gather(self.all_relations_i, indices)

        else:
            gathered_r, gathered_i = tf.gather(self.all_nodes_r, indices), tf.gather(self.all_nodes_i, indices)

        if special:
            reshaped_r = tf.expand_dims(gathered_r, 0)
            reshaped_i = tf.expand_dims(gathered_i, 0)

        else:
            reshaped_r = tf.expand_dims(gathered_r, 1)
            reshaped_i = tf.expand_dims(gathered_i, 1)

        return reshaped_r, reshaped_i

    def gather_and_reshape_transe(self, indices, special=False, relation=False):

        if relation:
            gathered = tf.gather(self.all_relations, indices)

        else:
            gathered = tf.gather(self.all_nodes, indices)

        if special:
            reshaped = tf.expand_dims(gathered, 0)
        else:
            reshaped = tf.expand_dims(gathered, 1)

        return reshaped

    def get_ranks(self, scores, correct_index):

        indices = tf.nn.top_k(scores, self.end - self.start).indices
        mask = tf.equal(indices, tf.expand_dims(correct_index, 1))
        ranks = tf.where(mask)[:, 1]
        # ranks = tf.Print(ranks, ['ranks_yo', ranks])
        return ranks + 1

    def compute_total_hits_at_n(self, ranks, n):

        mask = tf.less_equal(ranks, n)
        total = tf.reduce_sum(tf.cast(mask, tf.float32))

        return total

    def compute_total_metrics(self, ranks):

        total_rank = tf.reduce_sum(ranks)
        total_rec_rank = tf.reduce_sum(1/ranks)

        total_hits_at_1 = self.compute_total_hits_at_n(ranks, 1)
        total_hits_at_3 = self.compute_total_hits_at_n(ranks, 3)
        total_hits_at_10 = self.compute_total_hits_at_n(ranks, 10)

        ans = {'rank': total_rank,
               'rec_rank': total_rec_rank,
               'hits@1': total_hits_at_1,
               'hits@3': total_hits_at_3,
               'hits@10': total_hits_at_10,}

        return ans

    def evaluate_util(self, batch_triplets, head_labels, tail_labels, invalid_targets, predict_tails=True):

        self.batch_triplets = batch_triplets
        self.head_labels = head_labels
        self.tail_labels = tail_labels
        self.invalid_targets = invalid_targets

        batch_size = tf.shape(self.batch_triplets)[0]

        if self.config.decoder == 'dismult':
            self.complex_formulation = self.dismult_formulation
        if self.config.decoder == 'transe':
            self.complex_formulation = self.transe_formulation

        if predict_tails:

            if self.config.decoder == 'complex':
                src_r, src_i = self.gather_and_reshape(self.batch_triplets[:, 0])

                rel_r, rel_i = self.gather_and_reshape(self.batch_triplets[:, 1], relation=True)

                range = tf.range(self.start, self.end)

                tar_r, tar_i = self.gather_and_reshape(range, special=True)

                tar_r = tf.tile(tar_r, (batch_size, 1, 1))
                tar_i = tf.tile(tar_i, (batch_size, 1, 1))

                correct_targets = self.batch_triplets[:, 2]

                scores = self.complex_formulation(src_r, src_i, rel_r, rel_i, tar_r, tar_i)

            else:
                src = self.gather_and_reshape_transe(self.batch_triplets[:, 0])

                rel  = self.gather_and_reshape_transe(self.batch_triplets[:, 1], relation=True)

                range = tf.range(self.start, self.end)

                tar  = self.gather_and_reshape_transe(range, special=True)

                tar = tf.tile(tar, (batch_size, 1, 1))

                correct_targets = self.batch_triplets[:, 2]

                scores = self.complex_formulation(src, rel, tar)

            temp_inf = tf.ones_like(scores) * -np.inf

            filtered_scores = tf.where(self.tail_labels, x=temp_inf, y=scores)
            target_filtered_scores = tf.where(self.invalid_targets, x=temp_inf, y=filtered_scores)

            self.filtered_scores = filtered_scores
            self.target_filtered_scores = target_filtered_scores

            raw_ranks = self.get_ranks(scores, correct_targets)
            filtered_ranks = self.get_ranks(filtered_scores, correct_targets)
            target_filtered_ranks = self.get_ranks(target_filtered_scores, correct_targets)

            self.scores = scores
            self.sorted_scores = tf.nn.top_k(filtered_scores, k=self.end - self.start).indices
            self.rr = raw_ranks
            self.fr = filtered_ranks
            self.tfr = target_filtered_ranks

            raw_totals = self.compute_total_metrics(raw_ranks)
            filtered_totals = self.compute_total_metrics(filtered_ranks)
            target_filtered_totals = self.compute_total_metrics(target_filtered_ranks)
        else:
            if self.config.decoder == 'complex':

                tar_r, tar_i = self.gather_and_reshape(self.batch_triplets[:, 2])

                rel_r, rel_i = self.gather_and_reshape(self.batch_triplets[:, 1], relation=True)

                range = tf.range(self.start, self.end)

                src_r, src_i = self.gather_and_reshape(range, special=True)

                src_r = tf.tile(src_r, (batch_size, 1, 1))
                src_i = tf.tile(src_i, (batch_size, 1, 1))

                correct_sources = self.batch_triplets[:, 0]

                scores = self.complex_formulation(src_r, src_i, rel_r, rel_i, tar_r, tar_i)
            else:
                tar = self.gather_and_reshape_transe(self.batch_triplets[:, 2])

                rel = self.gather_and_reshape_transe(self.batch_triplets[:, 1], relation=True)

                range = tf.range(self.start, self.end)

                src = self.gather_and_reshape_transe(range, special=True)

                src = tf.tile(src, (batch_size, 1, 1))

                correct_sources = self.batch_triplets[:, 0]

                scores = self.complex_formulation(src, rel, tar)

            temp_inf = tf.ones_like(scores) * -np.inf

            # TODO source filtering

            filtered_scores = tf.where(self.head_labels, x=temp_inf, y=scores)

            self.filtered_scores = filtered_scores

            raw_ranks = self.get_ranks(scores, correct_sources)
            filtered_ranks = self.get_ranks(filtered_scores, correct_sources)

            self.scores = scores
            self.sorted_scores = tf.nn.top_k(filtered_scores, k=self.end - self.start).indices
            self.rr = raw_ranks
            self.fr = filtered_ranks

            raw_totals = self.compute_total_metrics(raw_ranks)
            filtered_totals = self.compute_total_metrics(filtered_ranks)
            target_filtered_totals = None

        return raw_totals, filtered_totals, target_filtered_totals

    def compute_totals(self, new, total):

        for key, value in new.items():
            total[key] += value

        return total

    def compute_means(self, total, n):

        for key, value in total.items():
            total[key] /= n

        return total

    def evaluate(self, all_nodes, all_relations, predict='tails',
                 split='valid_zero', types=('ed')):

        if self.config.decoder == 'complex':
            all_nodes_r = all_nodes[0]
            all_nodes_i = all_nodes[1]

            all_relations_r = all_relations[0]
            all_relations_i = all_relations[1]

        if split=='valid_zero':
            n_triples = self.config.n_val_zeroshot
        if split=='test_zero':
            n_triples = self.config.n_test_zeroshot
        if split=='test':
            types = ('ss')
            if self.config.decoder == 'complex':
                all_nodes_r = all_nodes_r[:self.config.n_nodes]
                all_nodes_i = all_nodes_i[:self.config.n_nodes]
            else:
                all_nodes = all_nodes[:self.config.n_nodes]
            n_triples = self.config.n_test
        if split=='valid':
            types = ('ss')

            if self.config.decoder == 'complex':
                all_nodes_r = all_nodes_r[:self.config.n_nodes]
                all_nodes_i = all_nodes_i[:self.config.n_nodes]
            else:
                all_nodes = all_nodes[:self.config.n_nodes]
            
            n_triples = self.config.n_val

        if split=='train':
            types = ('ss')

            if self.config.decoder == 'complex':
    
                all_nodes_r = all_nodes_r[:self.config.n_nodes]
                all_nodes_i = all_nodes_i[:self.config.n_nodes]
            else:
                all_nodes = all_nodes[:self.config.n_nodes]
            n_triples = self.config.n_train_for_evaluation

        if self.config.decoder == 'complex':
            self.all_nodes_r = tf.convert_to_tensor(all_nodes_r)
            self.all_nodes_i = tf.convert_to_tensor(all_nodes_i)
        else:
            self.all_nodes = tf.convert_to_tensor(all_nodes)

        if self.config.decoder == 'complex':
            self.all_relations_r = tf.convert_to_tensor(all_relations_r)
            self.all_relations_i = tf.convert_to_tensor(all_relations_i)
        else:
            self.all_relations = tf.convert_to_tensor(all_relations)

        triplets = self.dataset.batch_generator_for_eval_init(split=split, types=types)

        self.start = self.dataset.start
        self.end = self.dataset.end
        n_triples = self.dataset.current_n

        n = 0
        ranks = []

        for index, (batch_triplets, head_labels, tail_labels, invalid_targets) in tqdm(
                enumerate(self.dataset.batch_generator_for_eval(triplets)), mininterval=30):

            n += batch_triplets.shape[0]

            raw, filtered, target_filtered = self.evaluate_util(batch_triplets, head_labels, tail_labels,
                                                                invalid_targets, predict_tails=True)

            if types == ('ss'):
                n += batch_triplets.shape[0]

                raw_src, filtered_src, target_filtered_src = self.evaluate_util(batch_triplets, head_labels,
                                                                                tail_labels,
                                                                                invalid_targets, predict_tails=False)


            # ranks += fr.tolist()

            if index == 0:

                if types != ('ss'):
                    total_raw, total_filtered, total_target_filtered = raw, \
                        filtered, target_filtered
                else:

                    total_raw, total_filtered = self.compute_totals(raw, raw_src), \
                                                self.compute_totals(filtered, filtered_src)

            else:
                if types != ('ss'):
                    total_raw = self.compute_totals(raw, total_raw)
                    total_filtered = self.compute_totals(filtered, total_filtered)
                    total_target_filtered = self.compute_totals(target_filtered, total_target_filtered)

                else:
                    total_raw_step = self.compute_totals(raw, raw_src)
                    total_raw = self.compute_totals(total_raw, total_raw_step)

                    total_filtered_step = self.compute_totals(filtered, filtered_src)
                    total_filtered = self.compute_totals(total_filtered_step, total_filtered)

        # ranks = np.array(ranks)

        n_divide = n_triples
        if types == ('ss'):
            n_divide *= 2

        raw_metrics = self.compute_means(total_raw, n_divide)
        filtered_metrics = self.compute_means(total_filtered, n_divide)

        if types!= ('ss'):
            target_filtered_metrics = self.compute_means(total_target_filtered, n_divide)
        else:
            target_filtered_metrics = {}

        return raw_metrics, filtered_metrics, target_filtered_metrics


class Project(tf.keras.Model):
    def __init__(self, config):
        super(Project, self).__init__()

        self.config = config

        if self.config.project in ['affine', 'combine']:
            project_r = Dense(self.config.embedding_dim)
            project_i = Dense(self.config.embedding_dim)

        if self.config.project == 'linear':
            project_r = Dense(self.config.embedding_dim, use_bias=False)
            project_i = Dense(self.config.embedding_dim, use_bias=False)

        if self.config.project == 'mlp':
            # TODO vk modify this, embedding_dim is wrong
            project_r = Sequential()
            project_r.add(Dense(700, activation='selu'))
            project_r.add(Dense(300))

            project_i = Sequential()
            project_i.add(Dense(700, activation='selu'))
            project_i.add(Dense(300))

        self.project_r = project_r
        self.project_i = project_i

    @conditional_tfunction
    def call(self, d_r, d_i, e_r=None, e_i=None, train=False):

        if self.config.project == 'combine':
            combined_r = tf.concat((d_r / 1971, e_r), 1)
            combined_i = tf.concat((d_i / 1971, e_i), 1)

            return self.project_r(combined_r), self.project_i(combined_i)

            # combined_r = (1/1971) * self.project_r(d_r) + e_r
            # combined_i = (1/1971) * self.project_i(d_i) + e_i
            #
            # return combined_r, combined_i

        return self.project_r(d_r), self.project_i(d_i)

class ProjectTranse(tf.keras.Model):
    def __init__(self, config):
        super(ProjectTranse, self).__init__()

        self.config = config

        if self.config.project in ['affine', 'combine']:
            project = Dense(self.config.embedding_dim)

        if self.config.project == 'linear':
            project = Dense(self.config.embedding_dim, use_bias=False)

        if self.config.project == 'mlp':
            # TODO vk modify this, embedding_dim is wrong
            project = Sequential()
            project.add(Dense(700, activation='selu'))
            project.add(Dense(300))

        self.project = project

    @conditional_tfunction
    def call(self, d, e=None, train=False):

        if self.config.project == 'combine':
            combined = tf.concat((d / 1971, e), 1)

            return self.project(combined)

            # combined_r = (1/1971) * self.project_r(d_r) + e_r
            # combined_i = (1/1971) * self.project_i(d_i) + e_i
            #
            # return combined_r, combined_i

        return self.project(d)

class Roberta(tf.keras.Model):

    def __init__(self, config):
        super(Roberta, self).__init__()

        self.config = config

        if self.config.dataset_name == 'wn18rr':
            self.roberta = TFRobertaModel.from_pretrained('roberta-base')

        else:
            self.roberta = TFRobertaModel.from_pretrained('roberta-base')

        # self.roberta = TFRobertaModel.from_pretrained('distilroberta-base')
        self.pool_step = tf.keras.layers.MaxPool1D(self.config.max_seq_length)

        cls_mask = np.ones((1, self.config.max_seq_length), np.float32)
        cls_mask[:,0] = 0
        cls_mask = np.expand_dims(cls_mask, 2)
        self.cls_mask = tf.convert_to_tensor(cls_mask)

    @conditional_tfunction
    def call(self, node_tokens, node_masks, training):

        a = self.roberta([node_tokens, node_masks], training=training)[0]

        if self.config.roberta_cls:
            cls_tokens = a[:,0,:]

            return cls_tokens, cls_tokens

        elif not self.config.roberta_pool:
            masked = a * tf.expand_dims(node_masks, 2)
            masked = masked * self.cls_mask
            reduced = tf.reduce_sum(masked, axis=1)
            n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)
            averaged = reduced / n_tokens

            return averaged, averaged

        else:
            masked = a * tf.expand_dims(node_masks, 2)

            if not self.config.roberta_cls_avg:
                masked = masked * self.cls_mask
            # b = masked.numpy()
            reduced = tf.squeeze(self.pool_step(masked))
            # c = reduced.numpy()
            n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)
            averaged = reduced / n_tokens

            return averaged, averaged


class Aggregate(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(Aggregate, self).__init__()

        self.config = config

        if self.config.aggregation_function == 'attention':
            self.attention = Sequential(name='attention')
            self.attention.add(Masking())
            self.attention.add(Dense(1, activation='sigmoid'))

        if self.config.aggregation_function == 'bilstm':
            self.bid = Sequential(name='bilstm')
            self.bid.add(Bidirectional(LSTM(400, activation='selu', return_sequences=True)))

    @conditional_tfunction
    def call(self, node_token_embeddings, node_masks, train=False):

        if self.config.aggregation_function == 'average':

            masked = node_token_embeddings * tf.expand_dims(node_masks, 2)
            reduced = tf.reduce_sum(masked, axis=1)
            n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)
            averaged = reduced / n_tokens

            return averaged, averaged

        if self.config.aggregation_function == 'attention':

            masked = node_token_embeddings * tf.expand_dims(node_masks, 2)

            att_values = self.attention(masked)

            att_values = att_values * tf.expand_dims(node_masks, 2)

            masked = masked * att_values

            n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)
            reduced = tf.reduce_sum(masked, 1)
            averaged = reduced / n_tokens

            return averaged, averaged

        if self.config.aggregation_function == 'bilstm':

            sequences = self.bid(node_token_embeddings, node_masks)
            masked = sequences * tf.expand_dims(node_masks, 2)

            n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)
            reduced = tf.reduce_sum(masked, 1)
            averaged = reduced / n_tokens

            return averaged, averaged



class ComplexModel(KGModel):

    def __init__(self, config, **kwargs):
        super(ComplexModel, self).__init__(**kwargs)

        self.global_flag = None
        self.global_flag_lstm = None
        self.global_flag_projection = None
        self.node_embeddings_r = None
        self.word_embeddings_tf = None

        self.config = config

        # self.pos_samples = None
        # self.neg_samples = None

        ## strange stuff
        self.dropout = None
        self.features_unseen_nodes = None

        self.ss_trainable = self.config.ss_trainable
        self.dataset = kwargs['dataset']

        self.projection_embeddings_r = tf.zeros(shape=[1])
        self.projection_embeddings_i = tf.zeros(shape=[1])

        self.zeroshot_node_embeddings_r = tf.zeros(shape=[1])
        self.zeroshot_node_embeddings_i = tf.zeros(shape=[1])

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

            # self.word_embeddings_tf = kwargs['word_embeddings_tf']

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
                self.aggregated_embeddings_r = tf.zeros((10, 10), dtype=tf.float32)
                self.aggregated_embeddings_i = tf.zeros((10, 10), dtype=tf.float32)

            else:
                self.projected_embeddings_r = tf.zeros((self.config.n_nodes_total, self.config.embedding_dim), dtype=tf.float32)
                self.projected_embeddings_i = tf.zeros((self.config.n_nodes_total, self.config.embedding_dim), dtype=tf.float32)

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

        # temp = config.batch_size
        # config.batch_size = config.projection_batch_size

        # if config.folder_suffix == 'pre_mapping':
        #
        #     self.node_embeddings_r = None
        #     self.node_embeddings_i = None
        #
        #     self.relation_embeddings_r = None
        #     self.relation_embeddings_i = None

        if config.folder_suffix == 'pre_mapping':
            addendum = ''
        else:
            addendum = ''

        if self.node_embeddings_r is None or self.relation_embeddings_r is None: 
            with tf.device(self.config.emb_device):

                if self.config.pretrained_node is None:
                    self.node_embeddings_r = tf.Variable(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                        shape=[config.n_nodes, config.embedding_dim],
                        ), name='entity_embeddings_real' + addendum, trainable=self.ss_trainable)

                    self.node_embeddings_i = tf.Variable(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                        shape=[config.n_nodes, config.embedding_dim],
                        ), name='entity_embeddings_imaginary' + addendum, trainable=self.ss_trainable)

                else:

                    # a = np.load(self.config.pretrained_node[:-2] + '_r.p', allow_pickle=True)

                    if self.config.pretrained_node[-3:] == 'npy':
                        self.node_embeddings_r = tf.Variable(
                            np.load(self.config.pretrained_node[:-4] + '_r.npy', allow_pickle=True)[:config.n_nodes],
                            name='entity_embeddings_real' + addendum, trainable=self.ss_trainable)
                        self.node_embeddings_i = tf.Variable(
                            np.load(self.config.pretrained_node[:-4] + '_i.npy', allow_pickle=True)[:config.n_nodes],
                            name='entity_embeddings_imaginary' + addendum, trainable=self.ss_trainable)

                    else:
                        self.node_embeddings_r = tf.Variable(\
                            np.load(self.config.pretrained_node[:-2] + '_r.p', allow_pickle=True)[:config.n_nodes],
                            name='entity_embeddings_real'+ addendum,
                            trainable=self.ss_trainable)
                        self.node_embeddings_i = tf.Variable(\
                            np.load(self.config.pretrained_node[:-2] + '_i.p', allow_pickle=True)[:config.n_nodes],
                            name='entity_embeddings_imaginary' + addendum,
                            trainable=self.ss_trainable)

                if self.config.pretrained_rel == None:
                    self.relation_embeddings_r = tf.Variable(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                        shape=[config.n_relations, config.embedding_dim],
                        ), name='relation_embeddings_real' + addendum, trainable=self.ss_trainable)

                    self.relation_embeddings_i = tf.Variable(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                        shape=[config.n_relations, config.embedding_dim],
                        ), name='relation_embeddings_imaginary' + addendum, trainable=self.ss_trainable)

                else:
                    if self.config.pretrained_rel[-3:] == 'npy':
                        self.relation_embeddings_r = tf.Variable(
                            np.load(self.config.pretrained_rel[:-4] + '_r.npy', allow_pickle=True),
                            name='relation_embeddings_real' + addendum, trainable=self.ss_trainable)
                        self.relation_embeddings_i = tf.Variable(
                            np.load(self.config.pretrained_rel[:-4] + '_i.npy', allow_pickle=True),
                            name='relation_embeddings_imaginary' + addendum, trainable=self.ss_trainable)

                    else:
                        self.relation_embeddings_r = tf.Variable(
                            np.load(self.config.pretrained_rel[:-2] + '_r.p', allow_pickle=True),
                            name='relation_embeddings_real' + addendum, trainable=self.ss_trainable)
                        self.relation_embeddings_i = tf.Variable(
                            np.load(self.config.pretrained_rel[:-2] + '_i.p', allow_pickle=True),
                            name='relation_embeddings_imaginary' + addendum, trainable=self.ss_trainable)

            print(tf.reduce_sum(self.node_embeddings_r))
            print(tf.reduce_sum(self.relation_embeddings_r))

        elif config.folder_suffix == 'pre_mapping':
            self.node_embeddings_r.assign(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                shape=[config.n_nodes, config.embedding_dim]))

            self.node_embeddings_i.assign(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                shape=[config.n_nodes, config.embedding_dim],
            ),)

            self.relation_embeddings_r.assign(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                shape=[config.n_relations, config.embedding_dim],
            ),)

            self.relation_embeddings_i.assign(tf.keras.initializers.GlorotNormal(seed=config.seed)(
                shape=[config.n_relations, config.embedding_dim],
            ),)

        # config.batch_size = temp

    def init_projected_embeddings(self):

        if self.entity_proj_flag:
            a = self.dataset.batch_generator_for_save(0, self.config.n_nodes_total)

            description_embeddings_r = None

            for feed_nodes, ignore in tqdm(a):
                description_temp_r, description_temp_i = self.gather_and_aggregate_words(feed_nodes, initialization=True)
                if not self.config.projection_trainable:
                    description_temp_r, description_temp_i = self.project(description_temp_r, description_temp_i)

                description_temp_r = tf.stop_gradient(description_temp_r).numpy()[:ignore]
                description_temp_i = tf.stop_gradient(description_temp_i).numpy()[:ignore]

                if description_embeddings_r is None:
                    description_embeddings_r = description_temp_r
                    description_embeddings_i = description_temp_i

                else:
                    description_embeddings_r = np.concatenate((description_embeddings_r, description_temp_r))
                    description_embeddings_i = np.concatenate((description_embeddings_i, description_temp_i))

            with tf.device(self.config.emb_device):
                if self.config.projection_trainable:
                    self.aggregated_embeddings_r = tf.Variable(description_embeddings_r, dtype=tf.float32)
                    self.aggregated_embeddings_i = tf.Variable(description_embeddings_i, dtype=tf.float32)

                else:
                    self.projected_embeddings_r = tf.Variable(description_embeddings_r, dtype=tf.float32)
                    self.projected_embeddings_i = tf.Variable(description_embeddings_i, dtype=tf.float32)

            self.entity_proj_flag = False
            del self.project

    def init_entity_embeddings_we(self):

        if self.entity_we_flag:
            with tf.device(self.config.emb_device):
                nodes_r = np.zeros((self.config.n_nodes, self.config.embedding_dim), dtype=np.float32)
                nodes_i = np.zeros((self.config.n_nodes, self.config.embedding_dim), dtype=np.float32)

                if self.config.aggregation_function == 'average':
                    for i in tqdm(range(self.config.n_nodes)):
                        simply_gathered_r, simply_gathered_i = tf.squeeze(
                            self.gather_and_aggregate_words(
                                tf.expand_dims(tf.convert_to_tensor(np.array(i, dtype=np.int32)), 0), train=False))
                        nodes_r[i] = simply_gathered_r
                        nodes_i[i] = simply_gathered_i

                # nodes_i = np.matmul(nodes_i, np.random.normal(size=[self.config.embedding_dim, self.config.embedding_dim]))

                nodes_r /= 1971
                nodes_i /= 1971

                print('glorot', tf.reduce_sum(self.node_embeddings_r))
                print('glorot', tf.reduce_sum(self.node_embeddings_i))

                print('we', np.sum(nodes_r))
                print('we', np.sum(nodes_i))

                self.node_embeddings_r = tf.Variable(nodes_r, dtype=tf.float32)
                self.node_embeddings_i = tf.Variable(nodes_i, dtype=tf.float32)

                self.entity_we_flag = False

    def normalise_tokens(self, entityid2tokenids):

        longest = max([len(x) for x in entityid2tokenids])

        self.max_seq_length = longest
        self.config.max_seq_length = longest

        for i in tqdm(range(len(entityid2tokenids))):
            # print(i, 'iiiiiiiiiiiiiiiiiiiiiiii')
            while len(entityid2tokenids[i]) < longest:
                # print(len(entityid2tokenids[i]), 'ooo')
                if self.config.aggregation_function == 'roberta':
                    entityid2tokenids[i].append(1)

                else:
                    entityid2tokenids[i].append(0)

        tokens_new = np.array(entityid2tokenids, dtype=np.int32)

        return tokens_new

    # def project(self, features_r, features_i):
    #
    #     projected_features_r, projected_features_i = None, None
    #
    #     if self.config.project == 'affine':
    #         projected_features_r = Dense(self.config.embedding_dim)(features_r)
    #         projected_features_i = Dense(self.config.embedding_dim)(features_i)
    #
    #     if self.config.project == 'linear':
    #         projected_features_r = Dense(self.config.embedding_dim)(features_r, use_bias=False)
    #         projected_features_i = Dense(self.config.embedding_dim)(features_i, use_bias=False)
    #
    #     if self.config.project == 'mlp':
    #
    #         #TODO vk modify this, embedding_dim is wrong
    #         features_r = tf.reshape(features_r, (self.config.batch_size, self.config.embedding_dim))
    #         features_i = tf.reshape(features_i, (self.config.batch_size, self.config.embedding_dim))
    #
    #         r1 = Dense(700, activation='tanh')(features_r)
    #         projected_features_r = Dense(300)(r1)
    #
    #         i1 = Dense(700, activation='tanh')(features_i)
    #         projected_features_i = Dense(300)(i1)
    #
    #     return projected_features_r, projected_features_i

    def simply_project(self, nodes_to_project):
        self.zeroshot_node_embeddings = None
        self.projection_embeddings = None

        # nodes_to_project = tf.Tensor(nodes_to_project)

        self.simply_gathered_r, self.simply_gathered_i = self.gather_and_aggregate_words(nodes_to_project,
                                                                                         train=False)

        if self.config.project == 'combine':

            if nodes_to_project[0] >= self.config.n_nodes:
                return tf.convert_to_tensor(np.random.normal(size=(self.config.batch_size, self.config.embedding_dim)).astype(np.float32)), \
            tf.convert_to_tensor(np.random.normal(size=(self.config.batch_size, self.config.embedding_dim)).astype(np.float32))

            n_r = tf.gather(self.node_embeddings_r, nodes_to_project)
            n_i = tf.gather(self.node_embeddings_i, nodes_to_project)

            d_r = self.simply_gathered_r
            d_i = self.simply_gathered_i

            self.simply_projected_r, self.simply_projected_i = self.project(d_r, d_i, n_r, n_i)

        else:

            if not self.config.pretrained_projn:
                self.simply_projected_r, self.simply_projected_i = self.project(self.simply_gathered_r, self.simply_gathered_i)
            else:
                self.simply_projected_r, self.simply_projected_i = self.simply_gathered_r, self.simply_gathered_i

            if self.config.normalize_embed:
                self.simply_projected_r = tf.nn.l2_normalize(self.simply_projected_r, dim=1)
                self.simply_projected_i = tf.nn.l2_normalize(self.simply_projected_i, dim=1)

        return self.simply_projected_r, self.simply_projected_i

    @conditional_tfunction
    def call(self, positive_samples, negative_samples):

        self.pos_samples = positive_samples
        self.neg_samples = negative_samples

        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

        if self.config.normalize_embed:
            self.node_embeddings_r = tf.nn.l2_normalize(self.node_embeddings_r, dim=1)
            self.node_embeddings_i = tf.nn.l2_normalize(self.node_embeddings_i, dim=1)

            self.relation_embeddings_r = tf.nn.l2_normalize(self.relation_embeddings_r, dim=1)
            self.relation_embeddings_i = tf.nn.l2_normalize(self.relation_embeddings_i, dim=1)
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

        self.optimizer()
        return self.cost, self.projection_loss, self.loss_ss, self.loss_sd, self.loss_ds, self.loss_dd

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

    def bilstm(self, node_token_embeddings, node_masks, reuse=None, name_to_append=''):

        # answer = tf.zeros((self.config.batch_size, self.config.embedding_dim))
        #
        # answer = tf.Print(answer, ['node token embeddings', node_token_embeddings, 'shape', tf.shape(node_token_embeddings)], summarize=100)

        node_token_embeddings = tf.reshape(node_token_embeddings,
                                           (self.config.batch_size, self.max_seq_length, self.config.embedding_dim))

        node_masks = tf.reshape(node_masks, (self.config.batch_size, -1))
        n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

        # with tf.variable_scope('LSTM_encoding', reuse=True):

        embs_masked = Masking()(node_token_embeddings)

        with tf.variable_scope('description_lstm' + name_to_append, reuse=reuse):
            print('once')

            # lstm_out = Bidirectional(LSTM(450, activation='tanh', return_sequences=True))(embs_masked)
            # lstm_out = LSTM(450, activation='tanh', return_sequences=True)(embs_masked)
            lstm_forward = LSTM(450, activation='tanh', return_sequences=True)(embs_masked)
            # lstm_backward

            lstm_out = tf.reduce_sum(lstm_out, axis=1) / n_tokens

            lstm_out = Dense(300)(lstm_out)

            return lstm_out

    def lstm(self, node_token_embeddings, node_masks, reuse=None, name_to_append=''):

        # answer = tf.zeros((self.config.batch_size, self.config.embedding_dim))
        #
        # answer = tf.Print(answer, ['node token embeddings', node_token_embeddings, 'shape', tf.shape(node_token_embeddings)], summarize=100)

        node_token_embeddings = tf.reshape(node_token_embeddings,
                                           (self.config.batch_size, self.max_seq_length, self.config.embedding_dim))

        node_masks = tf.reshape(node_masks, (self.config.batch_size, -1))
        n_tokens = tf.expand_dims(tf.reduce_sum(node_masks, axis=1), 1)

        # with tf.variable_scope('LSTM_encoding', reuse=True):

        embs_masked = Masking()(node_token_embeddings)

        with tf.variable_scope('description_lstm' + name_to_append, reuse=reuse):
            print('once')

            # lstm_out = Bidirectional(LSTM(450, activation='tanh', return_sequences=True))(embs_masked)
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



    # def aggregate(self, node_token_embeddings, node_masks, train):
    #
    #     # node_token_embeddings = tf.Print(node_token_embeddings, ['ADFDFDFFF', tf.shape(node_token_embeddings)])
    #     if self.config.toggle:
    #         if train:
    #             return self.sum(node_token_embeddings, node_masks, divide_by=6.19)
    #
    #         else:
    #             a = self.average(node_token_embeddings, node_masks)
    #             return a, a
    #
    #     else:
    #         if self.config.aggregation_function == 'attention':
    #             a = self.attention(node_token_embeddings, node_masks, reuse=self.global_flag)
    #             if self.global_flag is None:
    #                 self.global_flag = True
    #             return a, a
    #
    #         if self.config.aggregation_function == 'average':
    #             a = self.average(node_token_embeddings, node_masks)
    #             return a, a
    #
    #         if self.config.aggregation_function == 'sum':
    #             a = self.sum(node_token_embeddings, node_masks)
    #             return a, a
    #
    #         if self.config.aggregation_function == 'bilstm':
    #
    #             if self.config.separate_aggregation:
    #
    #                 a_r = self.bilstm(node_token_embeddings, node_masks, reuse=self.global_flag_lstm, name_to_append='_r')
    #                 a_i = self.bilstm(node_token_embeddings, node_masks, reuse=self.global_flag_lstm, name_to_append='_i')
    #                 if self.global_flag_lstm is None:
    #                     self.global_flag_lstm = True
    #                 return a_r, a_i
    #
    #             else:
    #                 a = self.bilstm(node_token_embeddings, node_masks, reuse=self.global_flag_lstm)
    #                 if self.global_flag_lstm is None:
    #                     self.global_flag_lstm = True
    #                 return a, a
    #
    #         if self.config.aggregation_function == 'lstm':
    #
    #             if self.config.separate_aggregation:
    #
    #                 a_r = self.lstm(node_token_embeddings, node_masks, reuse=self.global_flag_lstm, name_to_append='_r')
    #                 a_i = self.lstm(node_token_embeddings, node_masks, reuse=self.global_flag_lstm, name_to_append='_i')
    #                 if self.global_flag_lstm is None:
    #                     self.global_flag_lstm = True
    #                 return a_r, a_i
    #
    #             else:
    #                 a = self.lstm(node_token_embeddings, node_masks, reuse=self.global_flag_lstm)
    #                 if self.global_flag_lstm is None:
    #                     self.global_flag_lstm = True
    #                 return a, a
    #
    #         if self.config.aggregation_function == 'cnn':
    #             a = self.cnn( node_token_embeddings, node_masks)
    #             return a, a

    def gather_and_aggregate_words(self, indices, train=True, initialization=False):

        indices = tf.convert_to_tensor(indices)

        node_tokens = tf.gather(self.entityid2tokenids_tf, indices, axis=0)
        node_masks = tf.gather(self.masks_tf, indices, axis=0)

        if initialization:
            if self.config.aggregation_function != 'roberta':
                node_token_embeddings = self.word_embeddings_tf(node_tokens)

                node_embeddings_r, node_embeddings_i = self.aggregate(node_token_embeddings, node_masks, train)

            else:
                node_embeddings_r, node_embeddings_i = self.roberta(node_tokens, node_masks, training=train)

        else:
            if self.config.aggregation_function != 'roberta' and not self.config.pretrained_projn:
                node_token_embeddings = self.word_embeddings_tf(node_tokens)

                node_embeddings_r, node_embeddings_i = self.aggregate(node_token_embeddings, node_masks, train)

            elif self.config.pretrained_projn:
                if self.config.projection_trainable:
                    node_embeddings_r = tf.gather(self.aggregated_embeddings_r, indices, axis=0)
                    node_embeddings_i = tf.gather(self.aggregated_embeddings_i, indices, axis=0)

                else:
                    node_embeddings_r = tf.gather(self.projected_embeddings_r, indices, axis=0)
                    node_embeddings_i = tf.gather(self.projected_embeddings_i, indices, axis=0)

            else:
                node_embeddings_r, node_embeddings_i = self.roberta(node_tokens, node_masks, training=train)

        return node_embeddings_r, node_embeddings_i

    def predict(self, inputs, pos=False):
        src_indices = tf.squeeze(gather_cols(tf.convert_to_tensor(inputs), tf.convert_to_tensor([0])))
        rel_indices = tf.squeeze(gather_cols(tf.convert_to_tensor(inputs), tf.convert_to_tensor([1])))
        tar_indices = tf.squeeze(gather_cols(tf.convert_to_tensor(inputs), tf.convert_to_tensor([2])))


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

        if self.config.zeroshot:

            if self.config.project == 'combine':
                src_gathered_r, src_gathered_i = self.gather_and_aggregate_words(src_indices, train=True)

                tar_gathered_r, tar_gathered_i = self.gather_and_aggregate_words(tar_indices, train=True)

                if not self.config.pretrained_projn or self.config.projection_trainable:
                    self.src_desc_embed_r, self.src_desc_embed_i = \
                        self.project(src_gathered_r, src_gathered_i,
                                     self.src_struc_embed_r, self.src_struc_embed_i, train=True)
                    self.tar_desc_embed_r, self.tar_desc_embed_i = \
                        self.project(tar_gathered_r, tar_gathered_i, self.tar_struc_embed_r,
                                     self.tar_struc_embed_i, train=True)

                else:
                    raise ValueError

            else:

                src_gathered_r, src_gathered_i = self.gather_and_aggregate_words(src_indices, train=True)

                tar_gathered_r, tar_gathered_i = self.gather_and_aggregate_words(tar_indices, train=True)

                if not self.config.pretrained_projn or self.config.projection_trainable:
                    self.src_desc_embed_r, self.src_desc_embed_i = self.project(src_gathered_r, src_gathered_i,
                                                                                train=True)
                    self.tar_desc_embed_r, self.tar_desc_embed_i = self.project(tar_gathered_r, tar_gathered_i,
                                                                                train=True)

                else:
                    self.src_desc_embed_r, self.src_desc_embed_i = src_gathered_r, src_gathered_i
                    self.tar_desc_embed_r, self.tar_desc_embed_i = tar_gathered_r, tar_gathered_i

        if self.config.decoder == 'complex':
            if self.config.loss1:
                if self.config.project != 'combine':
                    preds_ss = self.complex_formulation(self.src_struc_embed_r, self.src_struc_embed_i, rel_struc_embed_r, rel_struc_embed_i,\
                                                       self.tar_struc_embed_r, self.tar_struc_embed_i)

                else:
                    preds_ss = self.complex_formulation(self.src_desc_embed_r, self.src_desc_embed_i, rel_struc_embed_r, rel_struc_embed_i,\
                                                       self.tar_desc_embed_r, self.tar_desc_embed_i)

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
        # if self.config.separate_projection:
        #     self.opt = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate, initial_accumulator_value=1e-20)
        #     self.opt_proj = tf.train.AdamOptimizer(learning_rate=0.001)
        #
        #     self.opt_op = self.opt.minimize(self.cost)
        #     self.opt_op_proj = self.opt_proj.minimize(self.projection_loss)
        # # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        #
        # else:
        #     self.opt = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate, initial_accumulator_value=1e-20)
        #     self.opt_op = self.opt.minimize(self.cost)


    def complex_reduce(self, pos, neg, reduce=True):

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

        if reduce:
            loss = ( self.reduce_func(softplus(-pos)) + self.reduce_func(softplus(neg)) ) / 2
        else:
            loss =  ( softplus(-pos) + softplus(neg) ) / 2

        # print(loss)

        # g = tf.gradients(loss, self.src_struc_embed_r)[0]

        # print(g)
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

                # if self.config.train_intermediate:


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
                self.ploss_pre_src = tf.sqrt(tf.reduce_sum(tf.square(self.pos_src_struc_embed_r - self.pos_src_desc_embed_r), axis=1))\
                                  + tf.sqrt(tf.reduce_sum(tf.square(self.pos_src_struc_embed_i - self.pos_src_desc_embed_i),
                                  axis=1))

                self.ploss_pre_tar = tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed_r - self.pos_tar_desc_embed_r), axis=1))\
                                  + tf.sqrt(tf.reduce_sum(tf.square(self.pos_tar_struc_embed_i - self.pos_tar_desc_embed_i),
                                  axis=1))

                if self.config.weight_projections:
                    # print(self.pos_ss)
                    # self.pos_ss = tf.Print(self.pos_ss, ['pos_ss ',self.pos_ss], first_n=1e6)
                    # print(self.pos_ss)
                    # self.ploss_pre_src = tf.Print(self.ploss_pre_src, ['ploss_pre', self.ploss_pre_src], first_n=1e6)

                    copy = tf.identity(1 / self.pos_ss)

                    self.ploss_pre_src = self.ploss_pre_src * tf.nn.sigmoid(copy)
                    self.ploss_pre_tar = self.ploss_pre_tar * tf.nn.sigmoid(copy)

                projection_loss_source = self.reduce_func(self.ploss_pre_src)
                projection_loss_target = self.reduce_func(self.ploss_pre_tar)

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

        # self.emb_regularizer = ( self.reduce_func(self.pos_src_struc_embed_r ** 2) + \
        #                        self.reduce_func(self.pos_src_struc_embed_i ** 2) + \
        #                        self.reduce_func(self.pos_tar_struc_embed_r ** 2) + \
        #                        self.reduce_func(self.pos_tar_struc_embed_i ** 2) + \
        #                        self.reduce_func(self.pos_src ** 2) + \
        #                        self.reduce_func(self.pos_src_struc_embed_r ** 2) ) / 6

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



