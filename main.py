import time
# from functools import _lru_cache_wrapper

import os
import sys
from os import path
from tqdm import tqdm, tqdm_notebook

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['AUTOGRAPH_VERBOSITY'] = "10"

t_first = time.time()

from comet_ml import Experiment
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers import Adagrad
# from tensorflow.compat.v1.train import AdagradOptimizer
from tensorflow.keras.optimizers import Adagrad
import numpy as np

# from dataset_old import Dataset
from dataset import Dataset
from parser import Parser
import re
from config import Config
from tensorflow_addons.optimizers import AdamW
from transformers.optimization_tf import AdamWeightDecay
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import contextlib

@contextlib.contextmanager
def dummy_context_mgr():
    yield None
# from transformers.optimization import AdamW
# from transformers import get_linear_schedule_with_warmup

from models.model import ComplexModel
from models.dismult_model import TranseModel


strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])



with open('parser.py') as f:
    a = f.read()
    experiment.log_text(a)
    experiment.log_text('cool')

# experiment.set_name('cool')

experiment.log_asset('parser.py')

os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '750'

print('something is happening')

# Load and preprocess data
args = Parser().get_parser().parse_args()
config = Config(args)


def make_optimizer():
    if config.optimizer == 'adam':
        if config.adam_epsilon is not None:
            optimizer = Adam(learning_rate=config.learning_rate, epsilon=config.adam_epsilon)
        else:
            optimizer = Adam(learning_rate=config.learning_rate)
    elif config.optimizer == 'adamw':
        if config.adam_epsilon is not None:
            optimizer = AdamW(learning_rate=get_decayed_lr(step), epsilon=config.adam_epsilon, weight_decay=0)
        else:
            optimizer = AdamW(learning_rate=get_decayed_lr(step), weight_decay=0)
        # if config.adam_epsilon is not None:
        #     optimizer = AdamW(learning_rate=config.learning_rate, epsilon=config.adam_epsilon, weight_decay=0)
        # else:
        #     optimizer = AdamW(learning_rate=config.learning_rate, weight_decay=0)
    elif config.optimizer == 'adagrad':
        if config.adagrad_acc is not None:
            optimizer = Adagrad(learning_rate=config.learning_rate, initial_accumulator_value=config.adagrad_acc)
        else:
            optimizer = Adagrad(learning_rate=config.learning_rate)

    return optimizer

def save():
    print('saving current model ..')

    t0 = time.time()
    manager.save()

    print('model saved in %.3f seconds'% (time.time() - t0))

    def clean(path, max_to_keep):
        files = os.listdir(path)

        q1 = re.compile('epoch[0-9]*_')
        q2 = re.compile('h[0-9]*_')
        epochs_saved = set()

        for value in files:
            g1 = q1.match(value)

            if g1:
                g2 = q2.search(g1.group(0))

                epochs_saved.add(int(g2.group(0)[1:-1]))

        sorted_epochs = sorted(epochs_saved)
        to_remove = sorted_epochs[:-max_to_keep]

        to_remove = list(map(str, to_remove))
        # to_remove = [f'h{x}_' for x in to_remove]
        to_remove = ['h{}_'.format(x) for x in to_remove]

        for remove_string in to_remove:

            for file_name in files:
                file_path = os.path.join(path, file_name)

                if remove_string in file_name:
                    os.remove(file_path)

    max_to_keep = config.embeddings_max_to_keep

    path = config.paths['experiment']

    clean(path, max_to_keep)

    # '''

    t0 = time.time()
    save_prefix = os.path.join(config.paths['experiment'], 'epoch' + str(epoch) + '_')

    if config.decoder == 'complex':
        node_embeddings_r, node_embeddings_i, relation_embeddings_r, relation_embeddings_i = \
            (x.numpy() for x in (
                model.node_embeddings_r, model.node_embeddings_i, model.relation_embeddings_r,
                model.relation_embeddings_i))

    else:
        node_embeddings, relation_embeddings = \
            (x.numpy() for x in (
                model.node_embeddings, model.relation_embeddings))

    feed_dict = {}

    if config.zeroshot:

        if config.decoder == 'complex':
            zeroshot_node_embeddings_r = None

            a = dataset.batch_generator_for_save(config.n_nodes, config.n_nodes_total)

            for feed_nodes, ignore in a:
                projected_temp_r, projected_temp_i = model.simply_project(feed_nodes)
                projected_temp_r = projected_temp_r.numpy()[:ignore]
                projected_temp_i = projected_temp_i.numpy()[:ignore]

                if zeroshot_node_embeddings_r is None:
                    zeroshot_node_embeddings_r = projected_temp_r
                    zeroshot_node_embeddings_i = projected_temp_i

                else:
                    zeroshot_node_embeddings_r = np.concatenate((zeroshot_node_embeddings_r, projected_temp_r))
                    zeroshot_node_embeddings_i = np.concatenate((zeroshot_node_embeddings_i, projected_temp_i))

            print(node_embeddings_r.shape)
            print(zeroshot_node_embeddings_r.shape)

            np.save(save_prefix + 'node_embeddings_r.npy',
                    np.concatenate((node_embeddings_r, zeroshot_node_embeddings_r)))

            np.save(save_prefix + 'node_embeddings_i.npy',
                    np.concatenate((node_embeddings_i, zeroshot_node_embeddings_i)))

        else:
            zeroshot_node_embeddings = None

            a = dataset.batch_generator_for_save(config.n_nodes, config.n_nodes_total)

            for feed_nodes, ignore in a:
                projected_temp = model.simply_project(feed_nodes)
                projected_temp = projected_temp.numpy()[:ignore]

                if zeroshot_node_embeddings is None:
                    zeroshot_node_embeddings = projected_temp
                else:
                    zeroshot_node_embeddings = np.concatenate((zeroshot_node_embeddings, projected_temp))

            print(node_embeddings.shape)
            print(zeroshot_node_embeddings.shape)

            np.save(save_prefix + 'node_embeddings.npy',
                    np.concatenate((node_embeddings, zeroshot_node_embeddings)))

    else:
        if config.decoder == 'complex':
            np.save(save_prefix + 'node_embeddings_r.npy', node_embeddings_r)
            np.save(save_prefix + 'node_embeddings_i.npy', node_embeddings_i)
        else:
            np.save(save_prefix + 'node_embeddings.npy', node_embeddings)

    if config.decoder == 'complex':
        np.save(save_prefix + 'relation_embeddings_r.npy', relation_embeddings_r)
        np.save(save_prefix + 'relation_embeddings_i.npy', relation_embeddings_i)

    else:
        np.save(save_prefix + 'relation_embeddings.npy', relation_embeddings)

    print('embeddings saved in %.3f seconds ..' % (time.time() - t0))

def evaluate(splits):

    t0 = time.time()
    time_for_graph = time.time()

    global first_eval, best_mrr

    if config.project == 'combine':
        relation_embeddings_r, relation_embeddings_i = (x.numpy() for x in (model.relation_embeddings_r,
                model.relation_embeddings_i))

        a = dataset.batch_generator_for_save(0, config.n_nodes)

        node_embeddings_r = None

        for feed_nodes, ignore in tqdm(a):

            projected_temp_r, projected_temp_i = model.simply_project(feed_nodes)
            projected_temp_r = projected_temp_r.numpy()[:ignore]
            projected_temp_i = projected_temp_i.numpy()[:ignore]

            if node_embeddings_r is None:
                node_embeddings_r = projected_temp_r
                node_embeddings_i = projected_temp_i

            else:
                node_embeddings_r = np.concatenate((node_embeddings_r, projected_temp_r))
                node_embeddings_i = np.concatenate((node_embeddings_i, projected_temp_i))

    else:
        if config.decoder == 'complex':
            node_embeddings_r, node_embeddings_i, relation_embeddings_r, relation_embeddings_i = \
                (x.numpy() for x in (
                    model.node_embeddings_r, model.node_embeddings_i, model.relation_embeddings_r,
                    model.relation_embeddings_i))
        else:
            node_embeddings, relation_embeddings = \
                (x.numpy() for x in (
                    model.node_embeddings, model.relation_embeddings,))

    if config.zeroshot:

        if config.decoder == 'complex':
            # WILL THIS WORK ?
            zeroshot_node_embeddings_r = None

            a = dataset.batch_generator_for_save(config.n_nodes, config.n_nodes_total)

            for feed_nodes, ignore in tqdm(a):

                projected_temp_r, projected_temp_i = model.simply_project(feed_nodes)
                projected_temp_r = projected_temp_r.numpy()[:ignore]
                projected_temp_i = projected_temp_i.numpy()[:ignore]

                if zeroshot_node_embeddings_r is None:
                    zeroshot_node_embeddings_r = projected_temp_r
                    zeroshot_node_embeddings_i = projected_temp_i

                else:
                    zeroshot_node_embeddings_r = np.concatenate((zeroshot_node_embeddings_r, projected_temp_r))
                    zeroshot_node_embeddings_i = np.concatenate((zeroshot_node_embeddings_i, projected_temp_i))


            print(node_embeddings_r.shape)
            print(zeroshot_node_embeddings_r.shape)

            all_nodes_r = np.concatenate((node_embeddings_r, zeroshot_node_embeddings_r))
            all_nodes_i = np.concatenate((node_embeddings_i, zeroshot_node_embeddings_i))

        else:
            zeroshot_node_embeddings = None

            a = dataset.batch_generator_for_save(config.n_nodes, config.n_nodes_total)

            for feed_nodes, ignore in tqdm(a):

                projected_temp = model.simply_project(feed_nodes)
                projected_temp = projected_temp.numpy()[:ignore]

                if zeroshot_node_embeddings is None:
                    zeroshot_node_embeddings = projected_temp
                else:
                    zeroshot_node_embeddings = np.concatenate((zeroshot_node_embeddings, projected_temp))

            print(node_embeddings.shape)
            print(zeroshot_node_embeddings.shape)

            all_nodes = np.concatenate((node_embeddings, zeroshot_node_embeddings))

        ############################################################################

        def log_split(split):
            # print(split)

            if config.decoder == 'complex':
                raw_metrics, filtered_metrics, target_filtered_metrics = \
                    model.evaluate((all_nodes_r, all_nodes_i), (relation_embeddings_r, relation_embeddings_i),
                                   split=split, predict='tails')
            else:
                raw_metrics, filtered_metrics, target_filtered_metrics = \
                    model.evaluate(all_nodes,  relation_embeddings,
                               split=split, predict='tails')

            for metrics, name in zip((raw_metrics, filtered_metrics, target_filtered_metrics),
                                     ('raw_', 'filt_', 'tfilt_')):
                for m_name, m_value in metrics.items():
                    experiment.log_metric(split + '_' + name + m_name, m_value, step=step.numpy())

            if split !='test' and split != 'valid':
                return target_filtered_metrics['rec_rank']

            else:
                return filtered_metrics['rec_rank']

        ##################################################################################

        # if config.n_test_zeroshot != config.n_val_zeroshot:
        #     log_split('valid_zero')

        for s in splits:

            mrr = log_split(s)

            end_time = time.time()
            print('=================== time for graph is \n', str(end_time - time_for_graph), '\n=====================')

            if s == config.split_to_save_on:
                if not first_eval:
                    if mrr > best_mrr:
                        best_mrr = mrr


                        save()

                else:
                    first_eval = False
                    best_mrr = mrr

        print("time taken for evaluation is %.3f seconds"% (time.time() - t0))

def save_all_projected_embeddings(project=False):
    step = tf.Variable(1, name='global_step')

    print('loading data ..')
    t0 = time.time()
    dataset = Dataset(config=config)
    print("data loaded successfully. took %.2f seconds .." % (time.time() - t0))

    if not config.use_static_features:

        if config.decoder in ['dismult', 'transe']:
            we_dimensions = 768 if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[1]
            vocab_size = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[0]
            we_np = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np

            model = TranseModel(config=config, dataset=dataset,
                                vocab_size=vocab_size,
                                word_embeddings_dimension=we_dimensions,
                                # word_embeddings_tf=dataset.word_embeddings_tf,
                                entityid2tokenids=dataset.entityid2tokenids, node_dict=dataset.node_dict,
                                node_dict_zeroshot=dataset.node_dict_zeroshot,
                                word_embeddings=we_np)

        else:

            we_dimensions = 768 if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[1]
            vocab_size = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[0]
            we_np = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np

            model = ComplexModel(config=config, dataset=dataset,
                                 vocab_size=vocab_size,
                                 word_embeddings_dimension=we_dimensions,
                                 # word_embeddings_tf=dataset.word_embeddings_tf,
                                 entityid2tokenids=dataset.entityid2tokenids, node_dict=dataset.node_dict,
                                 node_dict_zeroshot=dataset.node_dict_zeroshot,
                                 word_embeddings=we_np)
    else:
        if config.decoder != 'complex':
            model = CombinationModel(config=config, placeholders=dataset.placeholders, node_dict=dataset.node_dict)

        else:
            model = ComplexModel(config=config, placeholders=dataset.placeholders, node_dict=dataset.node_dict)

    optimizer = make_optimizer()

    for b, s in dataset.batch_generator():
        a = model(b, s)
        break

    ckpt = tf.train.Checkpoint(step=step, net=model, optimizer=optimizer)

    if config.restore:
        restore_manager = tf.train.CheckpointManager(ckpt, config.paths['restore'], max_to_keep=1)

        print(restore_manager.latest_checkpoint)
        print(type(restore_manager.latest_checkpoint))

        ckpt.restore(restore_manager.latest_checkpoint)

    zeroshot_node_embeddings_r = None

    a = dataset.batch_generator_for_save(0, config.n_nodes_total)

    for feed_nodes, ignore in tqdm(a):

        if project:
            projected_temp_r, projected_temp_i = model.simply_project(feed_nodes)
        else:
            projected_temp_r, projected_temp_i = model.gather_and_aggregate_words(feed_nodes)

        projected_temp_r = projected_temp_r.numpy()[:ignore]
        projected_temp_i = projected_temp_i.numpy()[:ignore]

        if zeroshot_node_embeddings_r is None:
            zeroshot_node_embeddings_r = projected_temp_r
            zeroshot_node_embeddings_i = projected_temp_i

        else:
            zeroshot_node_embeddings_r = np.concatenate((zeroshot_node_embeddings_r, projected_temp_r))
            zeroshot_node_embeddings_i = np.concatenate((zeroshot_node_embeddings_i, projected_temp_i))

    print(zeroshot_node_embeddings_r.shape)

    save_prefix = os.path.join(config.paths['experiment'], 'all_embeddings' + '_')
    np.save(save_prefix + 'r', zeroshot_node_embeddings_r)
    np.save(save_prefix + 'i', zeroshot_node_embeddings_i)


# def main():
# with strategy.scope() if config.multi_gpu else dummy_context_mgr():
#
#     print(1)
#
#     print(config.learning_rate)
#     print(config.projection_weight)
#     print(config.paths['experiment'])
#     print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
#     os.system('cp parser.py ' + config.paths['experiment'])
#
#     np.random.seed(config.seed)
#     tf.random.set_seed(config.seed)
#
#     print(2)
#     print('loading data ..')
#     t0 = time.time()
#     dataset = Dataset(config=config)
#     print("data loaded successfully. took %.2f seconds .." % (time.time() - t0))
#
#     print("building the multi-relational model ...")
#     t0 = time.time()
#
#     if not config.use_static_features:
#
#         if config.decoder in ['dismult', 'transe']:
#             we_dimensions = 768 if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[1]
#             vocab_size = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[0]
#             we_np = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np
#
#             model = TranseModel(config=config, dataset=dataset,
#                                  vocab_size=vocab_size,
#                                  word_embeddings_dimension=we_dimensions,
#                                  # word_embeddings_tf=dataset.word_embeddings_tf,
#                                  entityid2tokenids=dataset.entityid2tokenids, node_dict=dataset.node_dict,
#                                  node_dict_zeroshot=dataset.node_dict_zeroshot,
#                                  word_embeddings=we_np)
#
#         else:
#
#             we_dimensions = 768 if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[1]
#             vocab_size = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np.shape[0]
#             we_np = None if config.aggregation_function == 'roberta' else dataset.word_embeddings_np
#
#             model = ComplexModel(config=config, dataset=dataset,
#                                  vocab_size=vocab_size,
#                                  word_embeddings_dimension=we_dimensions,
#                                  # word_embeddings_tf=dataset.word_embeddings_tf,
#                                  entityid2tokenids=dataset.entityid2tokenids, node_dict=dataset.node_dict,
#                                  node_dict_zeroshot=dataset.node_dict_zeroshot,
#                                  word_embeddings=we_np)
#     else:
#         if config.decoder != 'complex':
#             model = CombinationModel(config=config, placeholders=dataset.placeholders, node_dict=dataset.node_dict)
#
#         else:
#             model = ComplexModel(config=config, placeholders=dataset.placeholders, node_dict=dataset.node_dict)
#
#     print("model built successfully. took %.2f seconds .." % (time.time() - t0))
#
#     # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#     # sess = tf.Session()
#     # saver = tf.train.Saver(max_to_keep=1)
#
#     # if config.restore:
#     #     print('restoring stored session ..')
#     #     t0 = time.time()
#     #     # config.restore = '../Experiments/5|6|9:25:28/FB20k/Default/'
#     #     ckpt = tf.train.get_checkpoint_state(config.paths['restore'])
#     #     saver.restore(sess, ckpt.model_checkpoint_path)
#     #     print('session restored successfully. took %.2f seconds ..' % (time.time() - t0))
#
#     # Train the model
#     print('training started ..')
#
#
#
#     print_every = config.print_every
#     save_every = config.save_every
#
#     print(config.epochs)
#
#     step = tf.Variable(1, name='global_step')
#
#     print('=================')
#
#     vars_to_train = []
#     stop_gradients = []
#
#     cordon = ['layer_' + str(i) for i in range(11)]
#
#     # for b, s in dataset.batch_generator():
#     #     a = model(b, s)
#     #     break
#     with tf.GradientTape() as tape:
#         for b, s in dataset.batch_generator():
#             a = model(b, s)
#             for x in model.trainable_variables:
#
#                 if not config.projection_trainable:
#                     if 'project' in x.name or 'aggregate' in x.name or 'roberta' in x.name or '/embedding/' in x.name:
#                         continue
#
#                     else:
#                         vars_to_train.append(x)
#
#                 else:
#                     if 'roberta' in x.name:
#                         if config.roberta_full:
#                             vars_to_train.append(x)
#                         else:
#                             pass
#
#                     else:
#                         vars_to_train.append(x)
#
#             break
#
#         print(*[str(x.name) + ' ' + str(tf.shape(x)) for x in vars_to_train], sep='\n')
#
#     get_decayed_lr = PolynomialDecay(initial_learning_rate=config.learning_rate,
#                                      decay_steps=100000,
#                                      end_learning_rate=config.learning_rate / 1000)
#
#     optimizer = make_optimizer()
#
#     print('=================')
#
#     experiment.add_tags([config.timestamp, config.folder_suffix, config.dataset_name])
#
#
#
#     ckpt = tf.train.Checkpoint(step=step, net=model, optimizer=optimizer)
#     manager = tf.train.CheckpointManager(ckpt, config.paths['checkpoints'], max_to_keep=1)
#
#     if config.restore:
#         restore_manager = tf.train.CheckpointManager(ckpt, config.paths['restore'], max_to_keep=1)
#         s = restore_manager.latest_checkpoint
#
#         if not config.pretrained_projn:
#             ckpt.restore(restore_manager.latest_checkpoint)
#
#             #TODO restore adam properly
#             if config.restore and 'post_mapping' in config.paths['restore']:
#                 print('\n\n1e-3 ITER 3\n\n')
#                 optimizer = make_optimizer()
#                 step = tf.Variable(1, name='global_step')
#
#         else:
#             print(restore_manager.latest_checkpoint)
#             print(type(restore_manager.latest_checkpoint))
#
#             # if config.folder_suffix != 'pre_mapping':
#             ckpt.restore(restore_manager.latest_checkpoint)
#
#             if config.timestamp == '1e-3_retry'and config.folder_suffix == 'post_mapping':
#                 pass
#             else:
#                 optimizer = make_optimizer()
#
#             step = tf.Variable(1, name='global_step')
#
#             # model.load_weights(restore_manager.latest_checkpoint)
#             # else:
#             #     model.load_weights(restore_manager.latest_checkpoint, by_name=True, skip_mismatch=True)
#
#         ckpt = tf.train.Checkpoint(step=step, net=model, optimizer=optimizer)
#
#         print('restored')
#
#     if config.folder_suffix == 'pre_mapping':
#         print('fffffffffffffffffffff\nfffffffffffffffffffffff\n')
#         model.init_embeddings()
#
#         vars_to_train = []
#         stop_gradients = []
#
#         cordon = ['layer_' + str(i) for i in range(11)]
#
#         # for b, s in dataset.batch_generator():
#         #     a = model(b, s)
#         #     break
#
#         with tf.GradientTape() as tape:
#             for b, s in dataset.batch_generator():
#                 a = model(b, s)
#                 for x in model.trainable_variables:
#
#                     if not config.projection_trainable:
#                         if 'project' in x.name or 'aggregate' in x.name or 'roberta' in x.name or '/embedding/' in x.name:
#                             continue
#
#                         else:
#                             vars_to_train.append(x)
#
#                     else:
#                         if 'roberta' in x.name:
#                             if config.roberta_full:
#                                 vars_to_train.append(x)
#                             else:
#                                 pass
#
#                         else:
#                             vars_to_train.append(x)
#
#                 break
#
#             print(*[str(x.name) + ' ' + str(tf.shape(x)) for x in vars_to_train], sep='\n')
#
#     if config.pretrained_projn and config.loss3:
#         model.init_projected_embeddings()
#
#     first_eval = True
#     best_mrr = -1
#
#     evaluate(['valid_zero'])
#
#     for epoch in tqdm(range(1, config.epochs + 1)):
#         # print(type(epoch))
#         # Construct feed dictionaries
#         # experiment.set_epoch(epoch)
#
#         batches = dataset.batch_generator()
#         itr = 1
#
#         # if epoch == 1:
#         #     if not config.corr:
#         #         evaluate(config.eval_splits)
#
#
#         tar = np.zeros([config.batch_size*100, 300], np.float32)
#         tar_estimate = np.zeros([config.batch_size*100, 300], np.float32)
#
#         src = np.zeros([config.batch_size*100, 300], np.float32)
#         src_estimate = np.zeros([config.batch_size*100, 300], np.float32)
#
#         corr_count = 0
#
#         printflag = True
#         for pos_samples, neg_samples in tqdm(batches, mininterval=30):
#             t0 = time.time()
#
#             # experiment.set_step(step)
#
#             # Training step: run single weight update
#
#             with tf.GradientTape() as tape:
#
#                 if config.corr:
#                     cost, projection_loss, loss_ss, loss_sd, loss_ds, loss_dd, src_t, rel_t, tar_t = model(pos_samples, neg_samples)
#
#                     tar[corr_count: corr_count + config.batch_size] = tar_t
#                     tar_estimate[corr_count: corr_count + config.batch_size] = src_t + rel_t
#
#                     src[corr_count: corr_count + config.batch_size] = src_t
#                     src_estimate[corr_count: corr_count + config.batch_size] = tar_t - rel_t
#
#                     corr_count += config.batch_size
#
#                     if corr_count == config.batch_size * 100:
#
#                         def corr_coef(a, b):
#                                 a = a - np.mean(a, axis=0, keepdims=True)
#                                 b = b - np.mean(b, axis=0, keepdims=True)
#
#                                 cov = a.T @ b
#
#                                 var_a = np.sum(a * a, axis=0, keepdims=True)
#                                 var_b = np.sum(b * b, axis=0, keepdims=True)
#
#                                 var_mat = var_a.T @ var_b
#
#                                 return cov / np.sqrt(var_mat)
#
#                         src_corr_matrix = corr_coef(src, src_estimate)
#                         tar_corr_matrix = corr_coef(tar, tar_estimate)
#
#                         head_corr = np.mean(src_corr_matrix)
#                         tail_corr = np.mean(tar_corr_matrix)
#
#                         if printflag:
#                             print(tar.shape)
#                             print(tar_corr_matrix.shape)
#                             print(head_corr)
#                             print(tail_corr)
#                             printflag = False
#
#                         experiment.log_metric('head_corr', head_corr, step=step.numpy())
#                         experiment.log_metric('tail_corr', tail_corr, step=step.numpy())
#                         experiment.log_metric('head_corr_l2', np.mean(np.abs(src_corr_matrix)), step=step.numpy())
#                         experiment.log_metric('tail_corr_l2', np.mean(np.abs(tar_corr_matrix)), step=step.numpy())
#
#                         corr_count = 0
#
#                         tar *= 0
#                         tar_estimate *= 0
#
#                         src *= 0
#                         src_estimate *= 0
#
#                 else:
#                     cost, projection_loss, loss_ss, loss_sd, loss_ds, loss_dd = model(pos_samples, neg_samples)
#                 grads = tape.gradient(cost, vars_to_train)
#                 # gv = zip(grads, vars_to_train)
#                 gv = [(g, v) for g, v in zip(grads, vars_to_train) if g is not None]
#
#                 # print(optimizer)
#                 # for g,v in gv:
#                 #     print('gradients: ', g, 'variable: ', v)
#                 optimizer.apply_gradients(gv)
#
#             if (itr - 1) % print_every == 0:
#
#                 if config.loss3:
#                     experiment.log_metric('loss_projection', projection_loss, step=step.numpy())
#                 if config.loss1:
#                     experiment.log_metric('loss_ss', loss_ss, step=step.numpy())
#                 if config.loss4:
#                     experiment.log_metric('loss_sd', loss_sd, step=step.numpy())
#                 if config.loss5:
#                     experiment.log_metric('loss_ds', loss_ds, step=step.numpy())
#                 if config.loss2:
#                     experiment.log_metric('loss_dd', loss_dd, step=step.numpy())
#                 # print_summary = 'epoch: %d; itr: %d; time_taken: %.3f seconds; loss: %.3f ' \
#                 #                 % (epoch, itr, time.time() - t0, itr_cost)
#                 #
#                 # losses = ['proj: %.3f' % (projection_loss) if config.zeroshot else '',
#                 #           'ss: %.3f' % (loss_ss) if config.loss1 else '',
#                 #           'dd: %.3f' % (loss_dd) if config.loss2 else '',
#                 #           'sd: %.3f' % (loss_sd) if config.loss4 else '',
#                 #           'ds: %.3f' % (loss_ds) if config.loss5 else '']
#                 #
#                 # print_summary += '( ' + '; '.join([l for l in losses if l != '']) + ' )'
#                 # print(print_summary)
#
#             # print(step)
#             # print(config.eval_every)
#
#             if (step) % config.eval_every == 0 and config.aggregation_function == 'roberta' and not config.pretrained_projn:
#                 evaluate(config.eval_splits)
#
#             # if step % 2000 == 0:
#             #     print(cost)
#
#             itr += 1
#             step.assign_add(1)
#
#         if (epoch - 1) % config.eval_every == 0 and (config.aggregation_function != 'roberta' or config.pretrained_projn):
#             evaluate(config.eval_splits)
#
#         # if epoch % save_every == 0:
#
#
#     print('Optimization finished!')
#     print('time taken :', (time.time() - t_first) / 60)




# main()
save_all_projected_embeddings()
