import numpy as np
from os.path import join as oj
import json

# embs = np.load('/scratch/scratch1/vk/Experiments/FB15k-237-OWE/tedx/actual_tedx_1e-3/epoch951_node_embeddings_r.npy')
# embs = np.load('/scratch/scratch1/vk/Experiments/FB15k-237-OWE/roberta/iter2_real/epoch2_node_embeddings_r.npy')
embs = np.load('/scratch/scratch1/vk/owe_emb_saving_testing/_node_embeddings_r.npy')# FB20k
# embs = np.load('/scratch/scratch1/vk/owe_fb15k_r.npy') # FB15k
dataset = 'FB15k-237-OWE'

dataset_dir = oj('/scratch/scratch1/vk/Datasets', dataset)

eid_to_flag_tag_name = {}

with open(oj(dataset_dir, 'entity2wikidata.json'), 'r') as dfile:
    desc = json.load(dfile)

with open(oj(dataset_dir, 'entity2id.txt')) as f:
    i=0
    for line in f:
        i+=1
        tag, id = line.strip().split()
        id = int(id)
        eid_to_flag_tag_name[id] = {}
        eid_to_flag_tag_name[id]['flag'] = 'closed'
        eid_to_flag_tag_name[id]['tag'] = tag
        eid_to_flag_tag_name[id]['name'] = desc[tag]['label']

    n_closed = i


with open(oj(dataset_dir, 'entity2id_zeroshot.txt')) as f:
    i = 0
    for line in f:
        i += 1
        tag, id = line.strip().split()
        id = int(id)
        eid_to_flag_tag_name[id] = {}
        eid_to_flag_tag_name[id]['flag'] = 'open'
        eid_to_flag_tag_name[id]['tag'] = tag
        eid_to_flag_tag_name[id]['name'] = desc[tag]['label']

    n_open = i

np.savetxt('vecso20.tsv', embs, delimiter='\t')

with open('metao20.tsv', 'w') as f:
    f.write('flag\tname\n')
    for i in range(n_closed + n_open):
        f.write(eid_to_flag_tag_name[i]['flag'] + '\t' + eid_to_flag_tag_name[i]['name'] + '\n')

