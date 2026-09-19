import torch
import torch.nn as nn
import logging, os
import time
from tqdm import tqdm
import random
# from prettytable import PrettyTable
from torch.nn.init import xavier_normal_, uniform_
from torch.nn import Parameter
import numpy as np
from copy import deepcopy
# import torch_geometric.utils.scatter as scatter
from collections import defaultdict
import torch.nn.functional as F
from datasets import load_dataset


import time
import functools

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 1 == 1:
            result = func(*args, **kwargs)
            return result
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            try:
                class_name = args[0].__class__.__name__
                print(f"Method '{class_name}.{func.__name__}' executed in {execution_time} seconds.")
            except:
                print(f"Method '{func.__name__}' executed in {execution_time} seconds.")
            return result
    return wrapper


def same_seeds(seed):
    '''Set seed for reproduction'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_param(shape):
    '''create learnable parameters'''
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def retype_parameters(args):
    '''
    '''
    args.learning_rate = float(args.learning_rate)
    args.batch_size = int(args.batch_size)
    args.neg_ratio = int(args.neg_ratio)
    args.margin = float(args.margin)
    args.gpu = int(args.gpu)
    try:
        args.ea_expand_training = True if args.ea_expand_training == 'True' else False
    except:
        raise('wrong_expand:', args.ea_expand_training)

    args.seed = int(args.seed)
    args.emb_dim = int(args.emb_dim)
    args.use_known_alignment = True if args.use_known_alignment == 'True' else False
    args.use_pseudo_alignment = True if args.use_pseudo_alignment == 'True' else False
    args.use_pseudo_align_threshold = True if args.use_pseudo_align_threshold == 'True' else False
    args.use_inter_triples = True if args.use_inter_triples == 'True' else False
    args.use_triple_infonce = True if args.use_triple_infonce == 'True' else False
    args.use_pseudo_triple_infonce = True if args.use_pseudo_triple_infonce == 'True' else False
    args.use_known_expand_triple_loss = True if args.use_known_expand_triple_loss == 'True' else False
    args.use_pseudo_triple_loss = True if args.use_pseudo_triple_loss == 'True' else False
    args.use_entropy_selection = True if args.use_entropy_selection == 'True' else False



def load_fact(ds, order='hrt'):
    '''
    Load (sub, rel, obj) from file 'path'.
    :param path: xxx.txt
    :return: fact list: [(s, r, o)]
    '''
    facts = []
    # with open(path, 'r', encoding='utf-8') as f:
    for line in ds['text']:
        line = line.strip().split()
        if order=='hrt':
            s, r, o = line[0], line[1], line[2]
        elif order == 'htr':
            s, o, r = line[0], line[1], line[2]
        else:
            raise('wrong order')
        facts.append((s, r, o))
    return facts


def load_align_pair(path, file_name):
    '''
    Load (sub, rel, obj) from file 'path'.
    :param path: xxx.txt
    :return: fact list: [(s, r, o)]
    '''
    'known_shared_entities.txt'
    pairs = []
    ds = load_dataset(path+'cross', data_files=file_name)['train']['text']

    for line in ds:
        line = line.strip().split()
        e1, e2 = line[0], line[1]
        pairs.append((e1, e2))
    return pairs


def build_edge_index(s, o):
    '''build edge_index using subject and object entity'''
    index = [s + o, o + s]
    return torch.LongTensor(index)


def init_param(model):
    for name, param in model.named_parameters():
        print(name, param.shape)
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            try:
                nn.init.xavier_normal_(param)
            except:
                nn.init.constant_(param, 0.0)


def merge_dicts(dict1, dict2):
    merged_dict = {**dict1, **dict2}  
    return merged_dict


def rel2other_KG(rel, mid):
    if rel > mid:
        return rel-mid
    else:
        return rel


def create_batches(dataset, batch_size, according_to_relations=False):
    if not according_to_relations:
        batches = []
        num_batches = len(dataset) // batch_size + 1
        for i in range(num_batches):
            batches.append(dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))])
    else:
        relation_samples_ = {}
        for triplet in dataset:
            relation = triplet[1]
            if relation not in relation_samples_:
                relation_samples_[relation] = []
            relation_samples_[relation].append(triplet)
        relation_samples = [val for key, val in relation_samples_.items()]
        batches = []
        for samples in relation_samples:
            random.shuffle(samples)
            num_batches = len(samples) // batch_size
            for i in range(num_batches):
                batch = samples[i * batch_size: (i + 1) * batch_size]
                batches.append(batch)

        remaining_samples = []
        for samples in relation_samples.values():
            remaining_samples.extend(samples[num_batches * batch_size:])

        random.shuffle(remaining_samples)

        while len(remaining_samples) > 0:
            sub_batch_size = min(batch_size, len(remaining_samples))
            sub_batch = remaining_samples[:sub_batch_size]
            batches.append(sub_batch)
            remaining_samples = remaining_samples[sub_batch_size:]

    return batches