from torch.utils.data import Dataset
from src.utils import *
from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
import torch


'''for Cross-KG Link Prediction'''
class CrossLinkPredictionTrainDatasetMarginLoss(Dataset):
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode
        self.original_facts, self.original_is_expand, self.original_is_ea = self.build_facts()
        self.original_is_expand = torch.BoolTensor(self.original_is_expand).reshape(-1, 1)
        self.original_is_ea = torch.BoolTensor(self.original_is_ea).reshape(-1, 1)
        self.original_confidence = torch.ones([len(self.original_facts), 1], dtype=torch.float32)

        self.facts = deepcopy(self.original_facts)
        self.confidence = self.original_confidence.clone()
        self.is_expand = self.original_is_expand.clone()
        self.is_ea = self.original_is_ea.clone()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        '''
        :param idx: idx of the training fact
        :return: a positive facts and its negative facts
        '''
        fact = self.facts[idx]
        conf = self.confidence[idx]
        is_expand = self.is_expand[idx]
        is_ea = self.is_ea[idx]
        
        '''negative sampling'''
        fact, label, subsampling_weight = self.corrupt(fact, is_ea)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        num_samples = len(fact)
        conf_expanded = conf.repeat(num_samples, 1) if conf.dim() > 0 else conf.repeat(num_samples)
        subsampling_weight_expanded = subsampling_weight.repeat(num_samples)
        
        return fact, label, conf_expanded, self.mode, subsampling_weight_expanded, torch.ones(len(fact), dtype=torch.bool)*is_expand, torch.ones(len(fact), dtype=torch.bool)*is_ea
    
    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        conf = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        subsampling_weight = torch.cat([_[4] for _ in data], dim=0)
        
        is_expands = torch.cat([_[5] for _ in data], dim=0)
        is_eas = torch.cat([_[6] for _ in data], dim=0)
        
        return fact[:,0], fact[:,1], fact[:,2], label, conf, mode, subsampling_weight, is_expands, is_eas
    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        head_facts = list()
        is_expands = list()
        is_ea = list()
        all_relation, cross_relation_head, cross_relation_tail = set(), set(), set()
        for source_id, source in self.kg.source.items():
            expand_fact_num = 0
            for fact in source.train:
                s, r, o = fact
                all_relation.add(r)
                head_facts.append((s, r, o))
                is_expands.append(False)
                is_ea.append(False)
                if self.args.ea_expand_training:
                    if s in self.kg.ent2align.keys():
                        cross_relation_head.add(r)
                        head_facts.append((self.kg.ent2align[s], r, o))
                        is_expands.append(True)
                        is_ea.append(False)
                        expand_fact_num += 1
                    if o in self.kg.ent2align.keys():
                        cross_relation_tail.add(r)
                        head_facts.append((s, r, self.kg.ent2align[o]))
                        is_expands.append(True)
                        is_ea.append(False)
                        expand_fact_num += 1
                
        return head_facts, is_expands, is_ea

    def corrupt(self, fact, is_ea):
        s, r, o = fact
        s_temp, o_temp, r_temp = s, o, r
        try:
            subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(o, self.kg.relation2inv[r])]
        except:
            try:
                if hasattr(self.kg, 'ent2align') and s in self.kg.ent2align:
                    subsampling_weight = self.kg.count[(self.kg.ent2align[s], r)] + self.kg.count[(o, self.kg.relation2inv[r])]
                    s_temp = self.kg.ent2align[s]
                elif hasattr(self.kg, 'pseudo_ent2align') and s in self.kg.pseudo_ent2align:
                    subsampling_weight = self.kg.count[(self.kg.pseudo_ent2align[s], r)] + self.kg.count[(o, self.kg.relation2inv[r])]
                    s_temp = self.kg.pseudo_ent2align[s]
                else:
                    raise
            except:
                try:
                    if hasattr(self.kg, 'ent2align') and o in self.kg.ent2align:
                        subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(self.kg.ent2align[o], self.kg.relation2inv[r])]
                        o_temp = self.kg.ent2align[o]
                    elif hasattr(self.kg, 'pseudo_ent2align') and o in self.kg.pseudo_ent2align:
                        subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(self.kg.pseudo_ent2align[o], self.kg.relation2inv[r])]
                        o_temp = self.kg.pseudo_ent2align[o]
                    else:
                        raise
                except:
                    try:
                        try:
                            r_temp = self.same[r]
                            subsampling_weight = self.kg.count[(s, r_temp)] + self.kg.count[(o, self.kg.relation2inv[r_temp])]
                        except:
                            r_temp = self.kg.relation2inv[self.inverse[r]]
                            subsampling_weight = self.kg.count[(s, r_temp)] + self.kg.count[(o, self.kg.relation2inv[r_temp])]
                    except:
                        print(s, r, o)
                        subsampling_weight = 1.0
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        facts = [fact]
        label = [1]
        negative_sample_size = 0
        negative_sample_list = []

        while negative_sample_size < self.args.neg_ratio:
            if self.mode == 'head-batch':
                negative_sample = np.random.randint(0, self.kg.num_ent, self.args.neg_ratio * 2)

                mask = np.in1d(
                    negative_sample,
                    self.kg.sr2o_train[(o_temp, self.kg.relation2inv[r_temp])],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                negative_sample = np.random.randint(0, self.kg.num_ent, self.args.neg_ratio * 2)

                mask = np.in1d(
                    negative_sample,
                    self.kg.sr2o_train[(s_temp, r_temp)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_size += negative_sample.size
            negative_sample_list.append(negative_sample)

        negative_sample = np.concatenate(negative_sample_list)[:self.args.neg_ratio]
        if self.mode == 'head-batch':
            facts.extend([(negative_sample[i], r, o) for i in range(self.args.neg_ratio)])
            label += [-1 for i in range(self.args.neg_ratio)]
        elif self.mode == 'tail-batch':
            facts.extend([(s, r, negative_sample[i]) for i in range(self.args.neg_ratio)])
            label += [-1 for i in range(self.args.neg_ratio)]

        return facts, label, subsampling_weight

    
    def _compute_triple_scores_batch(self, model, triples, batch_size=256):
        device = next(model.parameters()).device
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(triples), batch_size):
                batch_triples = triples[i:i+batch_size]
                
                s_list = [t[0] for t in batch_triples]
                r_list = [t[1] for t in batch_triples]
                o_list = [t[2] for t in batch_triples]
                
                s_tensor = torch.LongTensor(s_list).to(device)
                r_tensor = torch.LongTensor(r_list).to(device)
                o_tensor = torch.LongTensor(o_list).to(device)
                
                mode = 'tail-batch'
                jobs = {
                    'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": s_tensor}, 'mode': mode},
                    'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": r_tensor}, 'mode': mode},
                    'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": o_tensor}, 'mode': mode},
                }
                
                batch_scores = model.forward(jobs=jobs, stage='train', mode=mode, margin=self.args.margin)-self.args.margin
                
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.cpu().numpy()
                    if batch_scores.ndim == 0:
                        scores.append(float(batch_scores))
                    else:
                        scores.extend(batch_scores.flatten().tolist())
                else:
                    scores.append(float(batch_scores))
        
        return scores
    
    def _compute_entity_degree(self, entity):
        out_degree = 0  
        in_degree = 0   
        
        for fact in self.kg.train:
            s, r, o = fact
            if s == entity:
                out_degree += 1
        
        for fact in self.kg.train:
            s, r, o = fact
            if o == entity:
                in_degree += 1
        
        avg_degree = (out_degree + in_degree) / 2.0 if (out_degree + in_degree) > 0 else 0.0
        return avg_degree
    
    def add_facts_using_pseudo_entities(self, model, replace_ratio=1.0):
        if not hasattr(self.kg, 'pseudo_ent2align') or len(self.kg.pseudo_ent2align) == 0:
            return 0
        
        pseudo_ent2align = self.kg.pseudo_ent2align
        
        pseudo_pairs = []
        used_entities = set()
        for e1, e2 in pseudo_ent2align.items():
            if e1 not in used_entities and e2 not in used_entities:
                pseudo_pairs.append((e1, e2))
                used_entities.add(e1)
                used_entities.add(e2)
        
        selected_pairs = pseudo_pairs
        
        selected_entities = set()
        for e1, e2 in selected_pairs:
            selected_entities.add(e1)
            selected_entities.add(e2)
        
        candidate_triples = []  
        for fact in self.kg.train:
            s, r, o = fact
            if s in selected_entities and s in pseudo_ent2align:
                aligned_s = pseudo_ent2align[s]
                candidate_triples.append((aligned_s, r, o, s, True))  
            if o in selected_entities and o in pseudo_ent2align:
                aligned_o = pseudo_ent2align[o]
                candidate_triples.append((s, r, aligned_o, o, False))  
        
        
        if len(candidate_triples) == 0:
            return 0
        
        selection_strategy = getattr(self.args, 'pseudo_triple_selection', 'score_high')
        use_entropy_selection = getattr(self.args, 'use_entropy_selection', True)
        
        
        if selection_strategy in ['score_high', 'score_low']:
            
            batch_size = getattr(self.args, 'score_batch_size', 256)
            scores = self._compute_triple_scores_batch(model, candidate_triples, batch_size=batch_size)
            
            triple_scores = list(zip(candidate_triples, scores))
            
            
            if selection_strategy == 'score_high':
                triple_scores.sort(key=lambda x: x[1], reverse=True)  
            else:
                triple_scores.sort(key=lambda x: x[1], reverse=False)  
            
            if use_entropy_selection:
                scores = np.array([item[1] for item in triple_scores])
                
                mu = scores.mean()
                var = scores.var(ddof=0)  
                
                n_sigma = getattr(self.args, 'entropy_n_sigma', 2.0)
                entropy_threshold = getattr(self.args, 'entropy_threshold', 10.0)
                
                selected_triples = []
                cumulative_entropy = 0.0
                
                for triple, score in triple_scores:
                    clamped_diff = min(0.0, score - mu)  
                    prob = np.exp(-((clamped_diff ** 2) / (2 * var / (n_sigma ** 2))))
                    
                    if prob <= 0.0:
                        entropy = 0.0
                    elif prob >= 1.0:
                        entropy = 0.0  
                    else:
                        entropy = -prob * np.log(prob)
                    
                    cumulative_entropy += entropy
                    
                    if cumulative_entropy > entropy_threshold:
                        break
                    
                    selected_triples.append(triple)
            else:
                selected_triples = [item[0] for item in triple_scores]
        facts, confidence, is_expands, is_ea = [], [], [], []
        for triple in selected_triples:
            s, r, o = triple[0], triple[1], triple[2]
            facts.append((s, r, o))
            confidence.append([1.0])
            is_expands.append(True)
            is_ea.append(True)
        
        
        if len(facts) > 0:
            self.facts = self.original_facts + facts
            self.confidence = torch.cat([self.original_confidence, torch.FloatTensor(confidence)], dim=0)
            self.is_expand = torch.cat([self.original_is_expand, torch.BoolTensor(is_expands).reshape(-1, 1)])
            self.is_ea = torch.cat([self.original_is_ea, torch.BoolTensor(is_ea).reshape(-1, 1)])
        
        return len(facts)


class RelationBatchSampler(BatchSampler):
    '''
    This class is created to save memory for the attn embedder.
    The training using attn embedder involves facts with the same relation placed within the same batch.
    '''
    def __init__(self, args, kg, dataset, batch_size, shuffle=True):
        self.args = args
        self.kg = kg
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        super(RelationBatchSampler, self).__init__(self.dataset, self.batch_size, self.shuffle)

    def _separate_relations(self):
        relation_samples = {}
        for idx, sample in enumerate(self.dataset):
            try:
                relation = sample['fact'][1]
            except:
                relation = sample[1]
            try:
                if relation not in relation_samples.keys():
                    relation_samples[relation] = []
            except:
                print(relation)
            relation_samples[relation].append(idx)
        return relation_samples

    def __iter__(self):
        dataset = [i for i in range(len(self.dataset))]
        batches = []
        if self.shuffle:
            random.shuffle(dataset)
        num_batches = len(dataset) // self.batch_size
        for i in range(num_batches):
            batch = list(dataset)[i * self.batch_size: (i + 1) * self.batch_size]
            batches.append(batch)
        if len(dataset) % self.batch_size > 0:
            batches.append(list(dataset)[num_batches * self.batch_size:])
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.dataset) - 1) // self.batch_size + 1


class CrossLinkPredictionValidDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode

        '''prepare data for validation and testing'''
        self.original_valid = self.build_facts()
        self.valid = deepcopy(self.original_valid)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        ele = self.valid[idx]
        fact, label, source_id = torch.LongTensor(ele['fact']), ele['label'], ele['source_id']

        label = self.get_label(list(label), source_id)
        return fact[0], fact[1], fact[2], label.float(), self.mode

    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        o = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return s, r, o, label, mode

    def get_label(self, label, source_id):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        y = np.zeros([self.kg.num_ent], dtype=np.float32)
        if source_id == 0:
            y[len(self.kg.source[0].entities):] = 1.0
        else:
            y[:len(self.kg.source[0].entities)] = 1.0
        for e2 in label: y[e2] = 1.0

        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid = []
        for source_id, source in self.kg.source.items():
            for fact in source.valid:
                s, r, o = fact
                if self.mode == 'head-batch':
                    label = set(self.kg.sr2o_valid[(o, r+1)])
                elif self.mode == 'tail-batch':
                    label = set(self.kg.sr2o_valid[(s, r)])
                valid.append({'fact': (s, r, o), 'label': label, 'source_id': source_id})
        return valid



class CrossLinkPredictionTestDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode

        '''prepare data for validation and testing'''
        self.test = self.build_facts()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        ele = self.test[idx]
        fact, label, source_id, known = torch.LongTensor(ele['fact']), ele['label'], ele['source_id'], ele['known']
        type = torch.LongTensor([ele['type']])
        label = self.get_label(list(label), known)
        return fact[0], fact[1], fact[2], label.float(), self.mode, type

    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        o = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        types = torch.stack([_[5] for _ in data], dim=0)
        return s, r, o, label, mode, types

    def get_label(self, label, known):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        if known:
            y = np.ones([self.kg.num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.kg.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        test = []
        inner_test = set(self.kg.inner_test)
        for source_id, source in self.kg.source.items():
            for fact in source.test:
                if fact in self.kg.expand_train:
                    continue
                else:
                    known = False
                s, r, o = fact
                if self.mode == 'tail-batch':
                    label = set(self.kg.sr2o_test[(s, r)])
                else:
                    label = set(self.kg.sr2o_test[(o, r+1)])
                if (s, r, o) in inner_test:
                    type = -1
                else:
                    type = 1
                test.append({'fact': (s, r, o), 'label': label, 'source_id': source_id, 'known': known, 'type': type})
        return test


