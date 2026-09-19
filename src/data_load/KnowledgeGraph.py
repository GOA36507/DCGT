from src.utils import *
from ..utils import *
from copy import deepcopy as dcopy
from datasets import load_dataset


class KnowledgeGraph():
    def __init__(self, args):
        self.args = args

        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()
        self.entity_pairs = list()

        self.train, self.valid, self.test = list(), list(), list()
        self.edge_s, self.edge_r, self.edge_o = list(), list(), list()
        self.inter_edge_s, self.inter_edge_r, self.inter_edge_o = None, None, None
        self._inter_graph_built = False
        self._cached_graph_stats = None
        self.source = {i: Source(self.args) for i in range(len(self.args.source_list))}
        self.data_path = self.args.root_dir + self.args.source_list[0] + '-' + self.args.source_list[1] + '/'

        self.expand_train = list()
        self.pseudo_ent2align = dict()
        self.load_data()
        self.prepare_data()

    @timing_decorator
    def load_data(self):
        '''
        load data from all source file
        '''

        sr2o_all = dict()
        sr2o_valid = dict()
        self.r2s, self.r2o = dict(), dict()

        data_path = self.args.root_dir + self.args.source_list[0] + '-' + self.args.source_list[1]+'/'

        for ss_id, source in enumerate(self.args.source_list):
            edge_s, edge_r, edge_o = [], [], []
            '''load facts'''
            order = 'hrt'
            ds = load_dataset(self.data_path + source)
            train_facts = load_fact(ds['train'], order)
            valid_facts = load_fact(ds['validation'], order)

            test_facts = load_fact(ds['test'], order)
            
            '''extract entities & relations from all facts'''
            all_facts = train_facts + valid_facts + test_facts
            entities, relations = self.expand_entity_relation(all_facts)


            entities_id = self.ent_rel2id(entities, 'ent')
            relations_id = self.ent_rel2id(relations, 'rel')

            '''read train/valid data'''
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts)

            self.train += train
            self.valid += valid

            edge_s, edge_o, edge_r = self.expand_kg(train, 'train', edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, self.r2s, self.r2o)
            self.edge_s = self.edge_s + edge_s
            self.edge_r = self.edge_r + edge_r
            self.edge_o = self.edge_o + edge_o
            _, _, _ = self.expand_kg(valid, 'valid', [], [], [], sr2o_all, sr2o_valid, None, None)

            '''store source'''
            self.store_source(ss_id, train, valid, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, entities_id, relations_id)
        
        self.sr2o_valid = sr2o_valid

        '''read shared entities'''
        align_file = 'known_shared_entities.txt'

        align_pairs = load_align_pair(data_path, align_file)
        print('Alignment: ', len(align_pairs))
        self.ent2align = dict()
        for pair in align_pairs:
            e1, e2 = pair
            e1, e2 = self.entity2id[e1], self.entity2id[e2]
            self.ent2align[e1] = e2
            self.ent2align[e2] = e1
        self.expand_align(align_pairs)

        for source_id, source in self.source.items():
            for fact in source.train:
                s, r, o = fact
                if s in self.ent2align.keys():
                    self.expand_train.append((self.ent2align[s], r, o))
                    self.expand_train.append((o, r+1, self.ent2align[s]))
                if o in self.ent2align.keys():
                    self.expand_train.append((s, r, self.ent2align[o]))
                    self.expand_train.append((self.ent2align[o], r+1, s))
        self.expand_train = set(self.expand_train)

        self.edge_s = torch.LongTensor(self.edge_s)
        self.edge_r = torch.LongTensor(self.edge_r)
        self.edge_o = torch.LongTensor(self.edge_o)

        self.sr2o_all = sr2o_all

        self.r2s, self.r2o = self.get_r2e(self.train)


    def prepare_data(self):
        '''from RotatE'''
        self.count = self.count_frequency(self.train)
        self.sr2o_train = self.get_true_head_and_tail(self.train)

    def count_frequency(self, triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for triple in triples:
            h, r, t = triple[0], triple[1], triple[2]
            if (h, r) not in count:
                count[(h, r)] = start
            else:
                count[(h, r)] += 1
            if (t, r+1) not in count:
                count[(t, r+1)] = start
            else:
                count[(t, r+1)] += 1
        return count

    def get_true_head_and_tail(self, triples):
        '''
        Build a dictionary of true triples that will
        be used to filter negative sampling
        '''
        sr2o = dict()
        for h, r, t in triples:
            if (h, r) not in sr2o:
                sr2o[(h, r)] = []
            if (t, r+1) not in sr2o:
                sr2o[(t, r+1)] = []
            sr2o[(t, r+1)].append(h)
            sr2o[(h, r)].append(t)
        for (h, r) in sr2o.keys():
            sr2o[(h, r)] = np.array(list(set(sr2o[(h, r)])))
        return sr2o
    
    def load_test(self):
        if len(self.test) > 0:
            return

        cross_ds = load_dataset(self.data_path + 'cross')
        cross_test_facts = load_fact(cross_ds['test'], 'hrt')
        self.cross_test = self.fact2id(cross_test_facts)

        self.inner_test = []
        self.inner_test_align_0, self.inner_test_align_1, self.inner_test_align_2 = [], [], []
        for ss_id, source in enumerate(self.args.source_list):
            ds = load_dataset(self.data_path + source)
            inner_test_facts = load_fact(ds['test'], 'hrt')
            self.inner_test += self.fact2id(inner_test_facts)
        self.test = self.cross_test + self.inner_test
        _, _, _ = self.expand_kg(self.test, 'test', [], [], [], self.sr2o_all, None, None, None)
        self.split_test_to_source(self.test)

        data_path = self.args.root_dir + self.args.source_list[0] + '-' + self.args.source_list[1] + '/'
        all_align_pairs = load_align_pair(data_path, 'all_shared_entities.txt')
        ent2align_all = dict()
        for pair in all_align_pairs:
            e1, e2 = pair
            ent2align_all[self.entity2id[e1]] = self.entity2id[e2]
            ent2align_all[self.entity2id[e2]] = self.entity2id[e1]
        self.sr2o_test = dict()
        for facts in [self.train, self.valid, self.test]:
            for fact in facts:
                s, r, o = fact
                item = self.sr2o_test.get((s, r), set())
                item.add(o)
                self.sr2o_test[(s, r)] = item
                item = self.sr2o_test.get((o, r + 1), set())
                item.add(s)
                self.sr2o_test[(o, r + 1)] = item
                if s in ent2align_all.keys():
                    item = self.sr2o_test.get((ent2align_all[s], r), set())
                    item.add(o)
                    self.sr2o_test[(ent2align_all[s], r)] = item
                    item = self.sr2o_test.get((o, r + 1), set())
                    item.add(ent2align_all[s])
                    self.sr2o_test[(o, r + 1)] = item
                if o in ent2align_all.keys():
                    item = self.sr2o_test.get((s, r), set())
                    item.add(ent2align_all[o])
                    self.sr2o_test[(s, r)] = item
                    item = self.sr2o_test.get((ent2align_all[o], r + 1), set())
                    item.add(s)
                    self.sr2o_test[(ent2align_all[o], r + 1)] = item

    def ent_rel2id(self, lst, mode='ent'):
        res = []
        for item in lst:
            if mode == 'ent':
                res.append(self.entity2id[item])
            else:
                res.append(self.relation2id[item])
        return res

    @timing_decorator
    def expand_entity_relation(self, facts):
        '''extract entities and relations from new facts'''
        new_entities, new_relations = set(), set()
        for (s, r, o) in facts:
            '''extract entities'''
            new_entities.add(s)
            new_relations.add(r)
            new_entities.add(o)
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_ent
                self.id2entity[self.num_ent] = s
                self.num_ent += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_ent
                self.id2entity[self.num_ent] = o
                self.num_ent += 1

            '''extract relations'''
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.id2relation[self.num_rel] = r
                self.id2relation[self.num_rel + 1] = r + '_inv'
                self.relation2inv[self.num_rel] = self.num_rel + 1
                self.relation2inv[self.num_rel + 1] = self.num_rel
                self.num_rel += 2
        print('Entities {} Relations {}'.format(self.num_ent, self.num_rel))
        return new_entities, new_relations

    @timing_decorator
    def split_test_to_source(self, test):
        for idx, kg in self.source.items():
            max_id = max(kg.entities)
            min_id = min(kg.entities)
            for fact in test:
                h, r, t = fact
                if h <= max_id and h >= min_id:
                    kg.test.add(fact)

    @timing_decorator
    def fact2id(self, facts):
        '''(s name, r name, o name)-->(s id, r id, o id)'''
        fact_id = []
        for (s, r, o) in facts:
            fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    @timing_decorator
    def expand_kg(self, facts, split, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, r2s, r2o):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        def add_key2val(dict, key, val):
            '''add {key: value} to dict'''
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)
        edge_s.clear()
        edge_o.clear()
        edge_r.clear()
        for (h, r, t) in facts:
            if split == 'train':
                '''edge_index'''
                edge_s.append(h)
                edge_r.append(r)
                edge_o.append(t)
                edge_s.append(t)
                edge_r.append(r+1)
                edge_o.append(h)
                add_key2val(r2s, r, h)
                add_key2val(r2o, r, t)
                add_key2val(r2s, r+1, t)
                add_key2val(r2o, r+1, h)
            '''sr2o'''
            add_key2val(sr2o_all, (h, r), t)
            add_key2val(sr2o_all, (t, self.relation2inv[r]), h)
            if split in ['train', 'valid']:
                add_key2val(sr2o_valid, (h, r), t)
                add_key2val(sr2o_valid, (t, self.relation2inv[r]), h)
        return edge_s, edge_o, edge_r
    
    @timing_decorator
    def expand_align(self, align_pairs):
        for pair in align_pairs:
            if pair[0] not in self.entity2id.keys() or pair[1] not in self.entity2id.keys():
                continue
            self.entity_pairs.append((self.entity2id[pair[0]], self.entity2id[pair[1]]))

    @timing_decorator
    def store_source(self, ss_id, train_new, valid, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, entities, relations):
        '''store source data'''
        if ss_id > 0:
            self.source[ss_id].num_ent = self.num_ent - self.source[ss_id - 1].num_ent
            self.source[ss_id].num_rel = self.num_rel - self.source[ss_id - 1].num_rel
        else:
            self.source[ss_id].num_ent = self.num_ent
            self.source[ss_id].num_rel = self.num_rel

        '''train, valid and test data'''
        self.source[ss_id].train = dcopy(train_new)
        self.source[ss_id].valid = dcopy(valid)
        self.source[ss_id].entities = sorted(entities)
        self.source[ss_id].relations = sorted(relations)

        '''edge_index, edge_type (for GCN)'''
        self.source[ss_id].edge_s = dcopy(edge_s)
        self.source[ss_id].edge_r = dcopy(edge_r)
        self.source[ss_id].edge_o = dcopy(edge_o)

    def get_r2e(self, facts):
        r2o, r2s = dict(), dict()
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
            # r2o
            item = r2o.get(r, set())
            item.add(o)
            r2o[r] = item
            # r2s
            item = r2s.get(r, set())
            item.add(s)
            r2s[r] = item
            # r_inv2o
            item = r2o.get(r+1, set())
            item.add(o)
            r2o[r+1] = item
            # r_inv2s
            item = r2s.get(r+1, set())
            item.add(o)
            r2s[r+1] = item
        return r2s, r2o

    def build_inter_graph_structure(self, train_datasets=None):
        if self._inter_graph_built and self._cached_graph_stats is not None:
            return self._cached_graph_stats
        
        inter_edge_s_list = []
        inter_edge_r_list = []
        inter_edge_o_list = []
        
        num_known_expand_triples = 0
        
        expand_triples = set()
        
        if train_datasets is not None:
            head_dataset, tail_dataset = train_datasets
            all_facts = set(head_dataset.facts) | set(tail_dataset.facts)
            
            head_facts_dict = {}
            for i, fact in enumerate(head_dataset.facts):
                is_expand = head_dataset.is_expand[i]
                is_ea = head_dataset.is_ea[i]
                if isinstance(is_expand, torch.Tensor):
                    is_expand = is_expand.squeeze().item() if is_expand.numel() > 0 else False
                if isinstance(is_ea, torch.Tensor):
                    is_ea = is_ea.squeeze().item() if is_ea.numel() > 0 else False
                head_facts_dict[fact] = (bool(is_expand), bool(is_ea))
            
            tail_facts_dict = {}
            for i, fact in enumerate(tail_dataset.facts):
                is_expand = tail_dataset.is_expand[i]
                is_ea = tail_dataset.is_ea[i]
                if isinstance(is_expand, torch.Tensor):
                    is_expand = is_expand.squeeze().item() if is_expand.numel() > 0 else False
                if isinstance(is_ea, torch.Tensor):
                    is_ea = is_ea.squeeze().item() if is_ea.numel() > 0 else False
                tail_facts_dict[fact] = (bool(is_expand), bool(is_ea))
            
            facts_info = {}
            facts_info.update(head_facts_dict)
            facts_info.update(tail_facts_dict)
            
            for fact in all_facts:
                if fact in facts_info:
                    is_expand, is_ea = facts_info[fact]
                    if is_expand and not is_ea:
                        expand_triples.add(fact)
                        num_known_expand_triples += 1
        else:
            for source_id, source in self.source.items():
                for fact in source.train:
                    s, r, o = fact
                    if s in self.ent2align.keys():
                        expand_triple = (self.ent2align[s], r, o)
                        expand_triples.add(expand_triple)
                        num_known_expand_triples += 1
                    if o in self.ent2align.keys():
                        expand_triple = (s, r, self.ent2align[o])
                        expand_triples.add(expand_triple)
                        num_known_expand_triples += 1
        
        for (h, r, t) in expand_triples:
            inter_edge_s_list.append(h)
            inter_edge_r_list.append(r)
            inter_edge_o_list.append(t)
            inter_edge_s_list.append(t)
            inter_edge_r_list.append(r + 1)
            inter_edge_o_list.append(h)
        
        if len(inter_edge_s_list) > 0:
            self.inter_edge_s = torch.LongTensor(inter_edge_s_list)
            self.inter_edge_r = torch.LongTensor(inter_edge_r_list)
            self.inter_edge_o = torch.LongTensor(inter_edge_o_list)
        else:
            self.inter_edge_s = torch.LongTensor([])
            self.inter_edge_r = torch.LongTensor([])
            self.inter_edge_o = torch.LongTensor([])
        
        num_known_expand_with_reverse = num_known_expand_triples * 2
        num_original_with_reverse = len(self.edge_s)  
        
        num_original_without_reverse = len(self.train)  
        
        
        stats = {
            'num_original_triples': num_original_with_reverse,
            'num_known_expand_triples': num_known_expand_with_reverse
        }
        
        self._cached_graph_stats = stats
        self._inter_graph_built = True
        
        return stats


class Source:
    def __init__(self, args):
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train, self.valid, self.test = list(), list(), set()
        self.cross_train = list()
        self.entities, self.relations = None, None

        self.edge_s, self.edge_r, self.edge_o = [], [], []
        self.cross_edge_s, self.cross_edge_r, self.cross_edge_o = [], [], []
        self.sr2o_all = dict()
        self.edge_index, self.edge_type = None, None



