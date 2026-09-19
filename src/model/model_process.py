from ..data_load.data_loader import *
from torch.utils.data import DataLoader
import math
import torch
import torch.nn.functional as F
from src.model.loss_func import EntityAlignmentLoss, TripleInfoNCELoss, MarginLoss


class TrainEvalBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg

    def _create_dataset(self):
        pass


'''Cross-Link Prediction'''
class CrossLinkPredictionTrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        '''prepare data'''
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.head_dataset = CrossLinkPredictionTrainDatasetMarginLoss(self.args, self.kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionTrainDatasetMarginLoss(self.args, self.kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      batch_sampler=self.head_sampler,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(self.args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)

        self.alignment_loss_fn = EntityAlignmentLoss(self.args, self.kg)
        if getattr(self.args, 'use_triple_infonce', False):
            infonce_temperature = getattr(self.args, 'triple_infonce_temperature', 0.07)
            self.triple_infonce_loss_fn = TripleInfoNCELoss(self.args, self.kg, temperature=infonce_temperature)
        else:
            self.triple_infonce_loss_fn = None

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        
        current_epoch = getattr(self.args, 'epoch', 0)
        
        if self.args.use_pseudo_alignment:
            self.compute_pseudo_alignment_pairs(model)
            num_pseudo_pairs = len(self.kg.pseudo_ent2align) // 2 if hasattr(self.kg, 'pseudo_ent2align') else 0
            if hasattr(self.kg, 'pseudo_ent2align') and len(self.kg.pseudo_ent2align) > 0:
                replace_ratio = 1.0  
                num_added = self.add_facts_using_pseudo_entities(model, replace_ratio)
        use_inter_triples = getattr(self.args, 'use_inter_triples', False)
        
        graph_stats = None
        should_compute_stats = not use_inter_triples or (current_epoch == 0)
        
        if should_compute_stats:
            graph_stats = self.kg.build_inter_graph_structure(
                train_datasets=(self.head_dataset, self.tail_dataset)
            )

        if graph_stats is not None:
            self._last_graph_stats = graph_stats

        head_data_loader = iter(self.head_data_loader)
        tail_data_loader = iter(self.tail_data_loader)
        success = False
        total_loss = 0.0
        num_known_pairs_total = 0
        num_pseudo_pairs_total = 0
        num_batches_with_known_align = 0
        num_batches_with_pseudo_align = 0
        num_original_triples = 0
        num_known_expand_triples = 0
        num_pseudo_expand_triples = 0
        while not success:
            total_loss = 0.0
            count_num = 0
            count_loss = None  
            count_batch = 0

            num_known_pairs_total = 0
            num_pseudo_pairs_total = 0
            num_batches_with_known_align = 0
            num_batches_with_pseudo_align = 0

            num_original_triples = 0
            num_known_expand_triples = 0
            num_pseudo_expand_triples = 0
            try:
                for idx_b, batch in enumerate(tqdm(range(2 * len(self.head_data_loader)))):
                    '''get loss'''
                    if idx_b % 2 == 0:
                        try:
                            iter_ = next(head_data_loader)
                        except:
                            continue
                        if iter_ is None:
                            continue
                        bh, br, bt, by, bc, mode, subsampling_weight, is_expand, is_ea = iter_
                        mode = 'head-batch'
                    else:
                        try:
                            iter_ = next(tail_data_loader)
                        except:
                            continue
                        if iter_ is None:
                            continue
                        bh, br, bt, by, bc, mode, subsampling_weight, is_expand, is_ea = iter_
                        mode = 'tail-batch'
                    current_samples_num = subsampling_weight.size(0)
                    bh = bh.to(self.args.device)
                    br = br.to(self.args.device)
                    bt = bt.to(self.args.device)
                    by = by.to(self.args.device)
                    bc = bc.to(self.args.device)
                    subsampling_weight = subsampling_weight.to(self.args.device)
                    if count_loss is None:
                        optimizer.zero_grad()
                        count_loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)

                    jobs = {
                        'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": bh}, 'mode': mode},
                        'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": br}, 'mode': mode},
                        'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": bt}, 'mode': mode},
                    }
                    pred = model.forward(jobs=jobs, stage='train', mode=mode, margin=self.args.margin)
                    

                    if is_expand is not None and is_ea is not None:
                        if not isinstance(is_expand, torch.Tensor):
                            is_expand = torch.tensor(is_expand, device=bh.device, dtype=torch.bool)
                        else:
                            is_expand = is_expand.to(bh.device).bool()
                        
                        if not isinstance(is_ea, torch.Tensor):
                            is_ea = torch.tensor(is_ea, device=bh.device, dtype=torch.bool)
                        else:
                            is_ea = is_ea.to(bh.device).bool()
                        
                        by_flat = by.reshape(-1) if by.dim() > 1 else by
                        by_len = len(by_flat)
                        
                        if len(is_expand) != by_len:
                            min_len = min(len(is_expand), by_len)
                            is_expand = is_expand[:min_len]
                            is_ea = is_ea[:min_len]
                            by_flat = by_flat[:min_len]
                        
                        original_mask = ~is_expand
                        known_expand_mask = is_expand & (~is_ea)
                        pseudo_mask = is_expand & is_ea

                        positive_mask = (by_flat > 0).bool()
                        
                        if len(is_expand) == len(positive_mask):
                            num_original_triples += (original_mask & positive_mask).sum().item()
                            num_known_expand_triples += (known_expand_mask & positive_mask).sum().item()
                            num_pseudo_expand_triples += (pseudo_mask & positive_mask).sum().item()
                            
                            if original_mask.any():
                                original_bh = bh[original_mask]
                                original_br = br[original_mask]
                                original_bt = bt[original_mask]
                                original_by = by[original_mask]
                                original_bc = bc[original_mask]
                                original_subsampling_weight = subsampling_weight[original_mask]
                                original_pred = pred[original_mask]
                                
                                original_loss = model.loss(original_pred, original_by, original_subsampling_weight, original_bc, neg_ratio=self.args.neg_ratio)
                                if isinstance(original_loss, torch.Tensor) and original_loss.dim() > 0:
                                    original_loss = original_loss.mean()
                            else:
                                original_loss = torch.tensor(0.0, device=bh.device, requires_grad=True)
                            
                            batch_loss = original_loss
                            
                            if getattr(self.args, 'use_known_expand_triple_loss', True):
                                known_expand_triple_weight = getattr(self.args, 'known_expand_triple_weight', 0.1)
                                
                                if known_expand_mask.any():
                                    known_expand_bh = bh[known_expand_mask]
                                    known_expand_br = br[known_expand_mask]
                                    known_expand_bt = bt[known_expand_mask]
                                    known_expand_by = by[known_expand_mask]
                                    known_expand_bc = bc[known_expand_mask]
                                    known_expand_subsampling_weight = subsampling_weight[known_expand_mask]
                                    known_expand_pred = pred[known_expand_mask]
                                    
                                    known_expand_loss = model.loss(known_expand_pred, known_expand_by, known_expand_subsampling_weight, known_expand_bc, neg_ratio=self.args.neg_ratio)
                                    if isinstance(known_expand_loss, torch.Tensor) and known_expand_loss.dim() > 0:
                                        known_expand_loss = known_expand_loss.mean()
                                    
                                    batch_loss = batch_loss + known_expand_triple_weight * known_expand_loss
                            
                            if self.args.use_pseudo_triple_loss:
                                pseudo_triple_weight = getattr(self.args, 'pseudo_triple_weight', 0.1)
                                
                                if pseudo_mask.any():
                                    pseudo_bh = bh[pseudo_mask]
                                    pseudo_br = br[pseudo_mask]
                                    pseudo_bt = bt[pseudo_mask]
                                    pseudo_by = by[pseudo_mask]
                                    pseudo_bc = bc[pseudo_mask]
                                    pseudo_subsampling_weight = subsampling_weight[pseudo_mask]
                                    pseudo_pred = pred[pseudo_mask]
                                    
                                    pseudo_loss = model.loss(pseudo_pred, pseudo_by, pseudo_subsampling_weight, pseudo_bc, neg_ratio=self.args.neg_ratio)
                                    if isinstance(pseudo_loss, torch.Tensor) and pseudo_loss.dim() > 0:
                                        pseudo_loss = pseudo_loss.mean()
                                    
                                    batch_loss = batch_loss + pseudo_triple_weight * pseudo_loss
                            
                            if getattr(self.args, 'use_triple_infonce', False) and self.triple_infonce_loss_fn is not None:
                                known_expand_positive_mask = known_expand_mask & positive_mask
                                if known_expand_positive_mask.any():
                                    known_infonce_loss = self.compute_triple_infonce_loss(
                                        model, bh, br, bt, known_expand_positive_mask, mode
                                    )
                                    if isinstance(known_infonce_loss, torch.Tensor) and known_infonce_loss.dim() > 0:
                                        known_infonce_loss = known_infonce_loss.mean()
                                    triple_infonce_weight = getattr(self.args, 'triple_infonce_weight', 0.1)
                                    batch_loss = batch_loss + triple_infonce_weight * known_infonce_loss
                                
                                use_pseudo_triple_infonce = getattr(self.args, 'use_pseudo_triple_infonce', False)
                                if use_pseudo_triple_infonce:
                                    pseudo_positive_mask = pseudo_mask & positive_mask
                                    if pseudo_positive_mask.any():
                                        pseudo_infonce_loss = self.compute_triple_infonce_loss(
                                            model, bh, br, bt, pseudo_positive_mask, mode
                                        )
                                        if isinstance(pseudo_infonce_loss, torch.Tensor) and pseudo_infonce_loss.dim() > 0:
                                            pseudo_infonce_loss = pseudo_infonce_loss.mean()
                                        pseudo_triple_infonce_weight = getattr(self.args, 'pseudo_triple_infonce_weight', 0.1)
                                        batch_loss = batch_loss + pseudo_triple_infonce_weight * pseudo_infonce_loss
                        else:
                            batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.args.neg_ratio)
                            if isinstance(batch_loss, torch.Tensor) and batch_loss.dim() > 0:
                                batch_loss = batch_loss.mean()
                    else:
                        batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.args.neg_ratio)
                        if isinstance(batch_loss, torch.Tensor) and batch_loss.dim() > 0:
                            batch_loss = batch_loss.mean()
                    
                    if self.args.use_known_alignment and hasattr(self.kg, 'entity_pairs') and len(self.kg.entity_pairs) > 0:
                        known_align_loss, num_known_pairs = self.compute_known_alignment_loss(model)
                        batch_loss = batch_loss + self.args.known_align_weight * known_align_loss
                        num_known_pairs_total += num_known_pairs
                        num_batches_with_known_align += 1
                    
                    if self.args.use_pseudo_alignment:
                        pseudo_align_loss, num_pseudo_pairs = self.compute_pseudo_alignment_loss(model)
                        batch_loss = batch_loss + self.args.pseudo_align_weight * pseudo_align_loss
                        num_pseudo_pairs_total += num_pseudo_pairs
                        num_batches_with_pseudo_align += 1

                    '''update'''
                    if count_loss is None:
                        count_loss = batch_loss * current_samples_num
                    else:
                        count_loss = count_loss + batch_loss * current_samples_num
                    count_num += current_samples_num
                    count_batch += 1
                    if count_num >= self.args.batch_size:
                        count_loss = count_loss / count_num
                        count_loss.backward()
                        optimizer.step()
                        if hasattr(model.encoder, 'invalidate_cache'):
                            model.encoder.invalidate_cache()
                        total_loss += count_loss.item()
                        count_loss = None  
                        count_num = 0
                        count_batch = 0
                    '''post processing'''
                success = True
            except:
                import sys, traceback
                e = sys.exc_info()[1]
                tb = traceback.format_exc()
                if 'CUDA out of memory' in str(e):
                    self.batch_size = self.batch_size // 2
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                        batch_sampler=self.head_sampler,
                                                        collate_fn=self.head_dataset.collate_fn,
                                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                        pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                        batch_sampler=self.tail_sampler,
                                                        collate_fn=self.tail_dataset.collate_fn,
                                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                        pin_memory=True)
                    print('Batch size is too large, reduce to', self.batch_size)
                else:
                    print('Error:', repr(e))
                    print(tb)
                    break

        avg_known_pairs = num_known_pairs_total // max(num_batches_with_known_align, 1) if num_batches_with_known_align > 0 else 0
        avg_pseudo_pairs = num_pseudo_pairs_total // max(num_batches_with_pseudo_align, 1) if num_batches_with_pseudo_align > 0 else 0
        
        triple_stats = {
            'num_original': num_original_triples,
            'num_known_expand': num_known_expand_triples,
            'num_pseudo_expand': num_pseudo_expand_triples
        }
        
        use_inter_triples = getattr(self.args, 'use_inter_triples', False)
        if hasattr(self, '_last_graph_stats'):
            triple_stats['graph_original'] = self._last_graph_stats.get('num_original_triples', 0)
            triple_stats['graph_known_expand'] = self._last_graph_stats.get('num_known_expand_triples', 0)
            triple_stats['graph_pseudo_expand'] = self._last_graph_stats.get('num_pseudo_expand_triples', 0)
            
        else:
            triple_stats['graph_original'] = 0
            triple_stats['graph_known_expand'] = 0
            triple_stats['graph_pseudo_expand'] = 0
        
        return total_loss, avg_known_pairs, avg_pseudo_pairs, triple_stats

    
    def compute_known_alignment_loss(self, model):
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)
        num_known_pairs = len(self.kg.entity_pairs) if hasattr(self.kg, 'entity_pairs') and len(self.kg.entity_pairs) > 0 else 0
        align_loss = self.alignment_loss_fn(ent_embeddings, self.kg.entity_pairs)
        return align_loss, num_known_pairs
    
    def compute_pseudo_alignment_pairs(self, model):
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)
        ent_embeddings = F.normalize(ent_embeddings, p=2, dim=1)
        num_ent_kg1 = self.kg.source[0].num_ent
        num_ent_kg2 = ent_embeddings.size(0) - num_ent_kg1
        ent_emb_kg1 = ent_embeddings[:num_ent_kg1]  
        ent_emb_kg2 = ent_embeddings[num_ent_kg1:]  
        similarity_matrix = torch.matmul(ent_emb_kg1, ent_emb_kg2.t()) 
        
        known_kg1_entities = set()  
        known_kg2_entities = set()  
        
        if hasattr(self.kg, 'entity_pairs') and len(self.kg.entity_pairs) > 0:
            for e1, e2 in self.kg.entity_pairs:
                if e1 < num_ent_kg1 and e2 >= num_ent_kg1:
                    known_kg1_entities.add(e1)
                    known_kg2_entities.add(e2 - num_ent_kg1)
                elif e2 < num_ent_kg1 and e1 >= num_ent_kg1:
                    known_kg1_entities.add(e2)
                    known_kg2_entities.add(e1 - num_ent_kg1)
        
        E1 = set(range(num_ent_kg1)) - known_kg1_entities
        E2 = set(range(num_ent_kg2)) - known_kg2_entities
        
        use_threshold = getattr(self.args, 'use_pseudo_align_threshold', True)
        if use_threshold:
            threshold = getattr(self.args, 'pseudo_align_threshold', 0.1)
        else:
            threshold = -1e4
        
        pseudo_pairs = []
        
        sim_matrix_work = similarity_matrix.clone()
        
        if known_kg1_entities:
            idx1 = torch.tensor(list(known_kg1_entities), device=sim_matrix_work.device, dtype=torch.long)
            sim_matrix_work[idx1, :] = float('-inf')
        if known_kg2_entities:
            idx2 = torch.tensor(list(known_kg2_entities), device=sim_matrix_work.device, dtype=torch.long)
            sim_matrix_work[:, idx2] = float('-inf')
        
        num_mutual_best_match = 0
        num_above_threshold = 0
        confidence_values = []


        current_round = 0
        max_rounds = getattr(self.args, 'pseudo_align_max_rounds', 2) 

        while len(E1) > 0 and len(E2) > 0 and current_round < max_rounds:

            E1_list = sorted(list(E1))
            E2_list = sorted(list(E2))
            
            if len(E1_list) == 0 or len(E2_list) == 0:
                break
            
            E1_tensor = torch.tensor(E1_list, device=sim_matrix_work.device, dtype=torch.long)
            E2_tensor = torch.tensor(E2_list, device=sim_matrix_work.device, dtype=torch.long)
            
            sub_sim = sim_matrix_work[E1_tensor][:, E2_tensor]
            
            if sub_sim.numel() == 0:
                break
            
            max_vals_row, argmax_row = sub_sim.max(dim=1)  # [len(E1)]
            
            max_vals_col, argmax_col = sub_sim.max(dim=0)  # [len(E2)]
            
            mutual_mask = (argmax_col[argmax_row] == torch.arange(len(E1_list), device=sub_sim.device))
            
            mutual_indices = torch.where(mutual_mask)[0]  
            
            if len(mutual_indices) == 0:
                break
            
            num_mutual_best_match += len(mutual_indices)
            
            best_pairs_this_round = []
            
            for idx in mutual_indices:
                i_local = idx.item()  
                j_local = argmax_row[i_local].item()  
                
                i_global = E1_list[i_local]  
                j_global = E2_list[j_local]  
                
                S_ij = sub_sim[i_local, j_local].item()
                
                row_i = sub_sim[i_local, :]  # [len(E2)]
                if len(E2_list) > 1:
                    row_i_copy = row_i.clone()
                    row_i_copy[j_local] = float('-inf')
                    secmax_row = row_i_copy.max().item()
                else:
                    secmax_row = float('-inf')
                
                c_ei = S_ij - secmax_row if secmax_row != float('-inf') else S_ij
                
                col_j = sub_sim[:, j_local]  # [len(E1)]
                if len(E1_list) > 1:
                    col_j_copy = col_j.clone()
                    col_j_copy[i_local] = float('-inf')
                    secmax_col = col_j_copy.max().item()
                else:
                    secmax_col = float('-inf')
                
                c_ej = S_ij - secmax_col if secmax_col != float('-inf') else S_ij
                
                c_pair = (c_ei + c_ej) / 2.0
                confidence_values.append(c_pair)
                
                best_pairs_this_round.append((i_global, j_global, c_pair, S_ij))
            
            best_pairs_this_round.sort(key=lambda x: x[2], reverse=True)
            
            added_this_round = 0
            for i_global, j_global, c_pair, S_ij in best_pairs_this_round:
                if i_global not in E1 or j_global not in E2:
                    continue
                
                if c_pair > threshold:
                    num_above_threshold += 1
                    pseudo_pairs.append((i_global, j_global + num_ent_kg1, c_pair))
                    
                    E1.remove(i_global)
                    E2.remove(j_global)
                    
                    sim_matrix_work[i_global, :] = float('-inf')
                    sim_matrix_work[:, j_global] = float('-inf')
                    
                    added_this_round += 1
            
            if added_this_round == 0:
                break
            
            current_round += 1
        
        num_pseudo_pairs = len(pseudo_pairs)
        
        if confidence_values:
            conf_tensor = torch.tensor(confidence_values)
        
        if num_pseudo_pairs == 0:
            self.kg.pseudo_ent2align = dict()
            return
        
        self.kg.pseudo_ent2align = dict()
        for e1, e2, _ in pseudo_pairs:
            self.kg.pseudo_ent2align[e1] = e2
            self.kg.pseudo_ent2align[e2] = e1
        
    def compute_pseudo_alignment_loss(self, model):
        if not hasattr(self.kg, 'pseudo_ent2align') or len(self.kg.pseudo_ent2align) == 0:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True), 0
        
        pseudo_pairs = []
        used_entities = set()
        for e1, e2 in self.kg.pseudo_ent2align.items():
            if e1 not in used_entities and e2 not in used_entities:
                pseudo_pairs.append((e1, e2))
                used_entities.add(e1)
                used_entities.add(e2)
        
        num_pseudo_pairs = len(pseudo_pairs)
        if num_pseudo_pairs == 0:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True), 0
        
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)
        
        pseudo_align_loss = self.alignment_loss_fn(ent_embeddings, pseudo_pairs)
        
        return pseudo_align_loss, num_pseudo_pairs
    
    def compute_triple_infonce_loss(self, model, bh, br, bt, expand_mask, mode):
        if not expand_mask.any() or self.triple_infonce_loss_fn is None:
            return torch.tensor(0.0, device=bh.device, requires_grad=True)
        
        expand_bh = bh[expand_mask]
        expand_br = br[expand_mask]
        expand_bt = bt[expand_mask]
        
        h_emb = model.encoder.embed_ent(expand_bh, scorer=model.decoder.scorer, mode=mode)
        r_emb = model.encoder.embed_rel(expand_br)
        t_emb = model.encoder.embed_ent(expand_bt, scorer=model.decoder.scorer, mode=mode)
        
        infonce_loss = self.triple_infonce_loss_fn(h_emb, r_emb, t_emb)
        
        return infonce_loss
    
    def add_facts_using_pseudo_entities(self, model, replace_ratio=1.0):
        num_added_head = self.head_dataset.add_facts_using_pseudo_entities(model, replace_ratio)
        num_added_tail = self.tail_dataset.add_facts_using_pseudo_entities(model, replace_ratio)
        
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                           batch_sampler=self.head_sampler,
                                           collate_fn=self.head_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        
        return num_added_head + num_added_tail


class CrossLinkPredictionValidBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = self.args.test_batch_size
        '''prepare data'''
        self.head_dataset = CrossLinkPredictionValidDataset(args, kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionValidDataset(args, kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.valid, self.args.test_batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.valid, self.args.test_batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      # shuffle=False,
                                      batch_sampler=self.head_sampler,
                                      # batch_size=self.batch_size,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           # shuffle=False,
                                           batch_sampler=self.tail_sampler,
                                           # batch_size=self.batch_size,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(args.seed)),
                                           pin_memory=True)

    def process_epoch(self, model):
        model.eval()

        '''start evaluation'''
        success = False
        while not success:
            try:
                num = 0
                results = dict()
                for data_loader in [self.head_data_loader, self.tail_data_loader]:
                    for batch in data_loader:
                        sub, rel, obj, label, mode = batch
                        sub = sub.to(self.args.device)
                        rel = rel.to(self.args.device)
                        obj = obj.to(self.args.device)
                        label = label.to(self.args.device)
                        num += len(sub)

                        '''link prediction'''
                        if mode == 'tail-batch':
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = obj
                        else:
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = sub
                        pred = model(jobs=jobs, mode=mode, margin=self.args.margin)



                        b_range = torch.arange(pred.size()[0], device=self.args.device)
                        target_pred = pred[b_range, target]
                        pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)

                        pred[b_range, target] = target_pred

                        '''rank all candidate entities'''
                        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, target]

                        '''get results'''
                        ranks = ranks.float()
                        results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                        results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                        results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)

                        for k in range(10):
                            results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get('hits{}'.format(k + 1), 0.0)
                success = True
            except:
                import sys
                e = sys.exc_info()[0]
                if 'CUDA out of memory' in str(e):
                    print('CUDA out of memory, try to reduce batch size by half')
                    self.batch_size = self.batch_size // 2
                    self.head_dataset = CrossLinkPredictionValidDataset(self.args, self.kg, mode='head-batch')
                    self.tail_dataset = CrossLinkPredictionValidDataset(self.args, self.kg, mode='tail-batch')
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.valid,
                                                             self.args.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.valid,
                                                             self.args.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.head_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.head_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.tail_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.tail_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                else:
                    print('Unexpected error:', e)
                    break


        count = float(results['count'])
        for key, val in results.items():
            if key != 'count':
                results[key] = round(val / count, 4)
        return results



class CrossLinkPredictionTestBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = self.args.test_batch_size
        '''prepare data'''
        self.head_dataset = CrossLinkPredictionTestDataset(args, kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionTestDataset(args, kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.test, self.args.test_batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.test, self.args.test_batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      # shuffle=False,
                                      batch_sampler=self.head_sampler,
                                      # batch_size=self.batch_size,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           # shuffle=False,
                                           batch_sampler=self.tail_sampler,
                                           # batch_size=self.batch_size,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(args.seed)),
                                           pin_memory=True)

    def process_epoch(self, model):
        def update_results(results_, ranks_):
            results_['count'] = torch.numel(ranks_) + results_.get('count', 0.0)
            results_['mr'] = torch.sum(ranks_).item() + results_.get('mr', 0.0)
            results_['mrr'] = torch.sum(1.0 / ranks_).item() + results_.get('mrr', 0.0)
            for k in range(10):
                results_['hits{}'.format(k + 1)] = torch.numel(ranks_[ranks_ <= (k + 1)]) + results_.get(
                    'hits{}'.format(k + 1), 0.0)
            return results_


        def ave_results(results_):
            count = float(results_.get('count', 0.0))
            if count == 0.0:
                for k in ['mr', 'mrr'] + ['hits{}'.format(i + 1) for i in range(10)]:
                    results_.setdefault(k, 0.0)
                return results_
            for key, val in results_.items():
                if key != 'count':
                    results_[key] = round(val / count, 4)
            return results_

        model.eval()
        success = False
        while not success:
            try:
                num = 0
                results_inner, results_cross, results_cross_relation, results_cross_entity = dict(), dict(), dict(), dict()
                '''start evaluation'''
                for data_loader in [self.head_data_loader, self.tail_data_loader]:
                    for batch in tqdm(data_loader):
                        sub, rel, obj, label, mode, types = batch
                        sub = sub.to(self.args.device)
                        rel = rel.to(self.args.device)
                        obj = obj.to(self.args.device)
                        label = label.to(self.args.device)
                        types = types.to(self.args.device)

                        num += len(sub)
                        '''link prediction'''
                        if mode == 'tail-batch':
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = obj
                        else:
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = sub
                        pred = model(jobs=jobs, mode=mode, margin=self.args.margin)

                        # pred = model(jobs=jobs, mode=mode, margin=self.args.margin)
                        b_range = torch.arange(pred.size()[0], device=self.args.device)

                        # # filter
                        target_pred = pred[b_range, target]
                        pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)

                        pred[b_range, target] = target_pred

                        '''rank all candidate entities'''
                        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, target]
                        '''get results'''
                        ranks = ranks.float()

                        inner_idx = (types == -1).reshape(-1)
                        cross_idx = (types == 1).reshape(-1)
                        cross_relation_idx = ((sub<self.kg.source[0].num_ent) & (obj<self.kg.source[0].num_ent) & (rel>=self.kg.source[0].num_rel)) \
                                             + ((sub>=self.kg.source[0].num_ent)&(obj>=self.kg.source[0].num_ent)&(rel<self.kg.source[0].num_rel))
                        cross_entity_idx = (~cross_relation_idx) & cross_idx


                        ranks_cross_entity = ranks[cross_entity_idx]

                        results_cross_entity = update_results(results_cross_entity, ranks_cross_entity)

                success = True
            except Exception as e:
                if 'CUDA out of memory' in str(e):
                    self.args.logger.info('Error: {}'.format(e))
                    self.args.logger.info('Retry...')
                    self.batch_size = int(self.batch_size / 2)
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.test,
                                                             self.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.test,
                                                             self.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.head_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.head_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.tail_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.tail_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                else:
                    self.args.logger.info('Error: {}'.format(e))
                    break

        results_cross_entity = ave_results(results_cross_entity)

        self.args.logger.info('cross_entity_results:{}'.format(results_cross_entity))

        rr = results_cross_entity
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        return results_cross











