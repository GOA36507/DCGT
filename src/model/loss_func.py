from src.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(self, args, kg, model):
        super(MarginLoss, self).__init__()
        self.args = args
        self.kg = kg
        self.model = model

    def forward(self, input, target, subsampling_weight=None, confidence=1.0, neg_ratio=10):
        p_score, n_score, p_indices, n_indices = self.split_pn_score(input, target, neg_ratio)
        
        if isinstance(confidence, torch.Tensor) and confidence.numel() > 1:
            if confidence.dim() > 1:
                confidence = confidence.reshape(-1)
            p_confidence = confidence[p_indices] if len(confidence) > max(p_indices) else confidence[:len(p_score)]
        else:
            p_confidence = confidence
        
        if subsampling_weight is not None:
            if subsampling_weight.dim() > 1:
                subsampling_weight = subsampling_weight.reshape(-1)
            p_subsampling_weight = subsampling_weight[p_indices] if len(subsampling_weight) > max(p_indices) else subsampling_weight[:len(p_score)]
        else:
            p_subsampling_weight = None
        
        p_score = F.logsigmoid(p_score) * p_confidence
        if p_subsampling_weight is not None:
            p_loss = -(p_subsampling_weight * p_score).sum(dim=-1)/p_subsampling_weight.sum()
        else:
            p_loss = -p_score.mean()
            
        if neg_ratio > 0:
            n_confidence = p_confidence  
            
            n_subsampling_weight = p_subsampling_weight
            
            n_score = (F.softmax(n_score, dim=-1).detach()
                              * F.logsigmoid(-n_score)).sum(dim=-1) * n_confidence
            
            if n_subsampling_weight is not None:
                n_loss = -(n_subsampling_weight * n_score).sum(dim=-1) / n_subsampling_weight.sum()
            else:
                n_loss = -n_score.mean()
            loss = (p_loss + n_loss)/2
        else:
            loss = p_loss
        return loss


    def split_pn_score(self, score, label, neg_ratio):
        '''
        Get the scores of positive and negative facts
        :param score: scores of all facts
        :param label: positive facts: 1, negative facts: -1
        :return: p_score, n_score, p_indices, n_indices
        '''
        if neg_ratio <= 0:
            return score, 0, torch.arange(len(score), device=score.device), None
        
        p_mask = label > 0
        n_mask = label < 0
        
        if p_mask.dim() > 1:
            p_mask = p_mask.reshape(-1)
            n_mask = n_mask.reshape(-1)
        
        p_indices = torch.where(p_mask)[0]
        n_indices = torch.where(n_mask)[0]
        
        p_score = score[p_indices]
        n_score = score[n_indices].reshape(-1, neg_ratio)
        
        return p_score, n_score, p_indices, n_indices


class EntityAlignmentLoss(nn.Module):
    def __init__(self, args, kg):
        super(EntityAlignmentLoss, self).__init__()
        self.args = args
        self.kg = kg
        
    def forward(self, ent_embeddings, entity_pairs):
        if len(entity_pairs) == 0:
            return torch.tensor(0.0, device=ent_embeddings.device, requires_grad=True)
        
        pairs = torch.LongTensor(entity_pairs).to(ent_embeddings.device)
        e1_emb = ent_embeddings[pairs[:, 0]]  # [num_pairs, emb_dim]
        e2_emb = ent_embeddings[pairs[:, 1]]  # [num_pairs, emb_dim]
        
        e1_emb_norm = F.normalize(e1_emb, p=2, dim=1)
        e2_emb_norm = F.normalize(e2_emb, p=2, dim=1)
        
        cosine_sim = (e1_emb_norm * e2_emb_norm).sum(dim=1)  # [num_pairs]
        
        align_loss = (1 - cosine_sim).mean()
        
        return align_loss


class TripleInfoNCELoss(nn.Module):
    def __init__(self, args, kg, temperature=0.07):
        super(TripleInfoNCELoss, self).__init__()
        self.args = args
        self.kg = kg
        self.temperature = temperature
    
    def compute_query_emb(self, h_emb, r_emb):
        query_emb = h_emb + r_emb
        return F.normalize(query_emb, p=2, dim=1)
    
    def get_in_batch_negatives(self, t_emb, batch_size):
        in_batch_negs = t_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=t_emb.device)
        in_batch_negs = in_batch_negs[mask].reshape(batch_size, batch_size - 1, -1)
        
        return in_batch_negs
    
    def forward(self, h_emb, r_emb, t_emb):
        batch_size = h_emb.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=h_emb.device, requires_grad=True)
        
        query_emb = self.compute_query_emb(h_emb, r_emb)  # [batch_size, emb_dim]
        
        positive_emb = F.normalize(t_emb, p=2, dim=1)  # [batch_size, emb_dim]
        
        pos_sim = (query_emb * positive_emb).sum(dim=1, keepdim=True) / self.temperature  # [batch_size, 1]
        
        in_batch_negs = self.get_in_batch_negatives(t_emb, batch_size)  # [batch_size, batch_size-1, emb_dim]
        in_batch_negs_norm = F.normalize(in_batch_negs, p=2, dim=2)
        
        query_emb_expanded = query_emb.unsqueeze(1)  # [batch_size, 1, emb_dim]
        neg_sim = torch.bmm(query_emb_expanded, in_batch_negs_norm.transpose(1, 2)).squeeze(1) / self.temperature
        # [batch_size, batch_size-1]
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, batch_size]
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss



