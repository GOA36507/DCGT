from src.utils import *
from torch_scatter import scatter


class KgeEmbedder(torch.nn.Module):
    def __init__(self, args, kg):
        super().__init__()
        self.args = args
        self.kg = kg
        self.num_ent = kg.num_ent
        self.num_rel = kg.num_rel

        self.ent_dim = args.emb_dim
        self.rel_dim = args.emb_dim

    def embed_ent(self, indexes):
        pass

    def embed_rel(self, indexes):
        pass

    def embed_ent_all(self, indexes=None):
        pass

    def embed_rel_all(self, indexes=None):
        pass


class LookupEmbedder(KgeEmbedder):
    def __init__(self, args, kg):
        super().__init__(args, kg)

    def _create_embedding(self):
        self.ent_embeddings = nn.Embedding(self.num_ent, self.ent_dim).to(self.args.device)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.rel_dim).to(self.args.device)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.args.margin + 2.0) / self.ent_dim]),
            requires_grad=False
        )
        uniform_(self.ent_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        uniform_(self.rel_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        return self.ent_embeddings(indexes)

    def embed_rel(self, indexes):
        return self.rel_embeddings(indexes)

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None):
        return self.ent_embeddings.weight

    def embed_rel_all(self, indexes=None):
        rel_embeddings = self.rel_embeddings.weight
        return torch.cat([rel_embeddings[0::2], rel_embeddings[0::2]], dim=-1).reshape(-1, rel_embeddings.size(-1))

    def embed_ent_prototype(self, indexes, kg=None, scores=None, scorer=None, query_rel=None, mode=None):
        if kg is None or kg.all() or not kg.any():
            ent_embeddings = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
            edge_s = self.kg.edge_s.to(self.args.device)
            edge_r = self.kg.edge_r.to(self.args.device)

            s_embeddings = torch.index_select(ent_embeddings, 0, edge_s)
            proto_embeddings = scatter(src=s_embeddings, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')
            return proto_embeddings[indexes]
        else:  # for llemapping
            edge_s = self.kg.edge_s.to(self.args.device)
            edge_r = self.kg.edge_r.to(self.args.device)
            ent_embeddings_1, ent_embeddings_2 = self.embed_ent_all_double(kg=kg)
            # for kg 1
            s_embeddings_1 = torch.index_select(ent_embeddings_1, 0, edge_s)
            proto_embeddings_1 = scatter(src=s_embeddings_1, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')
            # for kg 2
            s_embeddings_2 = torch.index_select(ent_embeddings_2, 0, edge_s)
            proto_embeddings_2 = scatter(src=s_embeddings_2, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')

            # fill prototype embeddings
            proto = torch.zeros([indexes.size(0), ent_embeddings_1.size(-1)], dtype=torch.float).to(self.args.device)
            proto[~kg] = proto_embeddings_1[indexes[~kg]]
            proto[kg] = proto_embeddings_2[indexes[kg]]
            return proto

    def forward(self, **kwargs):
        jobs = kwargs['jobs']
        res = dict()
        for job, value in jobs.items():
            opt_name = value['opt']
            if opt_name == 'ent_embedding':
                opt = self.embed_ent
            elif opt_name == 'rel_embedding':
                opt = self.embed_rel
            elif opt_name == 'ent_embedding_all':
                opt = self.embed_ent_all
            elif opt_name == 'rel_embedding_all':
                opt = self.embed_rel_all
            elif opt_name == 'ent_embedding_prototype':
                opt = self.embed_ent_prototype
            else:
                raise('Invalid embedding opration!')
            input = value['input']
            res[job] = opt(input['indexes'])
        return res

class LookupEmbedderGAT(LookupEmbedder):
    def __init__(self, args, kg):
        super().__init__(args, kg)
        self.drop = torch.nn.Dropout(p=0.0, inplace=False)
        
        self.W_intra = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_align = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_inter = nn.Linear(args.emb_dim, args.emb_dim)
        
        self.cached_emb = None          
        self.cache_version = -1         
        self.emb_version = 0            
        
        self.edge_s_intra = None
        self.edge_r_intra = None
        self.edge_o_intra = None
        self.edge_s_inter = None
        self.edge_r_inter = None
        self.edge_o_inter = None
        
        self.align_src_indices = None   
        self.align_tgt_indices = None   
    
    def _preprocess_edges(self):
        if self.edge_s_intra is None:
            self.edge_s_intra = self.kg.edge_s.to(self.args.device).long()
            self.edge_r_intra = self.kg.edge_r.to(self.args.device).long()
            self.edge_o_intra = self.kg.edge_o.to(self.args.device).long()
    
    def _preprocess_alignment(self):
        if self.align_src_indices is None and hasattr(self.kg, 'ent2align') and len(self.kg.ent2align) > 0:
            src_list = []
            tgt_list = []
            for src, tgt in self.kg.ent2align.items():
                src_list.append(src)
                tgt_list.append(tgt)
            if len(src_list) > 0:
                self.align_src_indices = torch.LongTensor(src_list).to(self.args.device)
                self.align_tgt_indices = torch.LongTensor(tgt_list).to(self.args.device)
    
    def _preprocess_inter_edges(self):
        use_inter_triples = getattr(self.args, 'use_inter_triples', False)
        if use_inter_triples and self.edge_s_inter is None:
            if hasattr(self.kg, 'inter_edge_s') and self.kg.inter_edge_s is not None and len(self.kg.inter_edge_s) > 0:
                self.edge_s_inter = self.kg.inter_edge_s.to(self.args.device).long()
                self.edge_r_inter = self.kg.inter_edge_r.to(self.args.device).long()
                self.edge_o_inter = self.kg.inter_edge_o.to(self.args.device).long()
    
    def invalidate_cache(self):
        self.emb_version += 1

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        if self.cached_emb is None or self.cache_version != self.emb_version:
            self.cached_emb = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
            self.cache_version = self.emb_version
        return self.cached_emb[indexes]

    def _compute_message(self, X, R, edge_s, edge_r, edge_o, W_transform):
        N, d = X.size(0), X.size(1)
        E = edge_s.size(0)
        
        if E == 0:
            return torch.zeros(N, d, device=X.device)
        
        h_u = X[edge_o]  
        h_r = R[edge_r]  
        
        
        phi_ur = h_u - h_r  # [E, d]
        
        msg = W_transform(phi_ur)  # [E, d]
        msg = self.drop(msg)
        
        aggregated_msg = scatter(msg, index=edge_s, dim=0, dim_size=N, reduce='sum')  # [N, d]
        
        return aggregated_msg

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None):
        self._preprocess_edges()
        self._preprocess_alignment()
        self._preprocess_inter_edges()
        
        X = super().embed_ent_all()  # [N, d]
        R = self.rel_embeddings.weight  # [num_rel, d]
        
        msg_intra = self._compute_message(X, R, self.edge_s_intra, self.edge_r_intra, 
                                          self.edge_o_intra, self.W_intra)  # [N, d]
        
        msg_align = torch.zeros_like(X)  # [N, d]
        if self.align_src_indices is not None:
            h_align = X[self.align_tgt_indices]  
            msg_align_values = self.W_align(h_align)  
            msg_align.index_add_(0, self.align_src_indices, msg_align_values)
        
        msg_inter = torch.zeros_like(X)  # [N, d]
        if self.edge_s_inter is not None:
            msg_inter = self._compute_message(X, R, self.edge_s_inter, self.edge_r_inter, 
                                             self.edge_o_inter, self.W_inter)  # [N, d]
        
        aggregated_msg = msg_intra + msg_align + msg_inter  # [N, d]
        
        X_new = X + torch.relu(aggregated_msg)  # [N, d]
        
        return X_new
