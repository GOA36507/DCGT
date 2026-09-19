from src.utils import *
from src.model.decoder.triplet_scorer.base_scorer import BaseScorer


class TransEScorer(BaseScorer):
    def __init__(self, args, kg):
        super().__init__(args, kg)
        self.p = 1

    def forward(self, s, r, o, mode=None, margin=6.0):
        if mode == 'head-batch' or o.shape==r.shape:
            pred = o - r
            real = s
        elif mode == 'tail-batch' or s.shape==r.shape:
            pred = s + r
            real = o

        if pred.shape == real.shape:
            score = margin - torch.norm(pred - real, p=1, dim=-1)
        else:
            score = margin - torch.norm(pred.unsqueeze(1) - real, p=1, dim=-1)
        return score

    def decode(self, s, r, modes):
        r[modes] *= -1
        pred = s + r
        return pred




