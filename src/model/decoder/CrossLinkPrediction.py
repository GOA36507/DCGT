from src.model.decoder.base_decoder import *


class CrossLinkPrediction(torch.nn.Module):
    def __init__(self, args, kg):
        super().__init__()
        self.args = args
        self.kg = kg
        self.scorer = self._create_scorer()

    def _create_scorer(self):
        if self.args.scorer == 'TransE':
            return TransEScorer(self.args, self.kg)

    def forward(self, feat, mode, margin):
        s, r, o = feat
        score = self.scorer(s, r, o, mode, margin)
        return score





