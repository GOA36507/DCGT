from src.utils import *


class BaseScorer(torch.nn.Module):
    def __init__(self, args, kg):
        super().__init__()
        self.args = args
        self.kg = kg

    def norm(self, x):
        return nn.functional.normalize(x, 2, -1)
