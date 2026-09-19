import torch
from .utils import *
from .model.model_process import CrossLinkPredictionTestBatchProcessor


class Tester():
    def __init__(self, args, kg, model):
        self.args = args
        self.kg = kg
        self.model = model
        self.test_processor = CrossLinkPredictionTestBatchProcessor(args, kg)

    def test(self):
        torch.cuda.empty_cache()
        self.args.valid = False
        res = self.test_processor.process_epoch(self.model)
        self.args.valid = True

        return res


