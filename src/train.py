from .model.model_process import *


class Trainer:
    def __init__(self, args, kg, model, optimizer):
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        self.valid_processor = CrossLinkPredictionValidBatchProcessor(self.args, self.kg)
        self.train_processor = CrossLinkPredictionTrainBatchProcessor(args, kg)

    def run_epoch(self):
        torch.cuda.empty_cache()
        self.args.valid = True

        # train for facts
        loss, num_known_pairs, num_pseudo_pairs, triple_stats = self.train_processor.process_epoch(self.model, self.optimizer)
        torch.cuda.empty_cache()
        res = self.valid_processor.process_epoch(self.model)
        self.args.valid = False
        return loss, res, num_known_pairs, num_pseudo_pairs, triple_stats



