from src.utils import *
from src.model.encoder.embedder import *
from src.model.decoder.CrossLinkPrediction import *
from src.model.loss_func import *


class Controller(torch.nn.Module):
    def __init__(self, args, kg):
        super().__init__()
        self.args = args
        self.kg = kg
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.loss_func = self._create_loss_func()

    def _create_encoder(self):
        if self.args.encoder.lower() == 'lookup_gat':
            encoder = LookupEmbedderGAT(self.args, self.kg)
            encoder._create_embedding() 
        else:
            raise('Invalid encoder name:{}'.format(self.args.encoder))
        return encoder
        

    def _create_decoder(self):
        decoder = CrossLinkPrediction(self.args, self.kg)
        return decoder

    def _create_loss_func(self):
        loss_func = MarginLoss(self.args, self.kg, self.encoder)
        return loss_func

    def process_feat(self, feat):
        return feat['sub_emb'], feat['rel_emb'], feat['obj_emb']

    def forward(self, **kwargs):
        kwargs['scorer'] = self.decoder.scorer
        feat = self.encoder(**kwargs)
        mode = kwargs['mode']
        margin = kwargs['margin']
        pred = self.decoder(self.process_feat(feat), mode=mode, margin=margin)
        return pred

    def loss(self, pred, label, subsampling_weight=None, confidence=None, neg_ratio=10):
        return self.loss_func(pred, label.reshape(pred.shape), subsampling_weight=subsampling_weight, confidence=confidence, neg_ratio=neg_ratio)

    def predict(self, **kwargs):
        feat = self.encoder(**kwargs)
        mode = kwargs['mode']
        pred = self.decoder.predict(self.process_feat(feat), mode=mode)
        return pred


