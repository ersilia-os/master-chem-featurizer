import sys
import os
from .molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(PATH, "model", "molbert_100epochs", "checkpoints", "last.ckpt")


class Featurizer(object):

    def __init__(self):
        self.f = MolBertFeaturizer(CHECKPOINT)

    def transform(self, smiles_list):
        features, masks = f.transform(smiles_list)
        assert all(masks)
        return features        
