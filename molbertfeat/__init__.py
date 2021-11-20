__version__ = "0.0.1"

import os
from molbertfeat.utils.featurizer.molbert_featurizer import MolBertFeaturizer

PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(
    PATH, "..", "model", "molbert_100epochs", "checkpoints", "last.ckpt"
)
REFERENCE_SMILES = os.path.join(PATH, "..", "data", "chembl_29_chemreps.txt")


class Featurizer(object):
    def __init__(self, standardise: bool = False):
        self.model = MolBertFeaturizer(CHECKPOINT, assume_standardised=not standardise)

    def transform(self, smiles_list):
        features, masks = self.model.transform(smiles_list)
        return features
