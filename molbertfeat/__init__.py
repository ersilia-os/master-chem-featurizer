__version__ = "0.0.1"

import os
import numpy as np
from tqdm import tqdm
from molbertfeat.utils.featurizer.molbert_featurizer import MolBertFeaturizer

PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(
    PATH, "..", "model", "molbert_100epochs", "checkpoints", "last.ckpt"
)
REFERENCE_SMILES = os.path.join(PATH, "..", "data", "reference_library.csv")

EMBEDDING_SIZE = 768

class Featurizer(object):
    def __init__(self, standardise: bool = False, chunksize: int = 1000):
        self.model = MolBertFeaturizer(CHECKPOINT, assume_standardised=not standardise)
        self.chunksize = chunksize

    def chunker(self, n):
        size = self.chunksize
        for i in range(0, n, size):
            yield slice(i, i + size)

    def transform(self, smiles_list):
        X = np.zeros((len(smiles_list), EMBEDDING_SIZE), np.float32)
        for chunk in tqdm(self.chunker(X.shape[0])):
            X[chunk], _ = self.model.transform(smiles_list[chunk])
        return X
