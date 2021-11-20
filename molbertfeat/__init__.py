__version__ = "0.0.1"

import os
import numpy as np
from tqdm import tqdm
from molbertfeat.utils.featurizer.molbert_featurizer import MolBertFeaturizer

PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(
    PATH, "..", "model", "molbert_100epochs", "checkpoints", "last.ckpt"
)
REFERENCE_SMILES = os.path.join(PATH, "..", "data", "chembl_29_chemreps.txt")

EMBEDDING_SIZE = 768

class Featurizer(object):
    def __init__(self, standardise: bool = False, chunksize: int = 1000):
        self.model = MolBertFeaturizer(CHECKPOINT, assume_standardised=not standardise)
        self.chunksize = chunksize
        
    def chunked_iterable(self, seq):
        return (seq[pos:pos + self.chunksize] for pos in range(0, len(seq), self.chunksize))

    def transform(self, smiles_list):
        X = np.zeros((len(smiles_list), EMBEDDING_SIZE), np.float32)
        idxs = np.array([i for i in range(len(smiles_list))], np.int8)
        for chunk in tqdm(self.chunked_iterable(idxs)):
            X[chunk], _ = self.model.transform([smiles_list[i] for i in chunk])
        return X
