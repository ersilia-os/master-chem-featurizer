__version__ = '0.0.1'

import sys
import os
import csv
import h5py
import numpy as np

from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(PATH, "..", "model", "molbert_100epochs", "checkpoints", "last.ckpt")
REFERENCE_SMILES = os.path.join(PATH, "..", "data", "chembl_29_chemreps.txt")

REFERENCE_DATA = "data.h5"


class Featurizer(object):

    def __init__(self):
        self.model = MolBertFeaturizer(CHECKPOINT)

    def transform(self, smiles_list):
        features, masks = self.model.transform(smiles_list)
        return features


class ReferenceLibrary(object):

    def __init__(self, file_name = None):
        if file_name is None:
            self.file_name = REFERENCE_SMILES
        else:
            self.file_name = file_name

    def _read_file_only_valid(self, dir_path):
        smiles_list = []
        with open(self.file_name, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                smiles_list += [r[1]]
        smiles_list = smiles_list[:10]
        f = self.mdl.model.featurizer
        std_smiles = []
        for smi in smiles_list:
            smi = f.standardise(smi)
            if smi is not None:
                std_smiles += [smi]
        val_smiles = []
        for standard_smiles in std_smiles:
            single_char_smiles = f.encode(standard_smiles)
            decorated_smiles = f.decorate(list(single_char_smiles))
            valid_smiles = f.is_legal(standard_smiles) and f.is_short(decorated_smiles)
            if valid_smiles:
                val_smiles += [standard_smiles]
        return val_smiles

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        self.mdl = Featurizer()
        smiles_list = self._read_file_only_valid(dir_path)
        X = self.mdl.transform(smiles_list)
        with h5py.File(os.path.join(dir_path, REFERENCE_DATA), "w") as f:
            f.create_dataset("Values", data=X)
            smiles_list = np.array(smiles_list, h5py.string_dtype())
            f.create_dataset("Inputs", data=smiles_list)

    def read(self, dir_path):
        with h5py.File(os.path.join(dir_path, REFERENCE_DATA), "r") as f:
            X = f["Values"][:]
            smiles_list = f["Inputs"][:]
            smiles_list = [x.decode("utf-8") for x in smiles_list]
        return X, smiles_list
