import sys
import os
import csv
import numpy as np
import itertools

try:
    import h5py
except:
    h5py = None

from . import Featurizer

from . import REFERENCE_SMILES
from tqdm import tqdm


class ReferenceLibrary(object):
    def __init__(self, file_name=None, max_molecules=100000000, chunksize=1000):
        if file_name is None:
            self.file_name = REFERENCE_SMILES
        else:
            self.file_name = file_name
        self.max_molecules = max_molecules
        self.chunksize = chunksize

    def _read_file_only_valid(self):
        smiles_list = []
        with open(self.file_name, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for i, r in enumerate(reader):
                smiles_list += [r[1]]
        smiles_list = smiles_list[:self.max_molecules]
        f = self.mdl.model.featurizer
        std_smiles = []
        for smi in tqdm(smiles_list):
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

    def save_only_smiles(self, csv_file):
        self.mdl = Featurizer(standardise=True)
        smiles_list = self._read_file_only_valid()
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            for smi in smiles_list:
                writer.writerow([smi])

    def save(self, h5_file):
        assert h5py is not None
        self.mdl = Featurizer(standardise=True, chunksize=self.chunksize)
        smiles_list = self._read_file_only_valid()
        X = self.mdl.transform(smiles_list)
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("Values", data=X)
            smiles_list = np.array(smiles_list, h5py.string_dtype())
            f.create_dataset("Inputs", data=smiles_list)

    def read(self, h5_file):
        assert h5py is not None
        with h5py.File(h5_file, "r") as f:
            X = f["Values"][:]
            smiles_list = f["Inputs"][:]
            smiles_list = [x.decode("utf-8") for x in smiles_list]
        return X, smiles_list
