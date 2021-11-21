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
    def __init__(self, file_name=None, max_molecules=100000000, chunksize=1000, write_chunksize=10000, standardise=False):
        assert chunksize <= write_chunksize
        if file_name is None:
            self.file_name = REFERENCE_SMILES
        else:
            self.file_name = file_name
        self.max_molecules = max_molecules
        self.chunksize = chunksize
        self.standardise = standardise

    def _read_file_only_valid(self):
        smiles_list = []
        with open(self.file_name, "r") as f:
            reader = csv.reader(f)
            for i, r in enumerate(reader):
                smiles_list += [r[0]]
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
        val_smiles = list(set(val_smiles)) # Unique
        return val_smiles

    def _read_file_assuming_valid(self):
        smiles_list = []
        with open(self.file_name, "r") as f:
            reader = csv.reader(f)
            for i, r in enumerate(reader):
                smiles_list += [r[0]]
        return smiles_list

    def save_only_smiles(self, csv_file):
        self.mdl = Featurizer(standardise=True)
        smiles_list = self._read_file_only_valid()
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            for smi in smiles_list:
                writer.writerow([smi])

    def _get_already_done_inputs(self, h5_file):
        with h5py.File(h5_file, "r") as f:
            inps = f["Inputs"][:]
            inps = set([x.decode("utf-8") for x in inps])
        return inps

    def _get_todo_smiles(self, smiles_list, h5_file):
        done_smiles = self._get_already_done_inputs(h5_file)
        todo_smiles = []
        for smi in smiles_list:
            if smi in done_smiles:
                todo_smiles += [smi]
        return todo_smiles

    def chunked_iterable(self, seq):
        return (seq[pos:pos + self.chunksize] for pos in range(0, len(seq), self.write_chunksize))

    def append_to_h5(self, h5_file, X, smiles_list):
        with open(h5_file, "a") as f:
            smiles_list = np.array(smiles_list, h5py.string_dtype())
            f["Values"].resize(f["Values"].shape[0] + X.shape[0], axis=0)
            f["Values"][-X.shape[0]:] = X
            f["Inputs"].resize(f["Inputs"].shape[0] + smiles_list.shape[0], axis=0)
            f["Inputs"][-smiles_list.shape[0]:] = smiles_list

    def write_to_h5(self, h5_file, X, smiles_list):
        with open(h5_file, "w") as f:
            smiles_list = np.array(smiles_list, h5py.string_dtype())
            f.create_dataset("Values", data=X)
            f.create_dataset("Inputs", data=smiles_list)

    def save(self, h5_file, append=True):
        assert h5py is not None
        self.mdl = Featurizer(standardise=self.standarise, chunksize=self.chunksize)
        if self.standardise:
            smiles_list = self._read_file_only_valid()
        else:
            smiles_list = self._read_file_assuming_valid()
        file_exists = os.path.isfile(h5_file)
        if file_exists and append:
            smiles_list = self._get_todo_smiles(smiles_list, h5_file)
            for chunk in self.chunked_iterable(smiles_list):
                X = self.mdl.transform(chunk)
                self.append_to_h5(h5_file, X, chunk)
        else:
            if file_exists:
                os.remove(h5_file)
            for i, chunk in self.chunked_iterable(smiles_list):
                X = self.mdl.transform(chunk)
                if i == 0:
                    self.write_to_h5(h5_file, X, chunk)
                else:
                    self.append_to_h5(h5_file, X, chunk)

    def read(self, h5_file):
        assert h5py is not None
        with h5py.File(h5_file, "r") as f:
            X = f["Values"][:]
            smiles_list = f["Inputs"][:]
            smiles_list = [x.decode("utf-8") for x in smiles_list]
        return X, smiles_list
