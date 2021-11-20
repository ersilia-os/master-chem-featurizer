from molbert import Featurizer

f = Featurizer(standardise=True)
print(f.transform(["C"]))
