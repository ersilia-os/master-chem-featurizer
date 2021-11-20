from mcf import ReferenceLibrary

rl = ReferenceLibrary()
rl.save("../reflib")
print(rl.load())
