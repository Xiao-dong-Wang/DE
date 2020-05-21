import numpy as np
import sys
import toml
import pickle
from util import get_funct
from DE import DE

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['funct']
funct = get_funct(name)
NP = conf['NP']
max_iter = conf['max_iter']
bounds = np.array(conf['bounds'])
lb = bounds[:,0]
ub = bounds[:,1]
F = conf['F']
CR = conf['CR']

solver = DE(funct, NP, max_iter, lb, ub, F, CR)
solver.optimize()
dataset = solver.dataset

with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)



