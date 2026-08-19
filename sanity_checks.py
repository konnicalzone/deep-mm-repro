from data_many_to_one import Data

G = Data(
    n_students=4,
    n_schools=3,
    min_capacity=1,
    max_capacity=2,
    prob=0.5,
    corr=0.5,
)

P, Q, C = G.generate_batch(batch_size=2)

assert P.shape == (2, 4, 3)
assert Q.shape == (2, 4, 3)
assert C.shape == (2, 3)
assert C.min() >= 1
assert C.max() <= 2

P_mis, Q_mis, C_mis = G.generate_complete_misreports(P, Q, C, student_idx=1)

num_misreports = G.generate_complete_ranking().shape[0]

assert P_mis.shape == (2, num_misreports, 4, 3)
assert Q_mis.shape == (2, num_misreports, 4, 3)
assert C_mis.shape == (2, num_misreports, 3)

P_mis, Q_mis, C_mis = G.generate_complete_misreports(P, Q, C, student_idx=1)

num_misreports = G.generate_complete_ranking().shape[0]

assert P_mis.shape == (2, num_misreports, 4, 3)
assert Q_mis.shape == (2, num_misreports, 4, 3)
assert C_mis.shape == (2, num_misreports, 3)

import numpy as np

assert np.all(Q_mis == Q[:, None, :, :])
assert np.all(C_mis == C[:, None, :])

import torch
import torch.nn as nn
from MLP import Net

model = Net([16], nn.LeakyReLU, n_students=4, n_schools=3)

p = torch.Tensor(P)
q = torch.Tensor(Q)
capacity = torch.Tensor(C)

r = model(p, q, capacity)

assert r.shape == (2, 4, 3)
assert torch.all(r.sum(dim=2) <= 1 + 1e-6)          # each student <= 1
assert torch.all(r.sum(dim=1) <= capacity + 1e-6)

import train_MLP
train_MLP.device = "cpu"
train_MLP.G = G
train_MLP.model = model

st = train_MLP.compute_st(r, p, q, capacity)
ic = train_MLP.compute_ic(r, p, q, P, Q, C)

assert torch.isfinite(st)
assert torch.isfinite(ic)

print("sanity ok")