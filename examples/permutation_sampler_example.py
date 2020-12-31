"""Permutation Sampler Example"""

import numpy as np
from shapley import PermutationSampler

W = np.random.uniform(0, 1, (1, 7))
W = W/W.sum()
q = 0.5

solver = PermutationSampler()
solver.solve_game(W, q)
shapley_values = solver.get_solution()

print(shapley_values)
