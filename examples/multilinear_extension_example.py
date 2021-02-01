"""Multilinear Extension Example"""

import numpy as np
from shapley import MultilinearExtension

W = np.random.uniform(0, 1, (20, 20))
W = W/W.sum(1)
q = 0.35

solver = MultilinearExtension()
solver.solve_game(W, q)
shapley_values = solver.get_solution()

print(shapley_values)
