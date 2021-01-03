"""Expected Marginal Contributions Example"""

import numpy as np
from shapley import ExpectedMarginalContributions

W = np.random.uniform(0, 1, (1, 20))
W = W/W.sum()
q = 0.35

solver = ExpectedMarginalContributions()
solver.solve_game(W, q)
shapley_values = solver.get_solution()

print(shapley_values)
