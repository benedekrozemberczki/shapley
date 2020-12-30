import numpy as np
import math
from shapley import PermutationSampler


def test_permutation_sampling():
    """
    Testing the Permutation Sampler class.
    """

    solver = PermutationSampler()

    W = np.random.uniform(0, 1, (100, 100))
    solver.solve_game(W, q = 10)
    Phi = solver.get_solution()
    Phi_tilde = solver.get_average_shapley()
    entropy = solver.get_shapley_entropy()

    assert Phi.shape == W.shape
    assert Phi_tilde.shape == (W.shape[1],)
    assert -math.log(1.0/W.shape[1])-entropy > 0

    solver = PermutationSampler(permutations=100)

    W = np.random.uniform(0, 1, (100, 50))
    solver.solve_game(W, q = 10)
    Phi = solver.get_solution()
    Phi_tilde = solver.get_average_shapley()
    entropy = solver.get_shapley_entropy()

    assert Phi.shape == W.shape
    assert Phi_tilde.shape == (W.shape[1],)
    assert -math.log(1.0/W.shape[1])-entropy > 0

    solver = PermutationSampler(permutations=10000)

    W = np.random.uniform(0, 1, (10, 13))
    solver.solve_game(W, q = 3)
    Phi = solver.get_solution()
    Phi_tilde = solver.get_average_shapley()
    entropy = solver.get_shapley_entropy()

    assert Phi.shape == W.shape
    assert Phi_tilde.shape == (W.shape[1],)
    assert -math.log(1.0/W.shape[1])-entropy > 0
