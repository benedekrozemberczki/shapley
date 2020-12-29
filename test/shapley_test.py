import numpy as np
from shapley import PermutationSampler


def test_permutation_sampling():
    """
    Testing the Permutation Sampler class.
    """

    solver = PermutationSampler()

    W = np.random.uniform(0,1,(100, 10))
    Phi = solver.solve_game(W, q = 50)

    assert Phi.shape == W.shape

    solver = PermutationSampler(permutations=10)

    W = np.random.uniform(0,1,(100, 100))
    Phi = solver.solve_game(W, q = 50)

    assert Phi.shape == W.shape

    solver = PermutationSampler(permutations=10000)

    W = np.random.uniform(0,1,(10, 10))
    Phi = solver.solve_game(W, q = 7)

    assert Phi.shape == W.shape
