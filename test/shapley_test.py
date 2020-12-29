import numpy as np
from shapley import PermutationSampler


def test_permutation_sampling():
    """
    Testing the Permutation Sampler class.
    """

    solver = PermutationSampler()

    W = np.random.uniform(0,1,(100, 10))
    Phi = solver.solve_game(W, q = 0.5)

    assert Phi.shape == W.shape
