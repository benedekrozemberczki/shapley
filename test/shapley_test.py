import numpy as np
from shapley import PermutationSampling


def test_permutation_sampling():
    """
    Testing the Permutation Sampler class.
    """
    solver = PermutationSampling(a = 1, b = 2)

    out = solver.solve_game()

    assert out == 3
