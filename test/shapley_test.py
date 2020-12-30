import numpy as np
from shapley import PermutationSampler


def test_permutation_sampling():
    """
    Testing the Permutation Sampler class.
    """

    solver = PermutationSampler()

    W = np.random.uniform(0,1,(100, 10))
    Phi = solver.solve_game(W, q = 50)
    Phi_tilde = solver.get_average_shapley()
    entropy = solver.get_shapley_entropy()

    assert Phi.shape == W.shape
    assert Phi_tilde.shape = (W.shape[0],)

    solver = PermutationSampler(permutations=10)

    W = np.random.uniform(0,1,(100, 100))
    Phi = solver.solve_game(W, q = 50)
    Phi_tilde = solver.get_average_shapley()
    entropy = solver.get_shapley_entropy()

    assert Phi.shape == W.shape
    assert Phi_tilde.shape = (W.shape[0],)

    solver = PermutationSampler(permutations=10000)

    W = np.random.uniform(0,1,(10, 10))
    Phi = solver.solve_game(W, q = 7)
    Phi_tilde = solver.get_average_shapley()
    entropy = solver.get_shapley_entropy()

    assert Phi.shape == W.shape
    assert Phi_tilde.shape = (W.shape[0],)
