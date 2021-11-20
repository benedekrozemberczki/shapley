"""Generic shape and entropy tests."""

import pytest
import unittest

import math
import numpy as np
from shapley import ExactEnumeration
from shapley import PermutationSampler
from shapley import MultilinearExtension
from shapley import ExpectedMarginalContributions


class TestShapley(unittest.TestCase):

    def test_permutation_sampling(self):
        """
        Testing the Permutation Sampler class.
        """
        solver = PermutationSampler()

        W = np.random.uniform(0, 1, (100, 100))
        solver.solve_game(W, q=50)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0

        solver = PermutationSampler(permutations=10000)

        W = np.random.uniform(0, 1, (100, 50))
        solver.solve_game(W, q=20)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0

        solver = PermutationSampler(permutations=10000)

        W = np.random.uniform(0, 1, (10, 13))
        solver.solve_game(W, q=3)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0


    def test_multilinear_extension(self):
        """
        Testing the Multilinear Extension class.
        """

        solver = MultilinearExtension()

        W = np.random.uniform(0, 1, (100, 97))
        solver.solve_game(W, q=0.25)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0

        solver = MultilinearExtension()

        W = np.random.uniform(0, 1, (100, 48))
        solver.solve_game(W, q=0.25)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0

        solver = MultilinearExtension()

        W = np.random.uniform(0, 1, (10, 13))
        solver.solve_game(W, q=0.13)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0

    def test_exact_enumeration(self):
        """
        Testing the Exact Enumeration class.
        """

        solver = ExactEnumeration()

        W = np.random.uniform(0, 1, (100, 7))
        solver.solve_game(W, q=2.5)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > -0.001

        solver = ExactEnumeration()

        W = np.random.uniform(0, 1, (100, 6))
        solver.solve_game(W, q=2)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > -0.001

        solver = ExactEnumeration()

        W = np.random.uniform(0, 1, (10, 5))
        solver.solve_game(W, q=3)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > -0.001


    def test_expected_marginal_contributions(self):
        """
        Testing the Expected Marginal Contributions class.
        """

        solver = ExpectedMarginalContributions()

        W = np.random.uniform(0, 1, (100, 97))
        solver.solve_game(W, q=15.5)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0

        solver = ExpectedMarginalContributions()

        W = np.random.uniform(0, 1, (100, 48))
        solver.solve_game(W, q=12.5)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0

        solver = ExpectedMarginalContributions(epsilon=10 ** -5)

        W = np.random.uniform(0, 1, (10, 13))
        solver.solve_game(W, q=3.0)
        Phi = solver.get_solution()
        Phi_tilde = solver.get_average_shapley()
        entropy = solver.get_shapley_entropy()

        assert Phi.shape == W.shape
        assert Phi_tilde.shape == (W.shape[1],)
        assert -math.log(1.0 / W.shape[1]) - entropy > 0
