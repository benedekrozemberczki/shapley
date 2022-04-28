"""Permutation Sampling Based Shapley Value Approximation."""

import random
import numpy as np
from shapley.solution_concept import SolutionConcept


class PermutationSampler(SolutionConcept):
    r"""Permutation sampler to solve a block of weighted voting games. The solver
    samples random permutations of the players uniformly. In each permutation we
    identify the pivotal voter. We use the empirical probability of becoming the
    pivotal player as the Shapley value estimate. For details see this paper:
    `"Bounding the Estimation Error of Sampling-based Shapley Value Approximation."
    <https://arxiv.org/abs/1306.4265>`_

    Args:
        permutations (int): Number of permutations. The default is 1000.
    """

    def __init__(self, permutations: int = 1000):
        self.permutations = permutations

    def setup(self, W: np.ndarray):
        """Creating an empty Shapley value matrix and a player pool."""
        self._Phi = np.zeros(W.shape)
        self._indices = [i for i in range(W.shape[1])]

    def _run_permutations(self, W: np.ndarray, q: float):
        """Creating Monte Carlo permutations and finding the marginal voter."""
        for _ in range(self.permutations):
            random.shuffle(self._indices)
            W_perm = W[:, self._indices]
            cum_sum = np.cumsum(W_perm, axis=1)
            pivotal = np.array(self._indices)[np.argmax(cum_sum > q, axis=1)]
            self._Phi[np.arange(W.shape[0]), pivotal] += 1.0
        self._Phi = self._Phi / self.permutations

    def solve_game(self, W: np.ndarray, q: float):
        r"""Solving the weigted voting game(s).

        Args:
            W (Numpy array): An :math:`n \times m` matrix of voting weights for the :math:`n` games with :math:`m` players.
            q (float): Quota in the games.
        """
        self._check_quota(q)
        self.setup(W)
        self._run_permutations(W, q)
        self._run_sanity_check(W, self._Phi)
        self._set_average_shapley()
        self._set_shapley_entropy()

    def get_solution(self) -> np.ndarray:
        r"""Returning the solution.

        Return Types:
            Phi (Numpy array): Approximate Shapley matrix of players in the game(s) with size :math:`n \times m`.
        """
        return self._Phi
