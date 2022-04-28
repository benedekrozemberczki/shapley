"""Exact Enumeration Based Shapley Value."""

import itertools
import numpy as np
from shapley.solution_concept import SolutionConcept


class ExactEnumeration(SolutionConcept):
    r"""Exact enumeration of all permutations and finding the pivotal voters. It
    is designed with a generator of the permutations. For details see this paper:
    `"A Value for N-Person Games." <https://www.rand.org/pubs/papers/P0295.html>`_
    """

    def setup(self, W: np.ndarray):
        """Creating an empty Shapley value matrix and a player pool."""
        self._Phi = np.zeros(W.shape)
        self._indices = [i for i in range(W.shape[1])]
        self.permutations = 0

    def _run_permutations(self, W: np.ndarray, q: float):
        """Creating Monte Carlo permutations and finding the marginal voter."""
        for perm in itertools.permutations(self._indices):
            self.permutations = self.permutations + 1
            indices = list(perm)
            W_perm = W[:, indices]
            cum_sum = np.cumsum(W_perm, axis=1)
            pivotal = np.array(indices)[np.argmax(cum_sum > q, axis=1)]
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
