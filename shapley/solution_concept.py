"""Solution Concept base class."""


import numpy as np
from abc import ABCMeta, abstractmethod


class SolutionConcept(metaclass=ABCMeta):
    """Solution Concept base class with constructor and public methods."""

    @abstractmethod
    def get_solution(self) -> np.ndarray:
        r"""Returning the solution.

        Return Types:
            Phi (Numpy array): Approximate Shapley matrix of players in the game(s) with size :math:`n \times m`.
        """
        pass

    @abstractmethod
    def solve_game(self, W: np.ndarray, q: float):
        r"""Solving the weigted voting game(s).

        Args:
            W (Numpy array): An :math:`n \times m` matrix of voting weights for the :math:`n` games with :math:`m` players.
            q (float): Quota in the games.
        """
        pass

    @abstractmethod
    def setup(self, W: np.ndarray):
        """Creating an empty Shapley value matrix and a player pool."""
        pass

    def _check_quota(self, q: float):
        """Checking for negative quota."""
        assert 0.0 <= q

    def _verify_result_shape(self, W: np.ndarray, Phi: np.ndarray):
        """Checking the shape of the Shapley value matrix."""
        assert W.shape == Phi.shape

    def _verify_distribution(self, Phi: np.ndarray):
        """Verify distribution hypothesis."""
        assert (np.sum(Phi) - Phi.shape[0]) < 0.0001

    def _run_sanity_check(self, W: np.ndarray, Phi: np.ndarray):
        """Checking the basic assumptions about the Shapley values."""
        self._verify_result_shape(W, Phi)
        self._verify_distribution(Phi)

    def _set_average_shapley(self):
        """Calculating the average Shapley value scores."""
        self._Phi_tilde = np.mean(self._Phi, axis=0)

    def _set_shapley_entropy(self):
        """Calculating the Shapley entropy score."""
        self._shapley_entropy = -np.sum(self._Phi_tilde * np.log(self._Phi_tilde))

    def get_average_shapley(self) -> np.ndarray:
        """Getting the average Shapley value scores."""
        return self._Phi_tilde

    def get_shapley_entropy(self) -> float:
        """Getting the Shapley entropy score."""
        return self._shapley_entropy
