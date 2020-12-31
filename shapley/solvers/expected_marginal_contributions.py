import random
import numpy as np
from numba import jit
from scipy.stats import norm
from shapley.solution_concept import SolutionConcept


@jit
def create_integrand_vectors(player_count, q, w):
    """
    Creating the vector of standard deviations and integrand limits.
    :param player_count (int): Number of players in the game.
    :param q (float): Quota to win the game.
    :param w (float): Weight of the player in the game.
    
    :return a_s: Vector of lower integrand limits.
    :return b_s: Vector of upper integrand limits.
    """
    a_s = (q-w)/np.linspace(1, player_count-1, player_count-1)
    b_s = (q-10**-20)/np.linspace(1, player_count-1, player_count-1)
    return a_s, b_s


@jit
def create_standard_deviation_vector(var, player_count):
    """
    Creating the vector of standard deviations and integrand limits.
    :param var (float): Variance of the weights in the game.
    :param player_count (int): Number of players in the game.
    
    :return sigma_s: Vector of standard deviations for the game.
    """
    sigma_s = np.power(var/np.linspace(1, player_count-1, player_count-1), 0.5)
    return sigma_s

class ExpectedMarginalContributions(SolutionConcept):
    r"""The multilinear extension approximation of the Shapley value in a weighted
    voting game using the technique proposed by Owen. For details see this paper: 
    `"Multilinear Extensions of Games."  <https://www.jstor.org/stable/2661445#metadata_info_tab_contents>`_
    """

    def __init__(self, epsilon: float=10**-8):
        self.epsilon = epsilon

    def _setup(self, W: np.ndarray):
        """Creating an empty Shapley value matrix."""
        self._Phi = np.zeros(W.shape)

    def _approximate(self, W: np.ndarray, q: float):
        """Using the naive multilinear approximation method."""

        self._Phi = self._Phi / np.sum(self._Phi, axis=1).reshape(-1, 1)

    def solve_game(self, W: np.ndarray, q: float):
        r"""Solving the weigted voting game(s).

        Args:
            W (Numpy array): An :math:`n \times m` matrix of voting weights for the :math:`n` games with :math:`m` players.
            q (float): Quota in the games.
        """
        self._check_quota(q)
        self._setup(W)
        self._approximate(W, q)
        self._run_sanity_check(W, self._Phi)
        self._set_average_shapley()
        self._set_shapley_entropy()

    def get_solution(self) -> np.ndarray:
        r"""Returning the solution.

        Return Types:
            Phi (Numpy array): Approximate Shapley matrix of players in the game(s) with size :math:`n \times m`.
        """
        return self._Phi
