"""Expected Marginal Contributions Based Shapley Value Approximation."""

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
    a_s = (q - w) / np.linspace(1, player_count - 1, player_count - 1)
    b_s = (q - 10**-20) / np.linspace(1, player_count - 1, player_count - 1)
    return a_s, b_s


@jit
def create_standard_deviation_vector(var, player_count):
    """
    Creating the vector of standard deviations and integrand limits.
    :param var (float): Variance of the weights in the game.
    :param player_count (int): Number of players in the game.

    :return sigma_s: Vector of standard deviations for the game.
    """
    sigma_s = np.power(var / np.linspace(1, player_count - 1, player_count - 1), 0.5)
    return sigma_s


class ExpectedMarginalContributions(SolutionConcept):
    r"""The expected marginal contributions approximation of the Shapley value in a weighted
    voting game using the technique proposed by Fatima. For details see this paper:
    `"A Linear Approximation Method for the Shapley Value."
     <https://www.sciencedirect.com/science/article/pii/S0004370208000696>`_
    """

    def __init__(self, epsilon: float = 10**-8):
        self.epsilon = epsilon

    def setup(self, W: np.ndarray):
        """Creating an empty Shapley value matrix."""
        self._Phi = np.zeros(W.shape)

    def _approximate(self, W: np.ndarray, q: float):
        """Using the naive multilinear approximation method."""
        for data_point in range(W.shape[0]):
            mu = float(np.mean(W[data_point, :]))
            var = float(np.var(W[data_point, :]))
            sigma_s = create_standard_deviation_vector(var, W.shape[1])
            for player_index in range(W.shape[1]):
                w = W[data_point, player_index]
                a_s, b_s = create_integrand_vectors(W.shape[1], q, w)
                shap_in_game = norm.cdf(b_s, loc=mu, scale=sigma_s) - norm.cdf(
                    a_s, loc=mu, scale=sigma_s
                )
                self._Phi[data_point, player_index] += np.sum(shap_in_game)

        self._Phi = self._Phi / np.sum(self._Phi, axis=1).reshape(-1, 1)

    def solve_game(self, W: np.ndarray, q: float):
        r"""Solving the weigted voting game(s).

        Args:
            W (Numpy array): An :math:`n \times m` matrix of voting weights for the :math:`n` games with :math:`m` players.
            q (float): Quota in the games.
        """
        self._check_quota(q)
        self.setup(W)
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
