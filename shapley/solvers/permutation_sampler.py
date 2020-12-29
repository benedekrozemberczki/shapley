import random
import numpy as np
from shapley.solution_concept import SolutionConcept

class PermutationSampler(SolutionConcept):
    r"""Permutation sampler to solve a block of weighted voting games.
    For details see: `"" 
    <https://arxiv.org/abs/1709.04875>`_

    Based off the temporal convolution introduced in "Convolutional 
    Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    NB. Given an input sequence of length m and a kernel size of k
    the output sequence will have length m-(k-1)

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """    
    def __init__(self, permutations: int=1000):
        self.permutations = permutations

    def _setup(self, W: np.ndarray):
        """Creating an empty Shapley value matrix and a player pool."""
        self._Phi = np.zeros(W.shape)
        self._indices = [i for i in range(W.shape[1])]

    def _run_permutations(self, W: np.ndarray, q: float):
        """Creating Monte Carlo permutations and finding the marginal voter."""

        for _ in range(self.permutations):
            random.shuffle(self._indices)
            W_perm = W[:, self._indices]
            cum_sum = np.cumsum(W_perm, axis=1)
            pivotal = np.argmax(cum_sum>q, axis=1)
            self._Phi[np.arange(W.shape[0]), pivotal] += 1.0
        self._Phi = self._Phi/self.permutations

    def solve_game(self, W: np.ndarray, q: float) -> np.ndarray:
        r"""Solving the weigted voting game(s).

        Args:
            W (Numpy array): Weights in the games with size :math:`n \times m`.
            q (float): Quota in the games.

        Return Types:
            Phi (Numpy array): Approximate Shapley values of players in game(s).
        """
        self._check_quota(q)
        self._setup(W)
        self._run_permutations(W, q)
        self._run_sanity_check(W, self._Phi)
        return self._Phi
