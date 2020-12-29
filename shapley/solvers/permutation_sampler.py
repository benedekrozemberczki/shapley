import random
import numpy as np
from shapley.solution_concept import SolutionConcept

class PermutationSampler(SolutionConcept):

    def __init__(self, permutations: int=1000):
        self.permutations = permutations


    def _setup(self, W):
        self._Phi = np.zeros(W.shape)
        self._indices = [i for i in range(W.shape[1])]

    def _run_permutations(self, W, q):

        for _ in range(self.permutations):
            random.shuffle(self._indices)
            W_perm = W[:, self._indices]
            cum_sum = np.cumsum(W_perm, axis=1)
            pivotal = np.argmax(cum_sum>q, axis=1)
            self._Phi[np.arange(W.shape[0]), pivotal] = self._Phi[np.arange(W.shape[0]), pivotal] + 1
        self._Phi = self._Phi/self.permutations

    def solve_game(self, W: np.ndarray, q: float=0.5) -> np.ndarray:
        self._setup(W)
        self._run_permutations(W, q)
        self._run_sanity_check(W, self._Phi)
        return self._Phi


