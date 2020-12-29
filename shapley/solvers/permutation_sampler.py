import random
import numpy as np
from shapley.solution_concept import SolutionConcept

class PermutationSampler(object):

    def __init__(self, permutations: int=1000):
        self.permutations = permutations


    def _setup(self, W):
        self.Phi = np.zeros(W.shape)
        self.indices = [i for i in range(W.shape[1])]


    def _run_permutations(self, W):

        for _ in range(self.permutations):
            random.shuffle(self.indices)
            W_perm = W[:, self.indices]
            cum_sum = np.cumsum(W_perm, axis=1)
            pivotal = np.argmax(cum_sum>q, axis=1)
            Phi[np.arange(W.shape[0]), pivotal] = Phi[np.arange(W.shape[0]), pivotal] + 1
        Phi = Phi/self.permutations
        return Phi

    def solve_games(self, W: np.ndarray, q: float=0.5): -> Phi:
        self._setup(W)
        Phi = self._run_permutations(W)
        return Phi


