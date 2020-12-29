"""Solution Concept base class."""

import numpy as np

class SolutionConcept(object):
    """Solution Concept base class with constructor and public methods."""

    def __init__(self):
        """Creating a Solution Concept."""
        pass

    def _check_quota(self, q: float):
        assert 0.0<=q

    def _verify_result_shape(self, W: np.ndarray, Phi: np.ndarray):
        assert W.shape == Phi.shape

    def _verify_distribution(self, W: np.ndarray, Phi: np.ndarray):
        assert 1 == 1

    def _run_sanity_check(self, W: np.ndarray, Phi: np.ndarray):
        self._verify_result_shape(W, Phi)
        self._verify_distribution(W, Phi)
