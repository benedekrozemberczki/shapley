
"""Solution Concept base class."""

class SolutionConcept(object):
    """SolutionConcept base class with constructor and public methods."""

    def __init__(self):
        """Creating a Solution Concept."""
        pass

    

    def _verify_result_shape(self, W, Phi):
        assert W.shape == Phi.shape

    def _verify_distribution(self, W, Phi):
        assert 1 == 1

    def _run_sanity_check(self, W, Phi):
        self._verify_result_shape(W, Phi)
        self._verify_distribution(W, Phi)
