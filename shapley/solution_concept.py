
"""Solution Concept base class."""

class SolutionConcept(object):
    """SolutionConcept base class with constructor and public methods."""

    def __init__(self):
        """Creating a Solution Concept."""
        pass

    def _verify_result(self, W, Phi):
        assert W.shape == Phi.shape

