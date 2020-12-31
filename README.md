[pypi-image]: https://badge.fury.io/py/shapley.svg
[pypi-url]: https://pypi.python.org/pypi/shapley
[size-image]: https://img.shields.io/github/repo-size/benedekrozemberczki/shapley.svg
[size-url]: https://github.com/benedekrozemberczki/shapley/archive/master.zip
[build-image]: https://github.com/benedekrozemberczki/shapley/workflows/CI/badge.svg
[build-url]: https://github.com/benedekrozemberczki/shapley/actions?query=workflow%3ACI
[docs-image]: https://readthedocs.org/projects/shapley/badge/?version=latest
[docs-url]: https://shapley.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/benedekrozemberczki/shapley/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/benedekrozemberczki/shapley?branch=master

<p align="center">
  <img width="90%" src="https://github.com/benedekrozemberczki/shapley/raw/master/shapley.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]
[![Repo size][size-image]][size-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Build Status][build-image]][build-url]

**[Documentation](https://shapley.readthedocs.io)** | **[External Resources](https://shapley.readthedocs.io/en/latest/notes/resources.html)**

*Shapley* is a Python library for evaluating binary classifiers in a machine learning ensemble.

<p align="justify">The library consists of various methods to compute (approximate) the Shapley value of players (models) in weighted voting games (ensemble games). We covered widely know approximation methods from well known economics and computer science research papers. There are also functionalities to identify the heterogeneity of the player pool. In addition, the framework comes with a detailed documentation, 100% test coverage and illustrative toy examples.</p>

--------------------------------------------------------------------------------

**Citing**


If you find *Shapley* please consider adding the following citation:

```bibtex
@misc{rozemberczki_shapley,
      author = {Benedek, Rozemberczki and Rik, Sarkar},
      title = {{The Shapley Value of Classifiers in Ensemble Games}},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/benedekrozemberczki/shapley}},
}
```

--------------------------------------------------------------------------------

**A simple example**

Shapley makes solving voting games quite easy - see the accompanying [tutorial](https://shapley.readthedocs.io/en/latest/notes/introduction.html#applications). For example, this is all it takes to solve a large weighted voting game defined on the fly:

```python
import numpy as np
from shapley import PermutationSampler

W = np.random.uniform(0, 1, (1, 100))
W = W/W.sum()
q = 0.5

solver = PermutationSampler()
solver.solve_game()
shapley_values = solver.get_solution()
```
--------------------------------------------------------------------------------

**Methods Included**

In detail, the following methods can be used.


* **[Expected Marginal Contribution Approximation](https://shapley.readthedocs.io/en/latest/modules/root.html#shapley.solvers.expected_marginal_contributions.ExpectedMarginalContributions)** from Fatima *et al.*: [A Linear Approximation Method for the Shapley Value](https://www.sciencedirect.com/science/article/pii/S0004370208000696)

* **[Multilinear Extension](https://shapley.readthedocs.io/en/latest/modules/root.html#shapley.solvers.multilinear_extension.MultilinearExtension)** from Owen: [Multilinear Extensions of Games](https://www.jstor.org/stable/2661445?seq=1#metadata_info_tab_contents)

* **[Monte Carlo Permutation Sampling](https://shapley.readthedocs.io/en/latest/modules/root.html#shapley.solvers.permutation_sampler.PermutationSampler)** from Maleki *et al.*: [Bounding the Estimation Error of Sampling-based Shapley Value Approximation](https://arxiv.org/abs/1306.4265)

* **[Exact Enumeration](https://shapley.readthedocs.io/en/latest/modules/root.html#shapley.solvers.exact_enumeration.ExactEnumeration)** from Shapley: [A Value for N-Person Games](https://www.rand.org/pubs/papers/P0295.html)

--------------------------------------------------------------------------------


Head over to our [documentation](https://shapley.readthedocs.io) to find out more about installation, creation of datasets and a full list of implemented methods and available datasets.
For a quick start, check out the [examples](https://github.com/benedekrozemberczki/shapley/tree/master/examples) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](https://benedekrozemberczki/shapley/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/benedekrozemberczki/shapley/issues).


--------------------------------------------------------------------------------

**Installation**

```
$ pip install shapley
```

**Running tests**

```
$ python setup.py test
```
--------------------------------------------------------------------------------

**Running examples**

```
$ cd examples
$ python permutation_example.py
```

--------------------------------------------------------------------------------

**License**

- [MIT License](https://github.com/benedekrozemberczki/shapley/blob/master/LICENSE)
