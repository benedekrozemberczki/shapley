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

*PyTorch Geometric Temporal* is a temporal (dynamic) extension library.

<p align="justify">The library consists of various dynamic and temporal geometric deep learning, embedding, and spatio-temporal regression methods from a variety of published research papers. In addition, it consists of an easy-to-use dataset loader and iterator for dynamic and temporal graphs, gpu-support. It also comes with a number of benchmark datasets with temporal and dynamic graphs (you can also create your own datasets).</p>

--------------------------------------------------------------------------------

**Citing**


If you find *Shapley* please consider adding the following citation:

```bibtex
@misc{pytorch_geometric_temporal,
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

PyTorch Geometric Temporal makes implementing Dynamic and Temporal Graph Neural Networks quite easy -- see the accompanying [tutorial](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#applications). For example, this is all it takes to implement a recurrent graph convolutional network with two consecutive [graph convolutional GRU](https://arxiv.org/abs/1612.07659) cells and a linear layer:

```python
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class RecurrentGCN(torch.nn.Module):

    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32, 5)
        self.recurrent_2 = GConvGRU(32, 16, 5)
        self.linear = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
```
--------------------------------------------------------------------------------

**Methods Included**

In detail, the following methods can be used.


* **[Voting Game Approximation](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.dcrnn.DCRNN)** from Fatima *et al.*: [A Linear Approximation Method for the Shapley Value](https://www.sciencedirect.com/science/article/pii/S0004370208000696)

* **[Multilinear Extension](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gconv_gru.GConvGRU)** from Owen: [Multilinear Extensions of Games](https://www.jstor.org/stable/2661445?seq=1#metadata_info_tab_contents)

* **[Permutation Sampling](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gconv_lstm.GConvLSTM)** from Seo *et al.*: [Structured Sequence Modeling with Graph  Convolutional Recurrent Networks](https://arxiv.org/abs/1612.07659)

* **[Exact Enumeration](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gc_lstm.GCLSTM)** from Chen *et al.*: [GC-LSTM: Graph Convolution Embedded LSTM for Dynamic Link Prediction](https://arxiv.org/abs/1812.04206)

--------------------------------------------------------------------------------


Head over to our [documentation](https://pytorch-geometric-temporal.readthedocs.io) to find out more about installation, creation of datasets and a full list of implemented methods and available datasets.
For a quick start, check out the [examples](https://github.com/benedekrozemberczki/shapley/tree/master/examples) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](https://benedekrozemberczki/pytorch_geometric_temporal/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/rusty1s/pytorch_geometric/issues).


--------------------------------------------------------------------------------

**Installation**

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

- [MIT License](https://github.com/benedekrozemberczki/karateclub/blob/master/LICENSE)
