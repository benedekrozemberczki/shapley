Introduction by example
=======================

*Shapley* is a Python library for evaluating binary classifiers in a machine learning ensemble. The library consists of various methods to compute (approximate) the Shapley value of players (models) in weighted voting games (ensemble games) - a class of transferable utility cooperative games. We covered the exact enumeration based computation and various widely know approximation methods from economics and computer science research papers. There are also functionalities to identify the heterogeneity of the player pool based on the `Shapley entropy <https://arxiv.org/abs/2101.02153>`_. In addition, the framework comes with a `detailed documentation <https://shapley.readthedocs.io/en/latest/>`_, an intuitive `tutorial <https://shapley.readthedocs.io/en/latest/notes/introduction.html>`_, 100% test coverage and illustrative toy `examples <https://github.com/benedekrozemberczki/shapley/tree/master/examples>`_.


--------------------------------------------------------------------------------

**Citing**

If you find *Shapley* useful in your research, please consider citing the following paper:

.. code-block:: latex

    >@misc{rozemberczki2021shapley,
           title = {{The Shapley Value of Classifiers in Ensemble Games}}, 
           author = {Benedek Rozemberczki and Rik Sarkar},
           year = {2021},
           eprint = {2101.02153},
           archivePrefix = {arXiv},
           primaryClass = {cs.LG}
    }

Overview
=======================
--------------------------------------------------------------------------------

We shortly overview the fundamental concepts and features of Shapley through simple examples. These are the following:

.. contents::
    :local:

Standardized dataset ingestion
------------------------------

Shapley assumes that the weights in the game(s) are provided by a ``Numpy float array``. This array is 2 dimensional, the first dimension corresponds to the games and the second one corresponds to the players. The quota for the games is set universally by a ``float`` value. We assume that all of the games can be won by a certain coalition. 

Standardized output generation
------------------------------
The Shapley values are returned as a ``Numpy float array`` which has 2 dimensions - rows represent the games and columns are players. A given value in the ``Numpy array`` is the Shapley value of a player in a game. 

API driven design
-----------------


Exact solution
-----------------

Approximate solution
-----------------



