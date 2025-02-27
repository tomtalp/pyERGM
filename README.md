# pyERGM - A Python implementation for ERGM's

An exponential random graphs model (**ERGM**) is a statistical model that describes a distribution of random graphs. This package provides a simple and easy way to fit and sample from ERGMs.

An ERGM defines a random variable $\mathbf{Y}$, which is simply a random graph on $n$ nodes. The probability of observing a specific graph $y\in \lbrace 0,1 \rbrace ^{n \times n}$ is given by -

$$\Pr(\mathbf{Y}=y | \theta) = \frac{\exp(\theta^Tg(y))}{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))}$$

where $g(y)$ is a vector of statistics that describe the graph $y$, and $\theta \in \mathbb{R}^q$ is a vector of model parameters. Each graph is represented by a binary adjacency matrix, where $y_{ij}=1$ if there is an edge between nodes $i$ and $j$ (and $y_{ji}=1$ in the undirected case).



Fitting a model for even moderately large graphs can be a computationally challenging task. pyERGM keeps this in mind and is implemented to be efficient and scalable by using `numpy` and `Numba`, as well as providing an interface for fitting models on a distributed computing environment.

[View the full documentation here](https://tomtalp.github.io/pyERGM/)


## Installation
https://pypi.org/project/pyERGM/


## Getting started
Fitting an ERGM model requires a graph and a set of statistics that describe the graph. The model is then fit by maximizing the likelihood of the observed graph under the model. 

The following example demonstrates how to fit a simple ERGM model from [Sampson's monastery data](https://networkdata.ics.uci.edu/netdata/html/sampson.html).

```python
from pyERGM import ERGM
from pyERGM.metrics import *
from pyERGM.datasets import load_sampson

sampson_matrix = load_sampson()

num_nodes = sampson_matrix.shape[0]
is_directed = True
metrics = [NumberOfEdgesDirected(), TotalReciprocity()]

model = ERGM(num_nodes, metrics, is_directed=is_directed)
model.fit(sampson_matrix)
```

The above example fits a model from the Sampson's monastery data using the number of edges and total reciprocity as statistics. The graph is represented as an adjacency matrix, but pyERGM also supports graphs represented as `networkx` objects (however there may be performance implications for that).
