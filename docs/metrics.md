Calculating a statistics vectors for a graph, $g(y)$, is a crucial component of ERGMs. 
Formally, $g$ is a function that receives a graph and returns a vector $g(y) \in \mathbb{R}^q$ with some statistics on that graph. These statistics can be as simple as counting the number of edges in the graph, calculating the number of triangles, or even more complex statistics that depend on exogenous variables for each node.

pyERGM provides a collection of metrics that can be used to calculate these statistics. These metrics are implemented as classes that inherit from the `Metric` class.

## The `Metric` class
Every metric in pyERGM is inherited from the `Metric` class, which defines the interface for calculating statistics on a graph. All metrics implement a `calculate` function, which receives a graph and returns the calculated statistics.

```python
pyERGM.metrics.Metric.calculate(input: np.ndarray | nx.Graph)
```
**Parameters**:

* **input** (*np.ndarray | nx.Graph*) - The input graph for which to calculate the statistics. The graph can be represented as a numpy adjacency matrix or a `networkx` graph object.

**Returns**:

* **result** (*np.ndarray*) - The calculated statistics vector of length $q \geq 1$, depending on how many  statistics the metric returns.

## Examples
### Basic metrics
As our first example, we will calculate two statistics for a directed graph with 4 nodes -

* The number of edges in the graph
* The number of reciprocal edges in the graph (i.e. how many node pairs have reciprocal edges between them).

```python
import numpy as np
from metrics import NumberOfEdgesDirected, TotalReciprocity

# Create a connectivity matrix for a directed graph with 4 nodes
W = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 0]
])


num_edges = NumberOfEdgesDirected()
total_reciprocity = TotalReciprocity()

print(f"Number of edges: {num_edges.calculate(W)}")
print(f"Total reciprocity: {total_reciprocity.calculate(W)}")
```

Output:
```
Number of edges: 6
Total reciprocity: 1
```

As expected, the graph has 6 edges and 1 reciprocal edge between nodes 1 and 2.

### Metrics with multiple statistics
As opposed to the `NumberOfEdgesDirected` metric which returns a scalar, some metrics return a vector of statistics. For example, the `InDegree` and `OutDegree` metrics calculate the indegrees and outdegrees of each node in the graph.

```python
from metrics import InDegree, OutDegree

indeg = InDegree()
outdeg = OutDegree()

print(f"In-degree: {indeg.calculate(W)}")
print(f"Out-degree: {outdeg.calculate(W)}")
```

Output:
```
In-degree: [2 2 1 1]
Out-degree: [1 2 2 1]
```

### Exogenous metrics

## Speeding up metric calculations
