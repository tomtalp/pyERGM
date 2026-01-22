Calculating a statistics vectors for a graph, $g(y)$, is a crucial component of ERGMs. 
Formally, $g$ is a function that receives a graph and returns a vector $g(y) \in \mathbb{R}^q$ with some statistics on that graph. These statistics can be as simple as counting the number of edges in the graph, calculating the number of triangles, or even more complex statistics that depend on exogenous variables for each node.

pyERGM provides a collection of metrics that can be used to calculate these statistics. These metrics are implemented as classes that inherit from the `Metric` class.

## The `Metric` class
Every metric in pyERGM is inherited from the `Metric` class, which defines the interface for calculating statistics on a graph. All metrics implement a `calculate` function, which receives a graph and returns the calculated statistics.

```python
pyERGM.metrics.Metric.calculate(input: np.ndarray)
```
**Parameters**:

* **input** (*np.ndarray*) - The input graph for which to calculate the statistics, represented as an adjacency matrix.

**Returns**:

* **result** (*np.ndarray*) - The calculated statistics vector of length $q \geq 1$, depending on how many  statistics the metric returns.

### Examples
#### Basic metrics
As our first example, we will calculate two statistics for a directed graph with 4 nodes -

* The number of edges in the graph
* The number of reciprocal edges in the graph (i.e. how many node pairs have reciprocal edges between them).

```python
import numpy as np
from pyERGM.metrics import NumberOfEdgesDirected, TotalReciprocity

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

#### Metrics with multiple statistics
As opposed to the `NumberOfEdgesDirected` metric which returns a scalar, some metrics return a vector of statistics. For example, the `InDegree` and `OutDegree` metrics calculate the indegrees and outdegrees of each node in the graph.

```python
from pyERGM.metrics import InDegree, OutDegree

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

#### Exogenous metrics
So far we've seen metrics that are only based on the graph's connectivity matrix. However, there are many scenarios in which the graph nodes & edges have additional attributes that are external to the connectivity matrix. For example, in a graph that represents a social network, each node might have an attribute representing the age of a person. These are called **exogenous attributes**. 

The exogenous attributes are passed to the metric as a collection of external attributes. The order of these attributes should correspond to the node order in the connectivity matrix.

In the following example, we take a graph with 3 nodes, and assign a number to each node. We then wish to sum these attributes across nodes that are connected to each other. This is done using the `NodeAttrSum` metric.

```python
from pyERGM.metrics import NodeAttrSum

W = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
])

# For each of the 3 nodes in W, we assign a number.
# The order of these attributes corresponds to the node order in the connectivity matrix.
external_attributes = [2, 1, 5]

node_attr_metric = NodeAttrSum(external_attributes, is_directed=True)

print(f"Sum of attributes: {node_attr_metric.calculate(W)}")
```

Output:
```
Sum of attributes: [13]
```

Let's verify this calculation, by observing the attribute sum for every edge - 

* Edge 1 $\rightarrow$ 2 : 2 + 1 = 3
* Edge 2 $\rightarrow$ 1 : 1 + 2 = 3
* Edge 3 $\rightarrow$ 1 : 5 + 2 = 7

which sums up to 13.

## A collection of metrics
It is often the case that an ERGM uses more than a single metric. To facilitate this, pyERGM provides a `MetricsCollection` class that can be used to collect multiple metrics together.

When an ERGM is initialized, it receives a list of metrics. For example - 
    
```python

from pyERGM.metrics import NumberOfEdgesUndirected, NumberOfTriangles
metrics = [NumberOfEdgesUndirected(), NumberOfTriangles()]
model = ERGM(num_nodes, metrics, is_directed=False)
```

The `MetricsCollection` object is initialized behind the scenes by the ERGM constructor, and is used throughout the model's lifecycle to calculate the statistics vector for the graph.


## Collinearity between metrics
Collineraity is a common issue in regression models, where two or more variables have some form of linear relationship. The degree to which the variables are correlated varies from perfect collinearity, where a pair of variables is perfectly correlated, to weaker forms of collinearity, in which the variables are correlated but not perfectly so. Collinearity can lead to issues in the optimization process, as well as in the interpretation of the model's coefficients.

As an example, let's initialize a model with two metrics - one that counts the number of edges, and one that counts the degree of every node in the graph.

1. `NumberOfEdgesUndirected`
2. `UndirectedDegree`


```python

from pyERGM.metrics import NumberOfEdgesUndirected, UndirectedDegree

W = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
])


num_edges = NumberOfEdgesUndirected()
degrees = UndirectedDegree()

print(f"Number of edges: {num_edges.calculate(W)}")
print(f"Degrees: {degrees.calculate(W)}")
```

Output:
```
Number of edges: 3
Degrees: [2, 1, 0]
```

In this example, there exists perfect collinearity between the number of edges and degree profile of all nodes in the graph. By definition, in an undirected graph the sum of degrees equals the number of edges, deeming one of these regressors to be redundant. This means that any regression model that includes all the 4 regressons will have perfect collinearity, which can lead to issues in the optimization process. 

The most straightforward solution is to remove one of the conflicting metrics. pyERGM offers two ways to handle such issues - 

**Automatic removal** - The `MetricsCollection` class provides a tool for automatically detecting and removing collinear metrics. This is the default setting, and can be turned off with the `fix_collinearity=False` flag in the ERGM constructor. (*TODO - elaborate on the algorithm*).

**Manual removal** - As the user building the model, you can manually remove metrics that you know are collinear. This can either be done in any of the following ways - 

1. <ins>Removing a full metric</ins> - Removing the metric from the list of metrics passed to the ERGM constructor is the most straightforward way to fix collinearity, but it might not always be the best solution. In the example above, the `NumberOfEdgesUndirected` metric seems more informative than the `UndirectedDegree` metric, so completely removing it might not be desired.
2. <ins>Ignoring a specific metric statistic</ins> - Metrics with more than a single statistic have an `indices_to_ignore` attribute that can be used to ignore specific statistics within the metric. It is a list of indices that should be ignored when calculating the statistics vector, and is passed when initializing the metric. 
The order of indices within the metric depends on the metric type - 
    * If the metric statistics depend only on the graph's connectivity matrix (e.g. `UndirectedDegree`), the indices correspond to the original ordering of the nodes in the graph. 
    * If the metric statistics depend on exogenous attributes, the indices correspond to the lexicographic ordering of the exogenous attributes passed to the metric, i.e. `sorted(external_attributes)`.
    In a metric such as `NumberOfEdgesTypesDirected` that creates statistics for type pairs, the ordering is the lexicographic ordering of the cartesian product of the node types. For example if `external_attributes=["A", "A", "B"]`, the metric will produce features for type pairs `["AA", "AB", "BA", "BB"]`, in a lexicographical ordering.


Let's see how we could have fixed the collinearity issue in the previous example.

**Automatic fixing** - 
```python
from pyERGM.metrics import NumberOfEdgesUndirected, UndirectedDegree
from pyERGM.metrics import MetricsCollection

W = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
])

num_edges = NumberOfEdgesUndirected()
degrees = UndirectedDegree()

metrics = MetricsCollection(metrics=[num_edges, degrees], is_directed=False, n_nodes=3, fix_collinearity=True)
```

which will output - 
```
Removing the 0 feature of undirected_degree to fix multi-collinearity
```
telling us that the 0-th idx feature of the `UndirectedDegree` metric was removed to fix the collinearity issue. We can also verify this by running - 

```python
print(f"Ignored features - ")
print(metrics.get_ignored_features())

print(f"Remaining features")
print(metrics.get_parameter_names())
```

with outputs -
```
Ignored features -
('undirected_degree_1',)
Remaining features
('num_edges_undirected', 'undirected_degree_2', 'undirected_degree_3')
```

**Ignoring the first statistic in** `UndirectedDegree` -
```python
from pyERGM.metrics import NumberOfEdgesUndirected, UndirectedDegree

W = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
])

num_edges = NumberOfEdgesUndirected()
degrees = UndirectedDegree(indices_to_ignore=[0])

print(f"Number of edges: {num_edges.calculate(W)}")
print(f"Degrees: {degrees.calculate(W)}")
```

Output:
```
Number of edges: 3
Degrees: [1, 0]
```