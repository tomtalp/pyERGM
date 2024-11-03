pyERGM offers a simple and straightforward way to create custom metrics. 

Generally speaking, any sort of statistic that can be calculated from a graph can be implemented as a metric.
All you need to do is - 

1. Create a new class that inherits from the `Metric` class
2. Implement a `calculate` function that receives a graph and returns the calculated statistics.

## A first example

Let's begin with a very simple metric, that counts the number of nodes in a directed graph, with in-degree $d \geq \frac{n}{2}$ - 

```python
from pyERGM.metrics import Metric

class NumberOfBigIndegreeNodes(Metric):
    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True
        self._is_dyadic_independent = True

    def calculate(self, input_graph: np.ndarray):
        n = input_graph.shape[0]

        indeg = np.sum(input_graph, axis=0)
        big_indegree_nodes = np.sum(indeg >= n/2)

        return big_indegree_nodes
``` 

which would then be used as follows - 
    
```python
num_big_indegree_nodes = NumberOfBigIndegreeNodes()

W = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0]
])
num_of_big_indegree_nodes = num_big_indegree_nodes.calculate(W)
print(f"# of nodes with in-degree >= n/2: {num_of_big_indegree_nodes}")
```

Output - 
```
# of nodes with in-degree >= n/2: 1
```

<br>
Let's break down the steps in the above example -

1. We began by creating a new class `NumberOfBigIndegreeNodes` that inherits from the `Metric` class.
2. The `__init__()` function deals with 3 important attributes of the metric - 
    * `requires_graph=False` is passed to the parent class, and indicates that the metric does not require a `networkx` input graph to be passed to the `calculate` function. This implies that it expects a numpy adjacency matrix as input. *(this can also be observed by the type hint we wrote in `calculate`)*
    * `self._is_directed = True` indicates that the metric is designed to work with directed graphs.
    * `self._is_dyadic_independent = True` indicates that the metric is dyadic independent, meaning that the existence of an edge `i -> j` does not depend on the presence of any other edge in the graph.  The fact that this is a dyadic independent metric means that `MPLE` can be performed for ERGM's using this metric.
3. The `calculate` function receives the input graph as a numpy adjacency matrix and calculates the number of nodes with in-degree greater than or equal to $\frac{n}{2}$ and returns a single scalar value.

## Supporting multiple statistics
Some metrics return multiple statistics. For example, the `InDegree` metric calculates the indegree of each node in the graph. To support multiple statistics, the `calculate` function should simply return a collection of statistics, instead of a scalar value. This can either be a list or a numpy array. 

Moreover, metrics with multiple statistics should also support the `indices_to_ignore` attribute, which allows pyERGM to exclude certain statistics from the calculation (see the [Introduction to graph metrics](../metrics/) page for more information). Your new metric should allow the user to pass a list of indices to ignore to the metric constructor, and it has to save it as `self._indices_to_ignore` - 

```python
from copy import deepcopy

class MetricWithMultipleStatistics(Metric):
    def __init__(self, indices_to_ignore=None):
        super().__init__(requires_graph=False)
        self._is_directed = True
        self._is_dyadic_independent = True
        
        if indices_to_ignore is None:
            self._indices_to_ignore = []
        else:
            self._indices_to_ignore = deepcopy(indices_to_ignore)
```


Finally, the `calculate` method can use the `_indices_to_ignore` attribute to exclude certain statistics from the calculation - 

```python
def calculate(self, input_graph):
    # Your calculation algorithm here
    return np.delete(statistics, self._indices_to_ignore)
```

## Creating exogenous metrics
Metrics that use exogenous variables are based on the same principles as the previous example. The only difference is that the `__init__` method now receives a collection of exogenous variables that will be used in the `calculate` function. 

As an example, let's create a directed metric that receives an exogenous type for each node, and counts the number of connections between nodes of the same type - 

```python

class NumberOfConnectionsBetweenSameType(Metric):
    def __init__(self, exogenous_types):
        super().__init__(requires_graph=False)
        self._is_directed = True
        self._is_dyadic_independent = True
        self._exogenous_types = exogenous_types

    def calculate(self, input_graph: np.ndarray):
        n = input_graph.shape[0]
        num_connections = 0

        for i in range(n):
            for j in range(n):
                if self._exogenous_types[i] == self._exogenous_types[j] and input_graph[i, j] == 1:
                    num_connections += 1

        return num_connections

```

Now let's verify the behavior of the metric - 

```python

W = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0]
])

types = [1, 1, 2, 1]

num_connections_same_type = NumberOfConnectionsBetweenSameType(types)
num_connections = num_connections_same_type.calculate(W)
print(f"# of connections between nodes of the same type: {num_connections}")
```

Output - 
```
# of connections between nodes of the same type: 4
```

There are 3 nodes of the same type (nodes 1, 2, and 4 are all of type `1`), and there are 4 connections between them - 
```
1 --> 2 , 2 --> 1, 2 --> 4, 4 --> 1
```

## Additional settings and configurations
### Naming a metric
A metric should be named in a way that is descriptive of what it calculates. The metric name is determined by implementing a `__str()__` function in the metric class. For example, the `NumberOfBigIndegreeNodes` metric could be named as follows - 

```python
class NumberOfBigIndegreeNodes(Metric):
    def __str__(self):
        return "num_big_indegree_nodes"
```

### Speeding up metric calculations
TODO

## Testing your metric
It is recommended that you test the behavior of your metric to ensure that it behaves as expected. This can be done by writing simple test functions and running them on a variety of graphs. Some things to consider when testing your metric -

* Does the `calculate` method return the correct results?
* Does the metric correctly implement the `indices_to_ignore` attributes? Namely, does a metric with multiple statistics allow for the exclusion of certain statistics?
* If the metric is dyadic dependent, does it correctly implement the `dyadic_dependent` attribute, forcing the use of `MCMLE`?

