pyERGM offers a simple and straightforward way to create custom metrics. All you need to do is - 


1. Create a new class that inherits from the `Metric` class
2. Implement a `calculate` function that receives a graph and returns the calculated statistics.

For example, let's we create a metric that counts the number of nodes in a directed graph, with in-degree $d \geq \frac{n}{2}$ - 

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

## 
asd