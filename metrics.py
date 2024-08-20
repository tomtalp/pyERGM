from abc import ABC, abstractmethod
from typing import Collection
from copy import deepcopy

import numpy as np
import networkx as nx

from utils import *


class Metric(ABC):
    def __init__(self, requires_graph=False):
        self.requires_graph = requires_graph
        # Each metric either expects directed or undirected graphs. This field should be initialized in the constructor
        # and should not change.
        self._is_directed = None

    @abstractmethod
    def calculate(self, input: np.ndarray | nx.Graph):
        pass

    def get_effective_feature_count(self, n):
        """
        How many features does this metric produce. Defaults to 1.
        """
        return 1

    def calc_change_score(self, net_1: np.ndarray | nx.Graph, net_2: np.ndarray | nx.Graph, indices: tuple):
        """
        The default naive way to calculate the change score (namely, the difference in statistics) of a pair of
        networks.

        Returns
        -------
        statistic of net_2 minus statistic of net_1.
        """
        net_2_stat = self.calculate(net_2)
        net_1_stat = self.calculate(net_1)
        change_score = net_2_stat - net_1_stat
        return change_score


class NumberOfEdgesUndirected(Metric):
    def __str__(self):
        return "num_edges_undirected"
    
    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = False

    def calculate(self, W: np.ndarray):
        return np.sum(W) // 2

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, indices: tuple):
        return 1 if net_2[indices[0], indices[1]] else -1


class NumberOfEdgesDirected(Metric):
    def __str__(self):
        return "num_edges_directed"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return np.sum(W)

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, indices: tuple):
        return 1 if net_2[indices[0], indices[1]] else -1


# TODO: change the name of this one to undirected and implement also a directed version?
class NumberOfTriangles(Metric):
    def __str__(self):
        return "num_triangles"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = False

    def calculate(self, W: np.ndarray):
        if not np.all(W.T == W):
            raise ValueError("NumOfTriangles not implemented for directed graphs")
        # the (i,j)-th entry of W^3 counts the number of 3-length paths from node i to node j. Thus, the i-th element on
        # the diagonal counts the number of triangles that node 1 is part of (3-length paths from i to itself). As the
        # graph is undirected, we get that each path is counted twice ("forward and backwards"), thus the division by 2.
        # Additionally, each triangle is counted 3 times by diagonal elements (once for each node that takes part in
        # forming it), thus the division by 3.
        return (np.linalg.matrix_power(W, 3)).diagonal().sum() // (3 * 2)

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, indices: tuple):
        # The triangles that are affected by the edge toggling are those that involve it, namely, if the (i,j)-th edge
        # is toggled, the change in absolute value equals to the number of nodes k for which the edges (i,k) and (j,k)
        # exist. This is equivalent to the number of 2-length paths from i to j, which is the (i,j)-th entry of W^2.
        # If the edge is turned on, the change is positive, and otherwise negative.
        sign = 1 if net_2[indices[0], indices[1]] else -1
        return sign * np.dot(net_1[indices[0]], net_1[:, indices[1]])

class BaseDegreeVector(Metric):
    """
    A base class for calculating a degree vector for a network.
    To avoid multicollinearity with other features, an optional parameter `base_idx` can be used to specify
    which index the calculation starts from.
    """
    def __init__(self, requires_graph: bool, is_directed: bool, base_idx=0):
        super().__init__(requires_graph=requires_graph)
        self._is_directed = is_directed
        self.base_idx = base_idx

    def get_effective_feature_count(self, n):
        return n - self.base_idx

class InDegree(BaseDegreeVector):
    """
    Calculate the in-degree of each node in a directed graph.
    """
    def __str__(self):
        return "indegree"

    def __init__(self, base_idx=0):
        super().__init__(requires_graph=False, is_directed=True, base_idx=base_idx) 
    
    def calculate(self, W: np.ndarray):
        return W.sum(axis=0)[self.base_idx:]

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, indices: tuple):
        n = net_1.shape[0]
        diff = np.zeros(n)
        i, j = indices

        sign = 1 if net_2[i, j] else -1

        diff[j] = sign
        return diff[self.base_idx:]


class OutDegree(BaseDegreeVector):
    """
    Calculate the out-degree of each node in a directed graph.
    """
    def __str__(self):
        return "outdegree"

    def __init__(self, base_idx=0):
        super().__init__(requires_graph=False, is_directed=True, base_idx=base_idx)  
    
    def calculate(self, W: np.ndarray):
        return W.sum(axis=1)[self.base_idx:]
    
    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, indices: tuple):
        n = net_1.shape[0]
        diff = np.zeros(n)
        i, j = indices

        sign = 1 if net_2[i, j] else -1

        diff[i] = sign
        return diff[self.base_idx:]

class UndirectedDegree(BaseDegreeVector):
    """
    Calculate the degree of each node in an undirected graph.
    """
    def __str__(self):
        return "undirected_degree"

    def __init__(self, base_idx=0):
        super().__init__(requires_graph=False, is_directed=False, base_idx=base_idx)  
    
    def calculate(self, W: np.ndarray):
        return W.sum(axis=0)[self.base_idx:]    

class Reciprocity(Metric):
    """
    The Reciprocity metric takes the connectivity matrix of a directed graph, and returns a vector
    of size n-choose-2 indicating whether nodes i,j are connected. i.e. $ y_{i, j} \cdot y_{j, i} $
    for every possible pair of nodes   
    """
    def __str__(self):
        return "reciprocity"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return (W * W.T)[np.triu_indices(W.shape[0], 1)]

    def get_effective_feature_count(self, n):
        # n choose 2
        return n * (n - 1) // 2

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, indices: tuple):
        # Note: we intentionally initialize the whole matrix and return np.triu_indices() by the end (rather than
        # initializing an array of zeros of size n choose 2) to ensure compliance with the indexing returned by
        # the calculate method.
        all_changes = np.zeros(net_1.shape)
        min_idx = min(indices)
        max_idx = max(indices)
        if net_1[indices[1], indices[0]] and net_2[indices[0], indices[1]]:
            all_changes[min_idx, max_idx] = 1
        elif net_1[indices[1], indices[0]] and not net_2[indices[0], indices[1]]:
            all_changes[min_idx, max_idx] = -1
        return all_changes[np.triu_indices(all_changes.shape[0], 1)]


class TotalReciprocity(Metric):
    """
    Calculates how many reciprocal connections exist in a network  
    """
    def __str__(self):
        return "total_reciprocity"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return (W * W.T).sum() / 2

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, indices: tuple):
        if net_1[indices[1], indices[0]] and net_2[indices[0], indices[1]]:
            return 1
        elif net_1[indices[1], indices[0]] and not net_2[indices[0], indices[1]]:
            return -1
        else:
            return 0

class MetricsCollection:
    def __init__(self, metrics: Collection[Metric], is_directed: bool, n_nodes: int, fix_collinearity=True):
        self.metrics = tuple([deepcopy(metric) for metric in metrics]) 
        self.metric_names = tuple([str(metric) for metric in self.metrics])

        self.requires_graph = any([x.requires_graph for x in self.metrics])
        self.is_directed = is_directed
        for x in self.metrics:
            if x._is_directed != self.is_directed:
                model_is_directed_str = "a directed" if self.is_directed else "an undirected"
                metric_is_directed_str = "a directed" if x._is_directed else "an undirected"
                raise ValueError(f"Trying to initialize {model_is_directed_str} model with {metric_is_directed_str} "
                                 f"metric `{str(x)}`!")
        self.num_of_metrics = len(self.metrics)

        self._fix_collinearity = fix_collinearity
        if self._fix_collinearity:
            self.collinearity_fixer()

        self.n_nodes = n_nodes

        # Returns the number of features that are being calculated. Since a single metric might return more than one feature, the length of the statistics vector might be larger than 
        # the amount of metrics. Since it also depends on the network size, n is a mandatory parameters. That's why we're using the get_effective_feature_count function
        self.num_of_features = sum([metric.get_effective_feature_count(self.n_nodes) for metric in self.metrics])

    def get_metric(self, metric_name: str) -> Metric:
        """
        Get a metric instance
        """
        return self.metrics[self.metric_names.index(metric_name)]


    def collinearity_fixer(self):
        """
        Find collinearity between metrics in the collection.

        Currently this is a naive version that only handles the very simple cases. 
        TODO - Implement a smarter solution
        """
        num_edges_name = str(NumberOfEdgesDirected())
        indegree_name = str(InDegree())
        outdegree_name = str(OutDegree())

        if num_edges_name in self.metric_names and indegree_name in self.metric_names:
            self.get_metric(indegree_name).base_idx = 1
        
        if num_edges_name in self.metric_names and outdegree_name in self.metric_names:
            self.get_metric(outdegree_name).base_idx = 1

        if indegree_name in self.metric_names and outdegree_name in self.metric_names:
            self.get_metric(indegree_name).base_idx = 1

    
    def calculate_statistics(self, W: np.ndarray):
        """
        Calculate the statistics of a graph, formally written as g(y).

        Parameters
        ----------
        W : np.ndarray
            A connectivity matrix.
        
        Returns
        -------
        statistics : np.ndarray
            An array of statistics
        """
        if self.requires_graph:
            G = connectivity_matrix_to_G(W, directed=self.is_directed)

        n_nodes = W.shape[0]
        statistics = np.zeros(self.num_of_features)

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:
                input = G
            else:
                input = W

            n_features_from_metric = metric.get_effective_feature_count(n_nodes)
            statistics[feature_idx:feature_idx + n_features_from_metric] = metric.calculate(input)
            feature_idx += n_features_from_metric

        return statistics

    def calc_change_scores(self, W1: np.ndarray, W2: np.ndarray, indices: tuple):
        """
        Calculates the vector of change scores, namely g(net_2) - g(net_1)
        """
        if W1.shape != W2.shape:
            raise ValueError(f"The dimensions of the given networks do not match! {W1.shape}, {W2.shape}")
        if self.requires_graph:
            G1 = connectivity_matrix_to_G(W1, directed=self.is_directed)
            G2 = connectivity_matrix_to_G(W2, directed=self.is_directed)

        n_nodes = W1.shape[0]
        change_scores = np.zeros(self.num_of_features)

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:
                inputs = (G1, G2)
            else:
                inputs = (W1, W2)

            n_features_from_metric = metric.get_effective_feature_count(n_nodes)
            change_scores[feature_idx:feature_idx + n_features_from_metric] = metric.calc_change_score(inputs[0],
                                                                                                       inputs[1],
                                                                                                       indices)
            feature_idx += n_features_from_metric

        return change_scores
