from abc import ABC, abstractmethod
from typing import Collection

import numpy as np
import networkx as nx

from utils import *


## TODO - Metrics needs to implement a diff() function
class Metric(ABC):
    def __init__(self, metric_name, requires_graph=False):
        self.metric_name = metric_name
        self.requires_graph = requires_graph
        # Each metric either expects directed or undirected graphs. This field should be initialized in the constructor
        # and should not change.
        self._is_directed = None

    @abstractmethod
    def calculate(self, input: np.ndarray | nx.Graph):
        pass

    def get_effective_feature_count(self, n):
        """
        How many features does this metric produce. Defaults to 1
        """
        return 1


class NumberOfEdgesUndirected(Metric):
    def __init__(self):
        super().__init__(metric_name="num_edges_undirected", requires_graph=False)
        self._is_directed = False

    def calculate(self, W: np.ndarray):
        return np.sum(W) // 2


class NumberOfEdgesDirected(Metric):
    def __init__(self):
        super().__init__(metric_name="num_edges_directed", requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return np.sum(W)


class NumberOfTriangles(Metric):
    def __init__(self):
        super().__init__(metric_name="num_triangles", requires_graph=True)
        self._is_directed = False

    def calculate(self, G: nx.Graph):
        if isinstance(G, nx.DiGraph):
            raise ValueError("NumOfTriangles not implemented for directed graphs")
        return sum(
            nx.triangles(G).values()) // 3  # nx.triangles counts each triangle 3 times, for every one of its nodes


class Reciprocity(Metric):
    """
    The Reciprocity metric takes the connectivity matrix of a directed graph, and returns a vector
    of size n-choose-2 indicating whether nodes i,j are connected. i.e. $ y_{i, j} \cdot y_{j, i} $
    for every possible pair of nodes   
    """

    def __init__(self):
        super().__init__(metric_name="reciprocity", requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return (W * W.T)[np.triu_indices(W.shape[0], 1)]

    def get_effective_feature_count(self, n):
        # n choose 2
        return n * (n - 1) // 2


class TotalReciprocity(Metric):
    """
    Calculates how many reciprocal connections exist in a network  
    """

    def __init__(self):
        super().__init__(metric_name="total_reciprocity", requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return (W * W.T).sum() / 2


class MetricsCollection:
    def __init__(self, metrics: Collection[Metric], is_directed: bool):
        self.metrics = tuple(metrics)
        self.requires_graph = any([x.requires_graph for x in self.metrics])
        self.is_directed = is_directed
        for x in self.metrics:
            if x._is_directed != self.is_directed:
                model_is_directed_str = "a directed" if self.is_directed else "an undirected"
                metric_is_directed_str = "a directed" if x._is_directed else "an undirected"
                raise ValueError(f"Trying to initialize {model_is_directed_str} model with {metric_is_directed_str} "
                                 f"metric {x.metric_name}!")
        self.num_of_metrics = len(self.metrics)

    def get_num_of_features(self, n: int):
        """
        Returns the number of features that are being calculated. Since a single metric might
        return more than one feature, the length of the statistics vector might be larger than 
        the amount of metrics. Since it also depends on the network size, n is a mandatory parameters.

        Parameters
        ----------
        n : int
            Number of nodes in network
        
        Returns
        -------
        num_of_features : int
            How many features will be calculated for a network of size n
        """
        num_of_features = sum([metric.get_effective_feature_count(n) for metric in self.metrics])
        return num_of_features

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
        statistics = np.zeros(self.get_num_of_features(n_nodes))

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
