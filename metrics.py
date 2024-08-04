from abc import ABC, abstractmethod
from typing import Collection

import numpy as np
import networkx as nx

from utils import *


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

    def calc_change_score(self, net_1: np.ndarray | nx.Graph, net_2: np.ndarray | nx.Graph, is_turned_on: bool,
                          indices: tuple):
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


# TODO: override the change_score function with a more efficient calculation when possible.
class NumberOfEdgesUndirected(Metric):
    def __init__(self):
        super().__init__(metric_name="num_edges_undirected", requires_graph=False)
        self._is_directed = False

    def calculate(self, W: np.ndarray):
        return np.sum(W) // 2

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, is_turned_on: bool, indices: tuple):
        return 1 if is_turned_on else -1


class NumberOfEdgesDirected(Metric):
    def __init__(self):
        super().__init__(metric_name="num_edges_directed", requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return np.sum(W)

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, is_turned_on: bool, indices: tuple):
        return 1 if is_turned_on else -1


# TODO: change the name of this one to undirected and implement also a directed version?
class NumberOfTriangles(Metric):
    def __init__(self):
        super().__init__(metric_name="num_triangles", requires_graph=False)
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

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, is_turned_on: bool, indices: tuple):
        # The triangles that are affected by the edge toggling are those that involve it, namely, if the (i,j)-th edge
        # is toggled, the change in absolute value equals to the number of nodes k for which the edges (i,k) and (j,k)
        # exist. This is equivalent to the number of 2-length paths from i to j, which is the (i,j)-th entry of W^2.
        # If the edge is turned on, the change is positive, and otherwise negative.
        sign = 1 if is_turned_on else -1
        return sign * ((net_1 @ net_1)[indices[0], indices[1]])


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

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, is_turned_on: bool, indices: tuple):
        # Note: we intentionally initialize the whole matrix and return np.triu_indices() by the end (rather than
        # initializing an array of zeros of size n choose 2) to ensure compliance with the indexing returned by
        # the calculate method.
        all_changes = np.zeros(net_1.shape)
        if net_1[indices[1], indices[0]] and is_turned_on:
            all_changes[indices[0], indices[1]] = 1
        elif net_1[indices[1], indices[0]] and not is_turned_on:
            all_changes[indices[0], indices[1]] = -1
        return all_changes[np.triu_indices(all_changes.shape[0], 1)]


class TotalReciprocity(Metric):
    """
    Calculates how many reciprocal connections exist in a network  
    """

    def __init__(self):
        super().__init__(metric_name="total_reciprocity", requires_graph=False)
        self._is_directed = True

    def calculate(self, W: np.ndarray):
        return (W * W.T).sum() / 2

    def calc_change_score(self, net_1: np.ndarray, net_2: np.ndarray, is_turned_on: bool, indices: tuple):
        if net_1[indices[1], indices[0]] and is_turned_on:
            return 1
        elif net_1[indices[1], indices[0]] and not is_turned_on:
            return -1
        else:
            return 0


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

    def calc_change_scores(self, W1: np.ndarray, W2: np.ndarray, is_turned_on: bool, indices: tuple):
        """
        Calculates the vector of change scores, namely g(net_2) - g(net_1)
        """
        if W1.shape != W2.shape:
            raise ValueError(f"The dimensions of the given networks do not match! {W1.shape}, {W2.shape}")
        if self.requires_graph:
            G1 = connectivity_matrix_to_G(W1, directed=self.is_directed)
            G2 = connectivity_matrix_to_G(W2, directed=self.is_directed)

        n_nodes = W1.shape[0]
        change_scores = np.zeros(self.get_num_of_features(n_nodes))

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:
                inputs = (G1, G2)
            else:
                inputs = (W1, W2)

            n_features_from_metric = metric.get_effective_feature_count(n_nodes)
            change_scores[feature_idx:feature_idx + n_features_from_metric] = metric.calc_change_score(inputs[0],
                                                                                                       inputs[1],
                                                                                                       is_turned_on,
                                                                                                       indices)
            feature_idx += n_features_from_metric

        return change_scores
