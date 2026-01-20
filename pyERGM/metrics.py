import abc
import enum
from abc import ABC, abstractmethod
from typing import Collection, Callable, Sequence, Any
from copy import deepcopy
import numpy as np
import pandas as pd
import sys
from scipy.spatial.distance import pdist, squareform
from enum import Enum

from pyERGM.logging_config import logger
from pyERGM.utils import *
from pyERGM.cluster_utils import *


class Metric(ABC):
    """
    Abstract base class for all network metrics in the ERGM framework.

    This class defines the interface for computing statistics on networks, including
    methods for calculating metrics, change scores, and handling sampling operations.
    All concrete metric implementations must inherit from this class.

    Parameters
    ----------
    requires_graph : bool, optional
        If True, the metric requires a NetworkX graph object as input.
        If False, the metric can work directly with adjacency matrices. Default is False.
    metric_type : str, optional
        Type of metric: 'node', 'binary_edge', or 'non_binary_edge'. Default is 'binary_edge'.
    metric_node_feature : str, optional
        Name of the node feature this metric operates on. Only relevant if metric_type='node'.
    """
    def __init__(self, requires_graph=False, metric_type='binary_edge', metric_node_feature=None):
        self.requires_graph = requires_graph
        # Each metric either expects directed or undirected graphs. This field should be initialized in the constructor
        # and should not change.
        self._is_directed = None
        self._is_dyadic_independent = True
        self._n_nodes = None
        self._indices_to_ignore = None
        self._metric_type = metric_type  # can have values "node", "binary_edge", "non_binary_edge"
        if self._metric_type not in ['node', 'binary_edge', 'non_binary_edge']:
            raise ValueError(
                f"invalid metric type: {self._metric_type}. Should be one of: 'node', 'binary_edge', 'non_binary_edge'")

        self.metric_node_feature = metric_node_feature  # relevant only if metric_type='node'

        self.does_support_mask = False

    def initialize_indices_to_ignore(self):
        """
        Initialize the array tracking which feature indices should be ignored.

        This method creates a boolean array to track features that should be excluded
        from calculations, typically to avoid multicollinearity issues.
        """
        self._indices_to_ignore = np.array([False] * self._get_total_feature_count())

        if hasattr(self, "_indices_from_user") and self._indices_from_user is not None:
            self.update_indices_to_ignore(self._indices_from_user)

    def _handle_indices_to_ignore(self, res, axis=0):
        if self._indices_to_ignore is None:
            return res

        if axis > 1:
            raise ValueError("Axis should be 0 or 1")
        if axis == 1:
            return res[:, ~self._indices_to_ignore]
        return res[~self._indices_to_ignore]

    @abstractmethod
    def calculate(self, input: np.ndarray | nx.Graph):
        pass

    def _get_effective_feature_count(self):
        """
        How many features does this metric produce. Defaults to 1.
        """
        if self._indices_to_ignore is None:
            return self._get_total_feature_count()
        return np.sum(~self._indices_to_ignore)

    def _get_total_feature_count(self):
        """
        How many features does this metric produce, including the ignored ones. Defaults to 1
        """
        return 1

    def update_indices_to_ignore(self, indices_to_ignore):
        """
        Mark specific feature indices to be ignored in metric calculations.

        Parameters
        ----------
        indices_to_ignore : array-like
            Indices of features that should be excluded from calculations.
        """
        self._indices_to_ignore[indices_to_ignore] = True

    def calc_change_score(self, current_network: np.ndarray | nx.Graph, indices: tuple):
        """
        The default naive way to calculate the change score (namely, the difference in statistics) of a pair of
        networks.

        The newly proposed network is created by flipping the edge denoted by `indices`

        Returns
        -------
        statistic of proposed_network minus statistic of current_network.
        """
        i, j = indices
        if self.requires_graph:
            proposed_network = current_network.copy()
            if proposed_network.has_edge(i, j):
                proposed_network.remove_edge(i, j)
            else:
                proposed_network.add_edge(i, j)
        else:
            proposed_network = current_network.copy()
            proposed_network[i, j] = 1 - proposed_network[i, j]

            if not self._is_directed:
                proposed_network[j, i] = 1 - proposed_network[j, i]

        proposed_network_stat = self.calculate(proposed_network)
        current_network_stat = self.calculate(current_network)
        return proposed_network_stat - current_network_stat

    def calculate_for_sample(self, networks_sample: np.ndarray | Collection[nx.Graph]):
        """
        Calculate metric statistics for a sample of networks.

        Parameters
        ----------
        networks_sample : np.ndarray or Collection[nx.Graph]
            A collection of networks. If array, shape is (n, n, sample_size).

        Returns
        -------
        np.ndarray
            Array of shape (num_features, sample_size) containing statistics for each network.
        """
        num_of_samples = networks_sample.shape[2]

        result = np.zeros((self._get_effective_feature_count(), num_of_samples))
        for i in range(num_of_samples):
            network = networks_sample[i] if self.requires_graph else networks_sample[:, :, i]
            result[:, i] = self.calculate(network)
        return result

    def calculate_mple_regressors(
            self,
            Xs_out: np.ndarray,
            feature_col_indices: npt.NDArray[np.int64],
            edge_indices_mask: npt.NDArray[bool],
            observed_network: np.ndarray | nx.Graph,
    ) -> None:
        """
        Calculate the design matrix (regressors) for Maximum Pseudo-Likelihood Estimation.

        This method populates the design matrix for MPLE optimization by computing
        change scores for each potential edge in the network.

        Parameters
        ----------
        Xs_out : np.ndarray
            Output array to populate with regressor values. Modified in-place.
        feature_col_indices : np.ndarray
            Column indices in Xs_out corresponding to this metric's features.
        edge_indices_mask : np.ndarray
            Boolean mask indicating which edges to include in the calculation.
        observed_network : np.ndarray or nx.Graph
            The observed network data.
        """
        edge_idx_in_full_xs = 0
        edge_idx_in_masked_xs = 0

        if self._is_directed:
            for i in range(self._n_nodes):
                for j in range(self._n_nodes):
                    if i == j:
                        continue
                    if not edge_indices_mask[edge_idx_in_full_xs]:
                        edge_idx_in_full_xs += 1
                        continue
                    indices = (i, j)
                    observed_edge_off = observed_network.copy()
                    if self.requires_graph:
                        if observed_edge_off.has_edge(i, j):
                            observed_edge_off.remove_edge(i, j)
                    else:
                        observed_edge_off[i, j] = 0

                    Xs_out[edge_idx_in_masked_xs, feature_col_indices] = self.calc_change_score(
                        observed_edge_off, indices
                    )
                    edge_idx_in_full_xs += 1
                    edge_idx_in_masked_xs += 1

        else:
            for i in range(self._n_nodes - 1):
                for j in range(i + 1, self._n_nodes):
                    if not edge_indices_mask[edge_idx_in_full_xs]:
                        edge_idx_in_full_xs += 1
                        continue

                    indices = (i, j)

                    observed_edge_off = observed_network.copy()
                    if self.requires_graph:
                        if observed_edge_off.has_edge(i, j):
                            observed_edge_off.remove_edge(i, j)
                    else:
                        observed_edge_off[i, j] = 0
                        observed_edge_off[j, i] = 0

                    Xs_out[edge_idx_in_masked_xs, feature_col_indices] = self.calc_change_score(
                        observed_edge_off, indices
                    )
                    edge_idx_in_full_xs += 1
                    edge_idx_in_masked_xs += 1

    def _get_metric_names(self):
        """
        Get the names of the features that this metric produces. Defaults to the name of the metric if the metric creates a single feature, and multiple
        names if the metric creates multiple features.

        Returns
        -------
        parameter_names: tuple
            A tuple of strings, each string is the name of a parameter that this metric produces.
        """
        total_n_features = self._get_total_feature_count()
        if total_n_features == 1:
            return (str(self),)
        else:
            parameter_names = ()
            for i in range(total_n_features):
                if self._indices_to_ignore is not None and self._indices_to_ignore[i]:
                    continue
                parameter_names += (f"{str(self)}_{i + 1}",)
            return parameter_names

    def _get_ignored_features(self):
        if self._indices_to_ignore is None or not np.any(self._indices_to_ignore):
            return tuple()

        ignored_features = ()
        for i in range(self._get_total_feature_count()):
            if self._indices_to_ignore is not None and self._indices_to_ignore[i]:
                ignored_features += (f"{str(self)}_{i + 1}",)

        return ignored_features

    def _calc_bootstrapped_scalar_feature(self, first_halves_to_use: np.ndarray,
                                          first_halves_indices: np.ndarray[int], second_halves_indices: np.ndarray[int],
                                          max_feature_val_calculator: Callable):
        num_nodes_in_observed = first_halves_indices.shape[0] + second_halves_indices.shape[0]
        num_nodes_in_first_half = first_halves_indices.shape[0]
        max_feature_in_observed = max_feature_val_calculator(num_nodes_in_observed)
        max_feature_in_first_half = max_feature_val_calculator(num_nodes_in_first_half)
        return self.calculate_for_sample(
            first_halves_to_use) * max_feature_in_observed / max_feature_in_first_half


class NumberOfEdges(Metric):
    """
    Abstract base class for counting edges in a network.

    This metric counts the total number of edges present in a network.
    Use NumberOfEdgesDirected or NumberOfEdgesUndirected for concrete implementations.
    """
    def __str__(self):
        raise NotImplementedError

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_dyadic_independent = True
        self.does_support_mask = True

    @staticmethod
    def _get_num_edges_in_mat_factor() -> int:
        """
        A getter to normalize calculations for directed/undirected graphs in children classes
        """
        raise NotImplementedError(
            "This class is abstract by nature, please use either NumberOfEdgesUndirected or NumberOfEdgesDirected"
        )

    def calculate(self, W: np.ndarray, mask: npt.NDArray[bool] | None = None) -> float:
        mat_sum = np.sum(W) if mask is None else np.sum(W * mask)
        return mat_sum // self._get_num_edges_in_mat_factor()

    @staticmethod
    @njit
    def calc_change_score(current_network: np.ndarray, indices: tuple):
        return -1 if current_network[indices[0], indices[1]] else 1

    def calculate_for_sample(
            self,
            networks_sample: np.ndarray | torch.Tensor,
            mask: npt.NDArray[bool] | None = None
    ) -> np.ndarray:
        """
        Sum each matrix over all matrices in sample
        """
        mat_sum = (
            np.sum(networks_sample, axis=(0, 1)) if mask is None else
            np.sum(
                networks_sample[np.where(reshape_flattened_off_diagonal_elements_to_square(mask, self._is_directed))],
                axis=0,
            )
        )
        return mat_sum // self._get_num_edges_in_mat_factor()

    def calculate_mple_regressors(
            self,
            Xs_out: np.ndarray,
            feature_col_indices: npt.NDArray[np.int64],
            edge_indices_mask: npt.NDArray[bool],
            observed_network=None,
    ):
        Xs_out[:, feature_col_indices] = 1

    def calculate_bootstrapped_features(self, first_halves_to_use: np.ndarray,
                                        second_halves_to_use: np.ndarray,
                                        first_halves_indices: np.ndarray[int], second_halves_indices: np.ndarray[int]):
        """
        Calculates the bootstrapped number of edges, by counting edges in the sampled subnetworks, and normalizing by
        network size (i.e., calculating the fraction of existing edges out of all possible ones in sampled subnetworks,
        and multiplying by the number of possible edges in the full observed network).
        Parameters
        ----------
        first_halves_to_use
            Multiple samples of subnetworks of an observed network, representing the connectivity between half of the
            nodes in the large network.
        second_halves_to_use
            The subnetworks formed by the complementary set of nodes of the large network for each sample.
        first_halves_indices
            The indices of the nodes in the first half of the large network for each sample, according to the ordering
            of the nodes in the large network.
        second_halves_indices
            The indices of the nodes in the second half of the large network for each sample, according to the ordering

        Returns
        -------
        Properly normalized statistics of subnetworks of an observed network.
        """
        return self._calc_bootstrapped_scalar_feature(first_halves_to_use, first_halves_indices, second_halves_indices,
                                                      lambda n: n * (n - 1) / self._get_num_edges_in_mat_factor())


class NumberOfEdgesUndirected(NumberOfEdges):
    """
    Metric for counting edges in an undirected network.

    In an undirected network, each edge is counted once (even though it appears
    twice in the adjacency matrix due to symmetry).
    """
    def __str__(self):
        return "num_edges_undirected"

    def __init__(self):
        super().__init__()
        self._is_directed = False

    @staticmethod
    def _get_num_edges_in_mat_factor() -> int:
        return 2


class NumberOfEdgesDirected(NumberOfEdges):
    """
    Metric for counting edges in a directed network.

    In a directed network, each directed edge (i -> j) is counted separately
    from its reverse (j -> i).
    """
    def __str__(self):
        return "num_edges_directed"

    def __init__(self):
        super().__init__()
        self._is_directed = True

    @staticmethod
    def _get_num_edges_in_mat_factor() -> int:
        return 1


# TODO: change the name of this one to undirected and implement also a directed version?
class NumberOfTriangles(Metric):
    """
    Metric for counting triangles in an undirected network.

    A triangle is a set of three nodes where each pair is connected by an edge.
    This is a measure of network clustering and transitivity.

    Notes
    -----
    Currently only implemented for undirected networks. The count is computed
    using matrix multiplication: tr(W^3) / 6, where W is the adjacency matrix.
    """
    def __str__(self):
        return "num_triangles"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = False
        self._is_dyadic_independent = False

    def calculate(self, W: np.ndarray):
        if not np.all(W.T == W):
            raise ValueError("NumOfTriangles not implemented for directed graphs")
        # the (i,j)-th entry of W^3 counts the number of 3-length paths from node i to node j. Thus, the i-th element on
        # the diagonal counts the number of triangles that node 1 is part of (3-length paths from i to itself). As the
        # graph is undirected, we get that each path is counted twice ("forward and backwards"), thus the division by 2.
        # Additionally, each triangle is counted 3 times by diagonal elements (once for each node that takes part in
        # forming it), thus the division by 3.
        return (np.linalg.matrix_power(W, 3)).diagonal().sum() // (3 * 2)

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        # The triangles that are affected by the edge toggling are those that involve it, namely, if the (i,j)-th edge
        # is toggled, the change in absolute value equals to the number of nodes k for which the edges (i,k) and (j,k)
        # exist. This is equivalent to the number of 2-length paths from i to j, which is the (i,j)-th entry of W^2.
        # If the edge is turned on, the change is positive, and otherwise negative.

        sign = -1 if current_network[indices[0], indices[1]] else 1
        return sign * np.dot(current_network[indices[0]], current_network[:, indices[1]])

    def calculate_for_sample(self, networks_sample: np.ndarray):
        return np.einsum('ijk,jlk,lik->k', networks_sample, networks_sample, networks_sample) // (3 * 2)


class BaseDegreeVector(Metric, abc.ABC):
    """
    Abstract base class for calculating degree vectors in networks.

    This class provides the foundation for degree-based metrics, which count
    the number of connections for each node. The specific degree type (in-degree,
    out-degree, or undirected degree) is determined by the summation axis.

    To avoid multicollinearity with other features, an optional parameter
    `indices_from_user` can be used to specify which node indices should be
    ignored in the calculation.

    Parameters
    ----------
    is_directed : bool
        Whether the network is directed.
    summation_axis : SummationAxis
        Axis along which to sum the adjacency matrix (ROWS for in-degree, COLUMNS for out-degree).
    indices_from_user : Sequence[int], optional
        Indices of nodes whose degrees should be excluded from the feature vector.
    """

    class SummationAxis(Enum):
        ROWS = 0
        COLUMNS = 1

    def __init__(
            self,
            is_directed: bool,
            summation_axis: SummationAxis,
            indices_from_user: Sequence[int] | None = None,
    ):
        super().__init__(requires_graph=False)

        self._indices_from_user = np.array(indices_from_user, dtype=int).copy() if indices_from_user is not None else None
        self._is_directed = is_directed
        self._is_dyadic_independent = True
        self.does_support_mask = True

        self._summation_axis = summation_axis

    @abc.abstractmethod
    def _get_change_score_indices_from_summation_axis(
            self,
            edge_indices: tuple[int, int],
    ) -> tuple[int, ...]:
        raise NotImplementedError(
            "This class is abstract by nature, please use one of InDegree, OutDegree, UndirectedDegree"
        )

    def _get_total_feature_count(self):
        return self._n_nodes

    def calculate(self, W: np.ndarray):
        return self.calculate_for_sample(W)

    def calc_change_score(self, current_network: np.ndarray, indices: tuple[int, int]):
        n = current_network.shape[0]
        diff = np.zeros(n)
        i, j = indices

        sign = -1 if current_network[i, j] else 1

        for changed_idx in self._get_change_score_indices_from_summation_axis(indices):
            diff[changed_idx] = sign
        return self._handle_indices_to_ignore(diff)

    def calculate_for_sample(self, networks_sample: np.ndarray, mask: npt.NDArray[bool] | None = None):
        if mask is None:
            all_degrees = np.sum(networks_sample, axis=self._summation_axis.value)
        else:
            square_mask = reshape_flattened_off_diagonal_elements_to_square(mask, self._is_directed).astype(float)
            einsum_result = 'jk' if self._summation_axis == BaseDegreeVector.SummationAxis.ROWS else 'ik'
            all_degrees = np.einsum(f'ijk,ij->{einsum_result}', networks_sample, square_mask)
        return self._handle_indices_to_ignore(all_degrees)

    def calculate_bootstrapped_features(self, first_halves_to_use: np.ndarray,
                                        second_halves_to_use: np.ndarray,
                                        first_halves_indices: np.ndarray[int], second_halves_indices: np.ndarray[int]):
        """
        Calculates the bootstrapped degree, by counting the connections of each node in the sampled subnetworks, and
        normalizing by network size (i.e., calculating the fraction of existing edges out of all possible ones for each
        node in sampled subnetworks, and multiplying by the number of possible edges of a single node in the full
        observed network).
        Each node appears in one of the sub-samples, so both are used, and the indices are used to order the calculated
        values in the entire features vector.
        Parameters
        ----------
        first_halves_to_use
            Multiple samples of subnetworks of an observed network, representing the connectivity between half of the
            nodes in the large network.
        second_halves_to_use
            The subnetworks formed by the complementary set of nodes of the large network for each sample.
        first_halves_indices
            The indices of the nodes in the first half of the large network for each sample, according to the ordering
            of the nodes in the large network.
        second_halves_indices
            The indices of the nodes in the second half of the large network for each sample, according to the ordering

        Returns
        -------
        Properly normalized statistics of subnetworks of an observed network.
        """
        num_nodes_in_observed = first_halves_indices.shape[0] + second_halves_indices.shape[0]
        num_nodes_in_first_half = first_halves_indices.shape[0]
        num_nodes_in_second_half = second_halves_indices.shape[0]

        # NOTE! we want the estimated covariance matrix from the bootstrap to match the dimension of samples from the
        # model during optimization, so we must first calculate the degrees for all the nodes (thus the initialization
        # of a new instance, which makes sure that no indices are ignored), and then ignore the indices that will be
        # removed by the metric.
        fresh_instance = self.__class__()  # No child class has a `is_directed` input to the constructor.
        degrees_first_halves = fresh_instance.calculate_for_sample(first_halves_to_use) * (
                num_nodes_in_observed - 1) / (num_nodes_in_first_half - 1)
        fresh_instance._n_nodes = num_nodes_in_second_half
        degrees_second_halves = fresh_instance.calculate_for_sample(second_halves_to_use) * (
                num_nodes_in_observed - 1) / (num_nodes_in_second_half - 1)
        num_sub_samples = first_halves_to_use.shape[2]
        bootstrapped_degrees = np.zeros((self._n_nodes, num_sub_samples))
        bootstrapped_degrees[first_halves_indices, np.arange(num_sub_samples)] = degrees_first_halves
        bootstrapped_degrees[second_halves_indices, np.arange(num_sub_samples)] = degrees_second_halves
        return self._handle_indices_to_ignore(bootstrapped_degrees)


class InDegree(BaseDegreeVector):
    """
    Calculate the in-degree of each node in a directed graph.

    In-degree is the number of incoming edges to a node. This metric produces
    a feature vector of length n (number of nodes), where each element represents
    the in-degree of the corresponding node.

    Parameters
    ----------
    indices_from_user : array-like, optional
        Indices of nodes whose in-degrees should be excluded from the feature vector
        to avoid multicollinearity.
    """

    def __str__(self):
        return "indegree"

    def __init__(self, indices_from_user=None):
        super().__init__(
            is_directed=True,
            summation_axis=BaseDegreeVector.SummationAxis.ROWS,
            indices_from_user=indices_from_user,
        )

    def _get_change_score_indices_from_summation_axis(
            self,
            edge_indices: tuple[int, int],
    ) -> tuple[int, ...]:
        # In degree - summing over rows, the statistic of the second node (target in edge) changes.
        return (edge_indices[1],)

    def calculate_mple_regressors(
            self,
            Xs_out: np.ndarray,
            feature_col_indices: npt.NDArray[np.int64],
            edge_indices_mask: npt.NDArray[bool],
            observed_network: np.ndarray | nx.Graph,
    ) -> None:
        num_neurons = num_edges_to_num_nodes(edge_indices_mask.size, is_directed=True)
        in_deg_xs = np.tile(np.eye(num_neurons), (num_neurons, 1))[~np.eye(num_neurons, dtype=bool).flatten()]
        in_deg_xs = self._handle_indices_to_ignore(in_deg_xs, axis=1)
        # Looping rather than setting with the full mask directly is negligible for small arrays but much faster for
        # large arrays due too under-the-hood copies of large arrays.
        for masked_idx, non_masked_idx in enumerate(np.where(edge_indices_mask)[0]):
            Xs_out[masked_idx, feature_col_indices] = in_deg_xs[non_masked_idx]


class OutDegree(BaseDegreeVector):
    """
    Calculate the out-degree of each node in a directed graph.

    Out-degree is the number of outgoing edges from a node. This metric produces
    a feature vector of length n (number of nodes), where each element represents
    the out-degree of the corresponding node.

    Parameters
    ----------
    indices_from_user : array-like, optional
        Indices of nodes whose out-degrees should be excluded from the feature vector
        to avoid multicollinearity.
    """

    def __str__(self):
        return "outdegree"

    def __init__(self, indices_from_user=None):
        super().__init__(
            is_directed=True,
            summation_axis=BaseDegreeVector.SummationAxis.COLUMNS,
            indices_from_user=indices_from_user,
        )

    def _get_change_score_indices_from_summation_axis(
            self,
            edge_indices: tuple[int, int],
    ) -> tuple[int, ...]:
        # Out degree - summing over columns, the statistic of the first node (source in edge) changes.
        return (edge_indices[0],)

    def calculate_mple_regressors(
            self,
            Xs_out: np.ndarray,
            feature_col_indices: npt.NDArray[np.int64],
            edge_indices_mask: npt.NDArray[bool],
            observed_network: np.ndarray | nx.Graph,
    ) -> None:
        num_neurons = num_edges_to_num_nodes(edge_indices_mask.size, is_directed=True)
        out_deg_xs = np.repeat(np.eye(num_neurons), num_neurons-1, axis=0)
        out_deg_xs = self._handle_indices_to_ignore(out_deg_xs, axis=1)
        # See comment in InDegree.calculate_mple_regressors on loop performance.
        for masked_idx, non_masked_idx in enumerate(np.where(edge_indices_mask)[0]):
            Xs_out[masked_idx, feature_col_indices] = out_deg_xs[non_masked_idx]

class UndirectedDegree(BaseDegreeVector):
    """
    Calculate the degree of each node in an undirected graph.

    Degree is the number of edges connected to a node. This metric produces
    a feature vector of length n (number of nodes), where each element represents
    the degree of the corresponding node.

    Parameters
    ----------
    indices_from_user : array-like, optional
        Indices of nodes whose degrees should be excluded from the feature vector
        to avoid multicollinearity.
    """

    def __str__(self):
        return "undirected_degree"

    def __init__(self, indices_from_user=None):
        super().__init__(
            is_directed=False,
            summation_axis=BaseDegreeVector.SummationAxis.ROWS, # it doesn't matter over which axis to sum
            indices_from_user=indices_from_user,
        )

    def _get_change_score_indices_from_summation_axis(
            self,
            edge_indices: tuple[int, int],
    ) -> tuple[int, ...]:
        # Undirected degree - the statistic of both nodes changes.
        return edge_indices

    def calculate_mple_regressors(
            self,
            Xs_out: np.ndarray,
            feature_col_indices: npt.NDArray[np.int64],
            edge_indices_mask: npt.NDArray[bool],
            observed_network: np.ndarray | nx.Graph,
    ) -> None:
        num_node_pairs = edge_indices_mask.size
        num_nodes = num_edges_to_num_nodes(num_node_pairs, is_directed=False)
        deg_xs = np.zeros((num_node_pairs, num_nodes))
        row_selector = np.arange(num_node_pairs, dtype=int)
        i_indices, j_indices = np.triu_indices(num_nodes, k=1)
        deg_xs[row_selector, i_indices] = 1
        deg_xs[row_selector, j_indices] = 1
        deg_xs = self._handle_indices_to_ignore(deg_xs, axis=1)
        # See comment in InDegree.calculate_mple_regressors on loop performance.
        for masked_idx, non_masked_idx in enumerate(np.where(edge_indices_mask)[0]):
            Xs_out[masked_idx, feature_col_indices] = deg_xs[non_masked_idx]


class Reciprocity(Metric):
    """
    Calculate reciprocity indicators for all node pairs in a directed graph.

    This metric produces a feature vector of size n-choose-2, where each element
    indicates whether a pair of nodes (i, j) has reciprocal connections, i.e.,
    both i -> j and j -> i edges exist. Formally: y_{i,j} * y_{j,i} for all pairs.

    Returns
    -------
    The metric returns a vector where 1 indicates reciprocal connection and 0 otherwise.
    """

    def __str__(self):
        return "reciprocity"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True
        self._is_dyadic_independent = False

    def calculate(self, W: np.ndarray):
        return (W * W.T)[np.triu_indices(W.shape[0], 1)]

    def _get_total_feature_count(self):
        # n choose 2
        return self._n_nodes * (self._n_nodes - 1) // 2

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        # Note: we intentionally initialize the whole matrix and return np.triu_indices() by the end (rather than
        # initializing an array of zeros of size n choose 2) to ensure compliance with the indexing returned by
        # the calculate method.
        i, j = indices
        all_changes = np.zeros(current_network.shape)
        min_idx = min(indices)
        max_idx = max(indices)

        if current_network[j, i] and not current_network[i, j]:
            all_changes[min_idx, max_idx] = 1
        elif current_network[j, i] and current_network[i, j]:
            all_changes[min_idx, max_idx] = -1
        return all_changes[np.triu_indices(all_changes.shape[0], 1)]


class TotalReciprocity(Metric):
    """
    Calculate the total number of reciprocal connections in a directed network.

    This metric counts the number of node pairs (i, j) where both i -> j and j -> i
    edges exist. Unlike the Reciprocity metric which returns a vector for each pair,
    this returns a single scalar value representing the total count.

    Returns
    -------
    float
        The total number of reciprocal dyads in the network.
    """

    def __str__(self):
        return "total_reciprocity"

    def __init__(self):
        super().__init__(requires_graph=False)
        self._is_directed = True
        self._is_dyadic_independent = False

    def calculate(self, W: np.ndarray):
        return (W * W.T).sum() / 2

    @staticmethod
    @njit
    def calc_change_score(current_network: np.ndarray, indices: tuple):
        i, j = indices

        if current_network[j, i] and not current_network[i, j]:
            return 1
        elif current_network[j, i] and current_network[i, j]:
            return -1
        else:
            return 0

    @staticmethod
    # @njit # Not supporting neither np.einsum nor sparse torch
    def calculate_for_sample(networks_sample: np.ndarray | torch.Tensor):
        if isinstance(networks_sample, torch.Tensor) and networks_sample.is_sparse:
            transposed_sparse_tensor = transpose_sparse_sample_matrices(networks_sample)
            return torch.sum(networks_sample * transposed_sparse_tensor, axis=(0, 1)) / 2
        elif isinstance(networks_sample, np.ndarray):
            return np.einsum("ijk,jik->k", networks_sample, networks_sample) / 2
        else:
            raise ValueError(f"Unsupported type of sample: {type(networks_sample)}! Supported types are np.ndarray and "
                             f"torch.Tensor with is_sparse=True")

    def calculate_bootstrapped_features(self, first_halves_to_use: np.ndarray,
                                        second_halves_to_use: np.ndarray,
                                        first_halves_indices: np.ndarray[int], second_halves_indices: np.ndarray[int]):
        """
        Calculates the bootstrapped number of reciprocal dyads, by counting such pairs in the sampled subnetworks, and
        normalizing by network size (i.e., calculating the fraction of existing reciprocal dyads out of all possible
        ones in sampled subnetworks, and multiplying by the number of possible reciprocal dyads in the full observed
        network).
        Parameters
        ----------
        first_halves_to_use
            Multiple samples of subnetworks of an observed network, representing the connectivity between half of the
            nodes in the large network.
        second_halves_to_use
            The subnetworks formed by the complementary set of nodes of the large network for each sample.
        first_halves_indices
            The indices of the nodes in the first half of the large network for each sample, according to the ordering
            of the nodes in the large network.
        second_halves_indices
            The indices of the nodes in the second half of the large network for each sample, according to the ordering

        Returns
        -------
        Properly normalized statistics of subnetworks of an observed network.
        """
        return self._calc_bootstrapped_scalar_feature(first_halves_to_use, first_halves_indices, second_halves_indices,
                                                      lambda n: n * (n - 1) / 2)


class ExWeightNumEdges(Metric):
    """
    Abstract base class for edge metrics weighted by exogenous node attributes.

    This class provides functionality for calculating weighted sums of edges,
    where edge weights are derived from exogenous attributes of the nodes.
    Concrete implementations must define how edge weights are calculated.

    Parameters
    ----------
    exogenous_attr : Collection
        Exogenous attributes for each node in the network.
    """

    # TODO: Collection doesn't necessarily support __getitem__, find a typing hint of a sized Iterable that does.
    def __init__(self, exogenous_attr: Collection):
        super().__init__(requires_graph=False)
        self.exogenous_attr = exogenous_attr
        self.edge_weights = None
        self._calc_edge_weights()
        self.does_support_mask = True

    @abstractmethod
    def _calc_edge_weights(self):
        ...

    @abstractmethod
    def _get_num_weight_mats(self):
        ...

    def _get_total_feature_count(self):
        return self._get_num_weight_mats()

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        sign = -1 if current_network[indices[0], indices[1]] else 1
        return sign * self._handle_indices_to_ignore(self.edge_weights)[:, indices[0], indices[1]]

    def calculate(self, input: np.ndarray, mask: npt.NDArray[bool] | None = None):
        return self.calculate_for_sample(expand_net_dims(input), mask=mask)[..., -1]

    def calculate_for_sample(self, networks_sample: np.ndarray, mask: npt.NDArray[bool] | None = None):
        if mask is not None:
            mask = reshape_flattened_off_diagonal_elements_to_square(mask, self._is_directed)
        res = self._numba_calculate_for_sample(networks_sample, self.edge_weights, mask)
        if not self._is_directed:
            res = res / 2
        return self._handle_indices_to_ignore(res)

    @staticmethod
    @njit
    def _numba_calculate_for_sample(
            networks_sample: np.ndarray,
            edge_weights: np.ndarray,
            mask: npt.NDArray[bool] | None = None,
    ):
        # transform to shape m X n(n-1) where m is the number of weight matrices (flatten each weight matrix)
        reshaped_edge_weights = edge_weights.reshape(edge_weights.shape[0], -1).astype(np.float64)

        masked_reshaped_edge_weights = reshaped_edge_weights.copy()
        if mask is not None:
            # This method expects squared masks, to match the dimensionality of edge_weights. Ignoring the elements on
            # the diagonal in calculations is handled by having zero edge weights for those elements.
            masked_reshaped_edge_weights *= mask.flatten()

        # transform to shape n(n-1) X k where k is the sample size (flatten each network in the sample)
        reshaped_networks_sample = networks_sample.reshape(-1, networks_sample.shape[-1]).astype(np.float64)

        # Calculate an m X k matrix where each column is the statistics (m node attributed sum) and there is a column
        # per network in the sample.
        res = masked_reshaped_edge_weights @ reshaped_networks_sample

        return res

    def calculate_mple_regressors(
            self,
            Xs_out: np.ndarray,
            feature_col_indices: npt.NDArray[np.int64],
            edge_indices_mask: npt.NDArray[bool],
            observed_network=None,
    ):
        # edge_weights shape is (num_weight_mats, n_nodes, n_nodes), the desired outcome is in the shape:
        # (n_nodes**2 - n_nodes, num_weight_mats)
        num_nodes = len(self.exogenous_attr)
        if self._is_directed:
            ex_weight_num_edges_xs = self.edge_weights[:, np.eye(
                num_nodes) == 0].transpose()
        else:
            up_triangle_indices = np.triu_indices(num_nodes, k=1)
            ex_weight_num_edges_xs = self.edge_weights[:, up_triangle_indices[0],
                                             up_triangle_indices[1]].transpose()
        # See comment in InDegree.calculate_mple_regressors on loop performance.
        for masked_idx, non_masked_idx in enumerate(np.where(edge_indices_mask)[0]):
            Xs_out[masked_idx, feature_col_indices] = ex_weight_num_edges_xs[non_masked_idx]


class NumberOfEdgesTypes(Metric):
    """
    An abstract metric to count how many edges exist between different node types in a graph (directed or undirected).
    """

    not_implemented_error_message = (
            "This class is abstract by nature, please use either " +
            "NumberOfEdgesTypesUndirected or NumberOfEdgesTypesDirected"
    )

    def __str__(self):
        raise NotImplementedError(NumberOfEdgesTypes.not_implemented_error_message)

    @staticmethod
    def _get_num_edges_in_mat_factor() -> int:
        """
        A getter to normalize calculations for directed/undirected graphs in children classes
        """
        raise NotImplementedError(NumberOfEdgesTypes.not_implemented_error_message)

    def _calc_type_pairs_indices(self):
        """
        Calculate a mapping between node type pairs (edge pairs) and their indices in the sorted type paris list.
        """
        self._sorted_type_pairs_indices = None
        raise NotImplementedError(NumberOfEdgesTypes.not_implemented_error_message)

    def _get_total_feature_count(self):
        raise NotImplementedError(NumberOfEdgesTypes.not_implemented_error_message)

    def _get_flattened_edge_type_idx_assignment(self):
        """
        Get the flattened version of the matrix that stores the edge type (node type pair) index for every entry in the
        matrix.
        """
        raise NotImplementedError(NumberOfEdgesTypes.not_implemented_error_message)

    def _get_sorted_canonical_type_pairs(self):
        """
        Get the list of type pairs representative (the representative depends on whether ordering is improtant, i.e.,
        directed or not) sorted according to their indexing in the metric statistics vector.
        """
        raise NotImplementedError(NumberOfEdgesTypes.not_implemented_error_message)

    def __init__(self, exogenous_attr: Sequence[Any], indices_from_user=None):
        self.exogenous_attr = exogenous_attr

        self.unique_types = sorted(list(set(self.exogenous_attr)))

        self._indices_from_user = indices_from_user.copy() if indices_from_user is not None else None
        super().__init__()

        self.does_support_mask = True

        self._effective_feature_count = self._get_effective_feature_count()
        self._indices_to_ignore_up_to_idx = np.cumsum(self._indices_to_ignore)

        self.indices_of_types = {}
        for i, t in enumerate(self.exogenous_attr):
            if t not in self.indices_of_types.keys():
                self.indices_of_types[t] = [i]
            else:
                self.indices_of_types[t].append(i)

        self.sorted_type_pairs = get_sorted_type_pairs(self.unique_types)
        self._calc_type_pairs_indices()

        self._calc_edge_type_idx_assignment()

    def _calc_edge_type_idx_assignment(self):
        """
        Each edge is assigned with the index of its corresponding sorted type pair.

        For example, for an n=4 matrix with types [A, B, A, B], the weight matrix is
            [ AA, AB, AA, AB
              BA, BB, BA, BB
              AA, AB, AA, AB
              BA, BB, BA, BB ]
        but instead of AA, AB, ..., we save the corresponding index of the type pair
        in self._sorted_type_pairs (which is sorted alphabetically.)

        """
        num_nodes = len(self.exogenous_attr)
        self._edge_type_idx_assignment = np.zeros((num_nodes, num_nodes)).astype(int)

        for i in range(num_nodes):
            for j in range(num_nodes):
                type_1 = self.exogenous_attr[i]
                type_2 = self.exogenous_attr[j]

                self._edge_type_idx_assignment[i, j] = self._sorted_type_pairs_indices[(type_1, type_2)]

        self._edge_type_idx_assignment += 1  # Increment by 1 to avoid 0-indexing (the index 0 will be kept for non-existing edges)

    def calculate_for_sample(self, networks_sample: np.ndarray, mask: npt.NDArray[bool] | None = None):
        sample_size = networks_sample.shape[2]
        stats = np.zeros((self._get_effective_feature_count(), sample_size))
        num_ignored = 0
        mask = None if mask is None else reshape_flattened_off_diagonal_elements_to_square(mask, self._is_directed)
        for type_pair in self.sorted_type_pairs:
            if (
                    self._indices_to_ignore is not None and
                    self._indices_to_ignore[self._sorted_type_pairs_indices[type_pair]]
            ):
                num_ignored += 1
                continue
            ix_grid = np.ix_(self.indices_of_types[type_pair[0]], self.indices_of_types[type_pair[1]])
            sub_array = networks_sample[ix_grid]
            if mask is None:
                stat = sub_array.sum(axis=(0, 1))
            else:
                mask_for_types = mask[ix_grid].astype(float)
                stat = np.einsum("ijk,ij->k", sub_array, mask_for_types)
            stats[self._sorted_type_pairs_indices[type_pair] - num_ignored] += stat
        return stats / self._get_num_edges_in_mat_factor()

    # TODO: when we finally make calculate_for_sample the mandatory abstract method in Metric and remove calculate
    #  entirely or return the first matrix returned by calculate_for_sample, change this accordingly as well.
    def calculate(self, network: np.ndarray, mask: npt.NDArray[bool] | None = None):
        return self.calculate_for_sample(expand_net_dims(network), mask=mask)[..., -1]

    def update_indices_to_ignore(self, indices_to_ignore):
        super().update_indices_to_ignore(indices_to_ignore)
        self._effective_feature_count = self._get_effective_feature_count()
        self._indices_to_ignore_up_to_idx = np.cumsum(self._indices_to_ignore)

    def initialize_indices_to_ignore(self):
        super().initialize_indices_to_ignore()
        self.update_indices_to_ignore(self._indices_to_ignore)

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        edge_type_pair = (self.exogenous_attr[indices[0]], self.exogenous_attr[indices[1]])
        idx_in_features_vec = self._sorted_type_pairs_indices[edge_type_pair]

        sign = -1 if current_network[indices[0], indices[1]] else 1

        change_score = np.zeros(self._effective_feature_count)
        if self._indices_to_ignore is not None:
            if self._indices_to_ignore[idx_in_features_vec]:
                return change_score

            change_score[idx_in_features_vec - self._indices_to_ignore_up_to_idx[idx_in_features_vec]] = sign

            return change_score
        else:
            change_score[idx_in_features_vec] = sign
            return change_score

    def calculate_mple_regressors(
            self,
            Xs_out: np.ndarray,
            feature_col_indices: npt.NDArray[np.int64],
            edge_indices_mask: npt.NDArray[bool],
            observed_network=None,
    ):
        # Get the type pair (edge type) index for every entry in the matrix, formatted as Xs (as if the matrix is
        # flattened).
        flattened_edge_type_idx_assignment = self._get_flattened_edge_type_idx_assignment()
        # For each row in Xs (which denotes and entry in the adjacency matrix), update the right column (columns
        # correspond to entries in the feature vector) - the column with the index of the corresponding type pair (the
        # pair of nodes of this entry in the adjacency matrix) in the list of type pairs.
        type_pair_indices_in_metric_cols = flattened_edge_type_idx_assignment[edge_indices_mask]
        # Sum up ignored indices up to each type pair, to subtract the number of ignored indices up to each idx when
        # populating the full Xs that is already indexed such that ignored indices are absent.
        if self._indices_to_ignore is not None:
            ignored_indices_compensation = np.cumsum(self._indices_to_ignore).astype(int)
        else:
            ignored_indices_compensation = np.zeros(self._get_total_feature_count(), dtype=int)

        # Compensate for ignored indices - transform the indexing from the full metric to the metric with ignored
        # indices by subtracting the number of ignored indices up to each entry
        type_pair_indices_in_metric_cols_ignored_idx_comp = (
                type_pair_indices_in_metric_cols - ignored_indices_compensation[type_pair_indices_in_metric_cols]
        )
        # Extract a mask of Xs rows with non-ignored features changes (these rows should be updated in Xs. Others
        # shouldn't be updated - they are ignored). These are exactly the rows for which the index of the columns to
        # update after compensation (which stands for type pair index) is non-negative (i.e., the type pair is not]
        # ignored).
        non_ignored_mask = type_pair_indices_in_metric_cols_ignored_idx_comp >= 0
        type_pair_indices_in_metric_collection_Xs_cols = feature_col_indices[
            type_pair_indices_in_metric_cols_ignored_idx_comp[non_ignored_mask]
        ]
        Xs_out[non_ignored_mask, type_pair_indices_in_metric_collection_Xs_cols] = 1


class NumberOfEdgesTypesUndirected(NumberOfEdgesTypes):
    def __str__(self):
        return "num_edges_between_types_undirected"

    @staticmethod
    def _get_num_edges_in_mat_factor():
        return 2

    def __init__(self, exogenous_attr: Sequence[Any], indices_from_user=None):
        super().__init__(exogenous_attr, indices_from_user)
        self._is_directed = False

    def _get_total_feature_count(self):
        num_unique_types = len(self.unique_types)
        return num_unique_types * (num_unique_types + 1) // 2

    def _get_sorted_canonical_type_pairs(self):
        return [p for p in self.sorted_type_pairs if p == tuple(sorted(p))]

    def _calc_type_pairs_indices(self):
        self._sorted_type_pairs_indices = {}

        sorted_canonical_type_paris = self._get_sorted_canonical_type_pairs()
        for i, (a, b) in enumerate(sorted_canonical_type_paris):
            self._sorted_type_pairs_indices[(a, b)] = i
            if a != b:
                self._sorted_type_pairs_indices[(b, a)] = i

    def _get_flattened_edge_type_idx_assignment(self):
        # Reducing 1 because the type indexing in self._edge_type_idx_assignment is 1-based (0 is reserved for
        # non-exising edges).
        return self._edge_type_idx_assignment[
            np.triu_indices(self._edge_type_idx_assignment.shape[0], k=1)].flatten() - 1

    def _get_metric_names(self):
        parameter_names = tuple()

        metric_name = str(self)

        sorted_canonical_type_pairs = self._get_sorted_canonical_type_pairs()
        for i in range(self._get_total_feature_count()):
            if self._indices_to_ignore is not None and self._indices_to_ignore[i]:
                continue
            type_pair = str(sorted_canonical_type_pairs[i][0]) + "__" + str(sorted_canonical_type_pairs[i][1])
            parameter_names += (f"{metric_name}_{type_pair}",)

        return parameter_names

    def _get_ignored_features(self):
        if self._indices_to_ignore is None or not np.any(self._indices_to_ignore):
            return tuple()

        metric_name = str(self)
        ignored_features = ()
        sorted_canonical_type_pairs = self._get_sorted_canonical_type_pairs()
        for i in range(self._get_total_feature_count()):
            if self._indices_to_ignore is not None and self._indices_to_ignore[i]:
                type_pair = str(sorted_canonical_type_pairs[i][0]) + "__" + str(sorted_canonical_type_pairs[i][1])
                ignored_features += (f"{metric_name}_{type_pair}",)

        return ignored_features


class NumberOfEdgesTypesDirected(NumberOfEdgesTypes):
    """
    A metric that counts how many edges exist between different node types in a directed graph.
    For example -
        A graph with `n` nodes, with an exogenous attribute `type=[A, B]` assigned to each node.
        The metric counts the number of edges between nodes of type A->A, A->B, B->A, B->B of a given graph,
        yielding len(type)**2 features.

    Parameters
    ----------
    exogenous_attr : Collection
        A collection of attributes assigned to each node in a graph with n nodes.

    indices_from_user : list, optional
        List of indices to ignore in the metric calculation.
    """

    def __str__(self):
        return "num_edges_between_types_directed"

    def __init__(self, exogenous_attr: Sequence[Any], indices_from_user=None):
        super().__init__(exogenous_attr, indices_from_user)
        self._is_directed = True

    @staticmethod
    def _get_num_edges_in_mat_factor():
        return 1

    def _get_sorted_canonical_type_pairs(self):
        return self.sorted_type_pairs

    def _calc_type_pairs_indices(self):
        self._sorted_type_pairs_indices = {pair: i for i, pair in enumerate(self.sorted_type_pairs)}

    def _get_total_feature_count(self):
        return len(self.unique_types) ** 2

    def _get_flattened_edge_type_idx_assignment(self):
        # Reducing 1 because the type indexing in self._edge_type_idx_assignment is 1-based (0 is reserved for
        # non-exising edges).
        return self._edge_type_idx_assignment[
            ~np.eye(self._edge_type_idx_assignment.shape[0], dtype=bool)].flatten() - 1

    def _get_metric_names(self):
        parameter_names = tuple()

        metric_name = str(self)

        for i in range(self._get_total_feature_count()):
            if self._indices_to_ignore is not None and self._indices_to_ignore[i]:
                continue
            type_pair = str(self.sorted_type_pairs[i][0]) + "__" + str(self.sorted_type_pairs[i][1])
            parameter_names += (f"{metric_name}_{type_pair}",)

        return parameter_names

    def _get_ignored_features(self):
        if self._indices_to_ignore is None or not np.any(self._indices_to_ignore):
            return tuple()

        metric_name = str(self)
        ignored_features = ()
        for i in range(self._get_total_feature_count()):
            if self._indices_to_ignore is not None and self._indices_to_ignore[i]:
                type_pair = str(self.sorted_type_pairs[i][0]) + "__" + str(self.sorted_type_pairs[i][1])
                ignored_features += (f"{metric_name}_{type_pair}",)

        return ignored_features


class NodeAttrSum(ExWeightNumEdges):
    """
    Sum of node attributes for all edges in the network.

    For each edge (i, j), this metric weights it by the sum of the attributes
    of nodes i and j. The total statistic is the sum of these weights across
    all edges.

    Parameters
    ----------
    exogenous_attr : Collection
        Numeric attribute values for each node.
    is_directed : bool
        Whether the network is directed.
    """
    def __init__(self, exogenous_attr: Collection, is_directed: bool):
        super().__init__(exogenous_attr)
        self._is_directed = is_directed

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self._get_num_weight_mats(), num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                self.edge_weights[0, i, j] = self.exogenous_attr[i] + self.exogenous_attr[j]

    def _get_num_weight_mats(self):
        return 1

    def __str__(self):
        return "node_attribute_sum"


class NodeAttrSumOut(ExWeightNumEdges):
    """
    Sum of sender node attributes for all edges in a directed network.

    For each edge (i, j), this metric weights it by the attribute of the source
    node i. This is useful for modeling sender effects in directed networks.

    Parameters
    ----------
    exogenous_attr : Collection
        Numeric attribute values for each node.
    """
    def __init__(self, exogenous_attr: Collection):
        super().__init__(exogenous_attr)
        self._is_directed = True

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self._get_num_weight_mats(), num_nodes, num_nodes))
        for i in range(num_nodes):
            self.edge_weights[0, i, :] = self.exogenous_attr[i] * np.ones(num_nodes)
            self.edge_weights[0, i, i] = 0

    def _get_num_weight_mats(self):
        return 1

    def __str__(self):
        return "node_attribute_sum_out"


class NodeAttrSumIn(ExWeightNumEdges):
    """
    Sum of receiver node attributes for all edges in a directed network.

    For each edge (i, j), this metric weights it by the attribute of the target
    node j. This is useful for modeling receiver effects in directed networks.

    Parameters
    ----------
    exogenous_attr : Collection
        Numeric attribute values for each node.
    """
    def __init__(self, exogenous_attr: Collection):
        super().__init__(exogenous_attr)
        self._is_directed = True

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.zeros((self._get_num_weight_mats(), num_nodes, num_nodes))
        for j in range(num_nodes):
            self.edge_weights[0, :, j] = self.exogenous_attr[j] * np.ones(num_nodes)
            self.edge_weights[0, j, j] = 0

    def _get_num_weight_mats(self):
        return 1

    def __str__(self):
        return "node_attribute_in"


class SumDistancesConnectedNeurons(ExWeightNumEdges):
    """
    Sum of Euclidean distances between all connected node pairs.

    This metric weights each edge by the Euclidean distance between the spatial
    positions of the connected nodes. Useful for modeling spatial effects in networks.

    Parameters
    ----------
    exogenous_attr : pd.DataFrame, pd.Series, np.ndarray, list, or tuple
        Spatial coordinates for each node. If 1D, interpreted as positions on a line.
        If 2D, each row represents a node and columns represent coordinate dimensions.
    is_directed : bool
        Whether the network is directed.
    """

    def __init__(self, exogenous_attr: pd.DataFrame | pd.Series | np.ndarray | list | tuple, is_directed: bool):
        # todo: decorator that checks inputs and returns np.ndarray if the inputs were valid, and an error otherwise
        if isinstance(exogenous_attr, (pd.DataFrame, pd.Series, list, tuple)):
            exogenous_attr = np.array(exogenous_attr)
        elif not isinstance(exogenous_attr, np.ndarray):
            raise ValueError(
                f"Unsupported type of positions: {type(exogenous_attr)}. Supported types are pd.DataFrame, "
                f"pd.Series, list, tuple and np.ndarray.")
        if len(exogenous_attr.shape) == 1:
            exogenous_attr = exogenous_attr.reshape(-1, 1)

        super().__init__(exogenous_attr)
        self._is_directed = is_directed

    def _calc_edge_weights(self):
        num_nodes = len(self.exogenous_attr)
        self.edge_weights = np.reshape(squareform(pdist(self.exogenous_attr, metric='euclidean')),
                                       (self._get_num_weight_mats(), num_nodes, num_nodes))

    def calc_change_score(self, current_network: np.ndarray, indices: tuple):
        sign = -1 if current_network[indices[0], indices[1]] else 1
        return sign * self.edge_weights[:, indices[0], indices[1]]

    def _get_num_weight_mats(self):
        return 1

    def __str__(self):
        return "sum_distances_connected_neurons"


class NumberOfNodesPerType(Metric):
    """
    Count the number of nodes in each category.

    This metric operates on node features (rather than edges) and counts how many
    nodes belong to each category. Returns n_categories - 1 features to avoid
    multicollinearity (the last category is redundant given the total node count).

    Parameters
    ----------
    metric_node_feature : str
        Name of the node feature to operate on.
    n_node_categories : int
        Total number of categories that nodes can belong to.
    feature_dim : int, optional
        Dimensionality of the node feature. Default is 1.
    """
    def __str__(self):
        return "num_nodes_per_type"

    def __init__(self, metric_node_feature, n_node_categories, feature_dim=1):
        super().__init__(requires_graph=False, metric_type='node', metric_node_feature=metric_node_feature)
        self._is_dyadic_independent = True
        self.n_node_categories = n_node_categories
        self.feature_dim = feature_dim

    def _get_total_feature_count(self):
        """
        How many features does this metric produce, including the ignored ones.
        """
        return self.n_node_categories - 1

    def calculate(self, V: np.ndarray):
        V = V.astype(int)
        if V.shape[1] != 1:
            raise ValueError("the metric NumberOfNodesPerType only works for one kind of node feature.")
        else:
            V = V[:, 0]
        return self._handle_indices_to_ignore(
            np.bincount(V, minlength=self.n_node_categories)[:-1])  # last category is redundant

    def calc_change_score(self, current_node_features: np.ndarray, idx: int, new_category: int, feature_to_flip=None):
        if current_node_features.shape[1] != 1:
            raise ValueError("the metric NumberOfNodesPerType only works for one kind of node feature.")
        else:
            current_node_features = current_node_features[:, 0]
        old_category = current_node_features[idx]
        changes = np.zeros(self.n_node_categories)
        changes[old_category] = -1
        changes[new_category] = 1
        return self._handle_indices_to_ignore(changes[:-1])  # last category is redundant

    def calculate_for_sample(self, networks_sample: np.ndarray | torch.Tensor):
        """
        We use a trick here to make the bincount operation vectorized: we add each column of the sample
        (namely, features of a different network) a distinct offset, then flatten the sample. Now, we can
        perform bincount on the flattened sample, and reshape it to get back to the counts per network.
        """
        networks_sample = networks_sample.astype(int)
        if networks_sample.shape[1] != 1:
            raise ValueError("the metric NumberOfNodesPerType only works for one kind of node feature.")
        else:
            networks_sample = networks_sample[:, 0]

        n_samples = networks_sample.shape[-1]
        offsets = np.arange(n_samples)[None, :] * self.n_node_categories
        flat_indices = networks_sample + offsets
        flat_counts = np.bincount(flat_indices.ravel(), minlength=n_samples * self.n_node_categories)
        counts_for_sample = flat_counts.reshape(n_samples, self.n_node_categories).T

        return self._handle_indices_to_ignore(counts_for_sample[:-1])  # last category is redundant


class MetricsCollection:
    """
    A collection of metrics for ERGM models.

    This class manages multiple metrics, handles feature calculations across samples,
    prepares data for MPLE optimization, and automatically detects and removes
    collinear features.

    Parameters
    ----------
    metrics : Collection[Metric]
        Collection of Metric instances to include in the model.
    is_directed : bool
        Whether the network is directed.
    n_nodes : int
        Number of nodes in the network.
    fix_collinearity : bool, optional
        If True, automatically detect and remove collinear features. Default is True.
    use_sparse_matrix : bool, optional
        If True, use sparse matrix representations for efficiency. Default is False.
    collinearity_fixer_sample_size : int, optional
        Number of random networks to sample for collinearity detection. Default is 1000.
    is_collinearity_distributed : bool, optional
        If True, distribute collinearity fixing computation. Default is False.
    do_copy_metrics : bool, optional
        If True, create deep copies of input metrics. Default is True.
    mask : np.ndarray, optional
        Boolean mask indicating which edges to consider (1D flattened). Default is None.
    **kwargs
        Additional keyword arguments for collinearity fixer configuration.
    """

    def __init__(self,
                 metrics: Collection[Metric],
                 is_directed: bool,
                 n_nodes: int,
                 fix_collinearity=True,
                 use_sparse_matrix=False,
                 collinearity_fixer_sample_size=1000,
                 is_collinearity_distributed=False,
                 # TODO: For tests only, find a better solution
                 do_copy_metrics=True,
                 mask: npt.NDArray[bool] | None = None,
                 **kwargs):
        logger.debug("Initializing MetricsCollection")
        sys.stdout.flush()
        if not do_copy_metrics:
            self.metrics = tuple([metric for metric in metrics])
        else:
            self.metrics = tuple([deepcopy(metric) for metric in metrics])

        for m in self.metrics:
            m._n_nodes = n_nodes
            m.initialize_indices_to_ignore()

        self.is_directed = is_directed
        self.mask = None
        if mask is not None:
            if mask.ndim != 1:
                raise ValueError(f"MetricsCollection expects flattened masks (1D). Got {mask.ndim}D")
            self.mask = mask.copy()
        for x in self.metrics:
            if (x._is_directed is not None) and (x._is_directed != self.is_directed):
                model_is_directed_str = "a directed" if self.is_directed else "an undirected"
                metric_is_directed_str = "a directed" if x._is_directed else "an undirected"
                raise ValueError(f"Trying to initialize {model_is_directed_str} model with {metric_is_directed_str} "
                                 f"metric `{str(x)}`!")
            if self.mask is not None and not x.does_support_mask:
                raise ValueError(f"Trying to initialize a masked model with metric `{str(x)}` which does not support "
                                 f"masks!")
        self.n_nodes = n_nodes
        node_feature_name_to_dim = {m.metric_node_feature: m.feature_dim for m in self.metrics if
                                    m._metric_type == 'node'}
        node_features_dims = list(node_feature_name_to_dim.values())
        node_features_indices_limits = np.cumsum(np.concatenate([[0], node_features_dims])).astype(int)
        node_features_indices = [np.arange(node_features_indices_limits[i], node_features_indices_limits[i + 1])
                                 for i in range(len(node_features_indices_limits) - 1)]
        node_feature_names = list(node_feature_name_to_dim.keys())
        self.node_feature_names = dict(zip(node_feature_names,
                                           node_features_indices))  # a dict with keys of node feature names and values of lists of indices
        self.n_node_features = sum([len(v) for v in self.node_feature_names.values()])
        self.node_features_n_categories = {m.metric_node_feature: m.n_node_categories for m in self.metrics if
                                           m._metric_type == 'node'}

        self.use_sparse_matrix = use_sparse_matrix
        self.requires_graph = any([x.requires_graph for x in self.metrics])

        # Returns the number of features that are being calculated. Since a single metric might return more than one
        # feature, the length of the statistics vector might be larger than the amount of metrics. Since it also depends
        # on the network size, n is a mandatory parameters. That's why we're using the get_effective_feature_count
        # function
        self.num_of_features = self.calc_num_of_features()
        self.features_per_metric = np.array([metric._get_effective_feature_count() for metric in self.metrics])

        self._fix_collinearity = fix_collinearity
        self.collinearity_fixer_sample_size = collinearity_fixer_sample_size
        if self._fix_collinearity:
            logger.debug("Starting collinearity fix")
            self.collinearity_fixer(sample_size=self.collinearity_fixer_sample_size,
                                    is_distributed=is_collinearity_distributed,
                                    num_samples_per_job=kwargs.get('num_samples_per_job_collinearity_fixer', 5),
                                    ratio_threshold=kwargs.get('ratio_threshold_collinearity_fixer', 5e-6),
                                    nonzero_thr=kwargs.get('nonzero_threshold_collinearity_fixer', 0.1),
                                    )

        self.num_of_metrics = len(self.metrics)
        self.metric_names = tuple([str(metric) for metric in self.metrics])
        self._has_dyadic_dependent_metrics = any([not x._is_dyadic_independent for x in self.metrics])

    def _delete_metric(self, metric: Metric):
        self.metrics = tuple([m for m in self.metrics if m != metric])
        self.requires_graph = any([x.requires_graph for x in self.metrics])

    def calc_num_of_features(self):
        return sum([metric._get_effective_feature_count() for metric in self.metrics])

    def get_metric(self, metric_name: str) -> Metric:
        """
        Get a metric instance
        """
        return self.metrics[self.metric_names.index(metric_name)]

    def get_metric_by_feat_idx(self, idx: int):
        cum_sum_num_feats = 0
        for m in self.metrics:
            cum_sum_num_feats += m._get_effective_feature_count()
            if cum_sum_num_feats > idx:
                return m

    def get_feature_idx_within_metric(self, idx: int):
        cum_sum_num_feats = 0
        for m_idx in range(len(self.metrics)):
            next_met_num_feats = self.metrics[m_idx]._get_effective_feature_count()
            if cum_sum_num_feats + next_met_num_feats > idx:
                # We want to return the index in the "full" array, namely regardless of ignored features. So, in case
                # there are indices that are ignored, we must take the returned index from the array of non-ignored
                # indices, to compensate for the ones ignored by the indexing of MetricCollection (which holds only
                # effective features, after ignoring).
                # For example - if we ignore the degree of the first node, and now want to ignore the degree of the
                # fifth node as well, we must return 4 for the metric to correctly ignore it. But it is the fourth
                # feature corresponding to the metric in the MetricCollection vector of features (not fifth, because the
                # first is missing, not returned by the Metric as it's ignored). By returning the fourth element from
                # the list of non-ignored indices, which is [1, 2, ..., n], we solve the problem.

                effective_idx_within_metric = idx - cum_sum_num_feats
                if hasattr(self.metrics[m_idx], "_indices_to_ignore"):
                    non_ignored_indices = np.where(~self.metrics[m_idx]._indices_to_ignore)[0]
                    # Return the index of the feature with relation to the whole set of features (without ignoring).
                    return non_ignored_indices[effective_idx_within_metric]
                else:
                    return effective_idx_within_metric
            else:
                cum_sum_num_feats += next_met_num_feats

    def calc_statistics_for_binomial_tensor_local(self, tensor_size, p=0.5):
        sample = generate_binomial_tensor(self.n_nodes, self.n_node_features, tensor_size, p=p)

        # Symmetrize samples if not directed
        if not self.is_directed:
            sample[:self.n_nodes, :self.n_nodes] = np.round(
                (sample[:self.n_nodes, :self.n_nodes] + sample[:self.n_nodes, :self.n_nodes].transpose(1, 0, 2)) / 2)

        # Make sure the main diagonal is 0
        sample[np.arange(self.n_nodes, dtype=int), np.arange(self.n_nodes, dtype=int), :] = 0

        # Calculate the features of the sample
        return self.calculate_sample_statistics(sample)

    def calc_statistics_for_binomial_tensor_distributed(self, tensor_size, p=0.5, num_samples_per_job=1):
        # TODO: currently, if tensor_size % num_samples_per_job != 0 the sample size will be larger than tensor_size
        #  specified by the user (it is (tensor_size // num_samples_per_job + 1) * num_samples_per_job). We can pass
        #  tensor_size as an additional argument to children jobs, and validate in
        #  sample_statistics_distributed_calcs.py that we don't calculate for too many networks.
        out_dir_path = (Path.cwd().parent / "OptimizationIntermediateCalculations").resolve()
        data_path = (out_dir_path / "data").resolve()
        os.makedirs(data_path, exist_ok=True)

        with open((data_path / 'metric_collection.pkl').resolve(), 'wb') as f:
            pickle.dump(self, f)

        cmd_line_for_bsub = (f'python ./sample_statistics_distributed_calcs.py '
                             f'--out_dir_path {out_dir_path} '
                             f'--num_samples_per_job {num_samples_per_job} '
                             f'--p {p}')

        num_jobs = int(np.ceil(tensor_size / num_samples_per_job))
        logger.debug(f"Sending {num_jobs} children job array for collinearity fixer")
        job_array_ids, children_logs_dir = run_distributed_children_jobs(out_dir_path, cmd_line_for_bsub,
                                                                         "distributed_binomial_tensor_statistics.sh",
                                                                         num_jobs, "sample_stats")

        # Wait for all jobs to finish.
        sample_stats_path = (out_dir_path / "sample_statistics").resolve()
        os.makedirs(sample_stats_path, exist_ok=True)
        wait_for_distributed_children_outputs(num_jobs, [sample_stats_path], job_array_ids, "sample_stats",
                                              children_logs_dir)

        # Clean current scripts and data
        shutil.rmtree(data_path)
        shutil.rmtree((out_dir_path / "scripts").resolve())

        # Aggregate results
        whole_sample_statistics = cat_children_jobs_outputs(num_jobs, sample_stats_path, axis=1)

        # clean outputs
        shutil.rmtree(sample_stats_path)
        return whole_sample_statistics

    def remove_feature_by_idx(self, idx: int):
        metric_of_feat = self.get_metric_by_feat_idx(idx)
        is_trimmable = metric_of_feat._get_effective_feature_count() > 1
        if not is_trimmable:
            logger.info(f"Removing the metric {str(metric_of_feat)} from the collection")
            sys.stdout.flush()
            self._delete_metric(metric=metric_of_feat)
        else:
            idx_to_delete_within_metric = self.get_feature_idx_within_metric(idx)
            logger.info(f"Removing the {idx_to_delete_within_metric} feature of {str(metric_of_feat)}")
            sys.stdout.flush()
            metric_of_feat.update_indices_to_ignore([idx_to_delete_within_metric])
        self.num_of_features = self.calc_num_of_features()

        # Re-calculate the number of features per metric after deleting a feature.
        self.features_per_metric = np.array([metric._get_effective_feature_count() for metric in self.metrics])

    def collinearity_fixer(self, sample_size=1000, nonzero_thr=10 ** -1, ratio_threshold=10 ** -6,
                           eigenvec_thr=10 ** -4, is_distributed=False, **kwargs):
        """
        Find collinearity between metrics in the collection.

        Currently this is a naive version that only handles the very simple cases.
        TODO: revisit the threshold and sample size
        """
        is_linearly_dependent = True

        # Sample networks from a maximum entropy distribution, for avoiding edge cases (such as a feature is 0 for
        # all networks in the sample).
        if not is_distributed:
            sample_features = self.calc_statistics_for_binomial_tensor_local(sample_size)
        else:
            logger.debug("Starting distributed collinearity fixing")
            sample_features = self.calc_statistics_for_binomial_tensor_distributed(sample_size,
                                                                                   num_samples_per_job=kwargs.get(
                                                                                       "num_samples_per_job", 5))
        while is_linearly_dependent:
            features_cov_mat = sample_features @ sample_features.T

            # Determine whether the matrix of features is invertible. If not - this means there is a non-trivial vector,
            # that when multiplied by the matrix gives the 0 vector. Namely, there is a single set of coefficients that
            # defines a non-trivial linear combination that equals 0, for *all* the sampled feature vectors. This means
            # the features are linearly dependent.
            eigen_vals, eigen_vecs = np.linalg.eigh(features_cov_mat)

            minimal_non_zero_eigen_val = np.min(np.abs(eigen_vals[np.abs(eigen_vals) > nonzero_thr]))
            small_eigen_vals_indices = np.where(np.abs(eigen_vals) / minimal_non_zero_eigen_val < ratio_threshold)[0]

            if small_eigen_vals_indices.size == 0:
                is_linearly_dependent = False
            else:
                logger.info("Collinearity detected, identifying features to remove")
                sys.stdout.flush()

                # For each linear dependency (corresponding to an eigen vector with a low value), mark the indices of
                # features that are involved (identified by a non-zero coefficient in the eigen vector).
                dependent_features_flags = np.zeros((small_eigen_vals_indices.size, self.num_of_features))
                for i in range(small_eigen_vals_indices.size):
                    dependent_features_flags[
                        i, np.where(np.abs(eigen_vecs[:, small_eigen_vals_indices[i]]) > eigenvec_thr)[0]] = 1

                # Calculate the fraction of dependencies each feature is involved in.
                fraction_of_dependencies_involved = dependent_features_flags.mean(axis=0)

                # Sort the features (their indices) by the fraction of dependencies they are involved in (remove first
                # features that are involved in more dependencies). Break ties by the original order of the features in
                # the array (for the consistency of sorting. E.g, if we need to get rid of degree features, always
                # remove them by the order of nodes).
                removal_order = np.lexsort((np.arange(self.num_of_features), -fraction_of_dependencies_involved))

                # Iterate the metrics to find one with multiple features, namely effective number of features that is
                # larger than 1 ('trimmable'). We prefer to trim metrics rather than totally eliminate them from the
                # collection.
                i = 0
                cur_metric = self.get_metric_by_feat_idx(removal_order[i])
                is_trimmable = cur_metric._get_effective_feature_count() > 1
                while (not is_trimmable and i < removal_order.size - 1 and fraction_of_dependencies_involved[
                    removal_order[i]] > 0):
                    i += 1
                    cur_metric = self.get_metric_by_feat_idx(removal_order[i])
                    is_trimmable = cur_metric._get_effective_feature_count() > 1

                # If a trimmable metric was not found (i.e., all features that are involved in the dependency are of
                # metrics with an effective number of features of 1), totally remove the metric that is involved in most
                # dependencies.
                if not is_trimmable:
                    i = 0
                self.remove_feature_by_idx(removal_order[i])

                sample_features = np.delete(sample_features, removal_order[i], axis=0)

    def calculate_statistics(self, W: np.ndarray):
        """
        Calculate the statistics of a graph, formally written as g(y).

        Parameters
        ----------
        W : np.ndarray
            A generalized connectivity matrix: a concatenation of N x N connectivity matrix with an N x k node features matrix.
        
        Returns
        -------
        statistics : np.ndarray
            An array of statistics
        """
        if self.requires_graph:
            G = connectivity_matrix_to_G(W[:self.n_nodes, :self.n_nodes], directed=self.is_directed)

        statistics = np.zeros(self.num_of_features)

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:  # it cannot require graph and also have _metric_type='node'
                input = G
            else:
                if metric._metric_type in ['binary_edge', 'non_binary_edge']:
                    input = W[:self.n_nodes, :self.n_nodes]
                elif metric._metric_type == 'node':
                    feature_indices_to_pass = self.node_feature_names.get(metric.metric_node_feature,
                                                                          list(np.arange(self.n_node_features)))
                    feature_indices_to_pass = [i + self.n_nodes for i in feature_indices_to_pass]
                    input = W[:, feature_indices_to_pass]

            n_features_from_metric = metric._get_effective_feature_count()
            statistics[feature_idx:feature_idx + n_features_from_metric] = metric.calculate(input)
            feature_idx += n_features_from_metric

        return statistics

    def calc_change_scores(self, current_network: np.ndarray, edge_flip_info: dict, node_flip_info={}) -> np.ndarray:
        """
        Calculates the vector of change scores, namely g(net_2) - g(net_1)

        NOTE - this function assumes that the size current_network and self.n_nodes are the same, and doesn't validate
        it, due to runtime considerations. Currently, the only use of this function is within ERGM and
        NaiveMetropolisHastings, so this is fine.
        """
        if self.requires_graph:
            G1 = connectivity_matrix_to_G(current_network[:self.n_nodes, :self.n_nodes], directed=self.is_directed)

        change_scores = np.zeros(self.num_of_features)

        feature_idx = 0
        for i, metric in enumerate(self.metrics):
            # n_features_from_metric = metric._get_effective_feature_count()
            n_features_from_metric = self.features_per_metric[i]

            if metric.requires_graph:  # it cannot require graph and also have _metric_type='node'
                input = G1
                change_scores[feature_idx:feature_idx + n_features_from_metric] = (
                    metric.calc_change_score(input, edge_flip_info['edge'])
                )
            else:
                if metric._metric_type in ['binary_edge', 'non_binary_edge']:
                    input = current_network[:self.n_nodes, :self.n_nodes]
                    change_scores[feature_idx:feature_idx + n_features_from_metric] = (
                        metric.calc_change_score(input, edge_flip_info['edge'])
                    )
                elif metric._metric_type == 'node':
                    feature_indices_to_pass = self.node_feature_names.get(metric.metric_node_feature,
                                                                          list(np.arange(self.n_node_features)))
                    feature_indices_to_pass = [i + self.n_nodes for i in feature_indices_to_pass]
                    if node_flip_info['feature'] not in feature_indices_to_pass:
                        continue
                    input = current_network[:, feature_indices_to_pass]
                    change_scores[feature_idx:feature_idx + n_features_from_metric] = (
                        metric.calc_change_score(
                            input,
                            current_network[:, feature_indices_to_pass],
                            node_flip_info['node'],
                            node_flip_info['new_category'],
                        )
                    )
            feature_idx += n_features_from_metric

        return change_scores

    def calculate_sample_statistics(
            self,
            networks_sample: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the statistics over a sample of networks

        Parameters
        ----------
        networks_sample
            The networks sample - an array of n X (n+k) X sample_size
        Returns
        -------
        an array of the statistics vector per sample (num_features X sample_size)
        """
        if networks_sample.shape[0] != self.n_nodes:
            raise ValueError(
                f"Got a sample of networks of size {networks_sample.shape[0]}, but Metrics expect size {self.n_nodes}")

        num_of_samples = networks_sample.shape[2]
        features_of_net_samples = np.zeros((self.num_of_features, num_of_samples))

        if self.requires_graph:
            networks_as_graphs = [connectivity_matrix_to_G(W[:self.n_nodes, :self.n_nodes], self.is_directed) for W in
                                  networks_sample]

        if self.use_sparse_matrix:
            networks_as_sparse_tensor = np_tensor_to_sparse_tensor(networks_sample)

        feature_idx = 0
        for metric in self.metrics:
            n_features_from_metric = metric._get_effective_feature_count()

            if metric.requires_graph:
                networks = networks_as_graphs

            elif self.use_sparse_matrix:
                if metric._metric_type in ['binary_edge', 'non_binary_edge']:
                    networks = networks_as_sparse_tensor[:self.n_nodes, :self.n_nodes]
                elif metric._metric_type == 'node':
                    feature_indices_to_pass = self.node_feature_names.get(metric.metric_node_feature,
                                                                          list(np.arange(self.n_node_features)))
                    feature_indices_to_pass = [i + self.n_nodes for i in feature_indices_to_pass]
                    networks = networks_as_sparse_tensor[:, feature_indices_to_pass]
            else:
                if metric._metric_type in ['binary_edge', 'non_binary_edge']:
                    networks = networks_sample[:self.n_nodes, :self.n_nodes]
                elif metric._metric_type == 'node':
                    feature_indices_to_pass = self.node_feature_names.get(metric.metric_node_feature,
                                                                          list(np.arange(self.n_node_features)))
                    feature_indices_to_pass = [i + self.n_nodes for i in feature_indices_to_pass]
                    networks = networks_sample[:, feature_indices_to_pass]

            calc_for_sample_kwargs = {'networks_sample': networks}
            if self.mask is not None:
                calc_for_sample_kwargs |= {'mask': self.mask}
            features = metric.calculate_for_sample(**calc_for_sample_kwargs)

            if isinstance(features, torch.Tensor):
                if features.is_sparse:
                    features = features.to_dense()
                features = features.numpy()

            features_of_net_samples[feature_idx:feature_idx + n_features_from_metric] = features
            feature_idx += n_features_from_metric

        return features_of_net_samples

    def calculate_change_scores_all_edges(self, current_network: np.ndarray):
        """
        Calculates the vector of change scores for every edge in the network.

        Parameters
        ----------
        current_network : np.ndarray
            The network matrix of size (n,n+k).
        
        Returns
        -------
        change_scores : np.ndarray
            A matrix of size (n**2-n, num_of_features), where each row corresponds to the change scores of the i,j-th edges.
        """
        if self.requires_graph:
            G1 = connectivity_matrix_to_G(current_network[:self.n_nodes, :self.n_nodes], directed=self.is_directed)

        n_nodes = current_network.shape[0]
        num_edges = n_nodes * n_nodes - n_nodes
        change_scores = np.zeros((num_edges, self.num_of_features))

        feature_idx = 0
        for metric in self.metrics:
            if metric.requires_graph:
                input = G1
            else:
                if metric._metric_type in ['binary_edge', 'non_binary_edge']:
                    input = current_network[:self.n_nodes, :self.n_nodes]
                elif metric._metric_type == 'node':
                    feature_indices_to_pass = self.node_feature_names.get(metric.metric_node_feature,
                                                                          list(np.arange(self.n_node_features)))
                    feature_indices_to_pass = [i + self.n_nodes for i in feature_indices_to_pass]
                    input = current_network[:, feature_indices_to_pass]

            n_features_from_metric = metric._get_effective_feature_count()

            if hasattr(metric, "calculate_change_score_full_network"):
                change_scores[:,
                feature_idx:feature_idx + n_features_from_metric] = metric.calculate_change_score_full_network(
                    input)
            else:
                edge_idx = 0
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i == j:
                            continue
                        indices = (i, j)
                        change_scores[edge_idx,
                        feature_idx:feature_idx + n_features_from_metric] = metric.calc_change_score(input, indices)
                        edge_idx += 1

            feature_idx += n_features_from_metric

        return change_scores

    def _get_mple_data_chunk_mask(
            self,
            edge_indices_lims: tuple[int, int] | None,
    ) -> npt.NDArray[bool]:
        full_net_size = self.n_nodes * self.n_nodes - self.n_nodes
        if not self.is_directed:
            full_net_size //= 2
        data_chunk_mask = np.zeros(full_net_size, dtype=bool)
        global_mask = self.mask if self.mask is not None else np.ones(full_net_size, dtype=bool)
        if edge_indices_lims is None:
            edge_indices_lims = (0, global_mask.sum())
        if (
                edge_indices_lims[0] < 0 or edge_indices_lims[1] < 0 or
                edge_indices_lims[0] > global_mask.sum() or edge_indices_lims[1] > global_mask.sum() or
                edge_indices_lims[0] >= edge_indices_lims[1]
        ):
            raise ValueError(
                'edge_indices_lims out of bounds. expected strictly monotonic increasing limits between 0 '
                'and the number of considered edges (the size of the dataset for MPLE optimization), which is '
                f'{global_mask.sum()}. Got {edge_indices_lims}')
        # First constraint to the "universe" of edge indices to consider by the global mask, then slice according to the
        # limits within the subset of considered edges.
        data_chunk_mask[np.where(global_mask)[0][edge_indices_lims[0]:edge_indices_lims[1]]] = True
        return data_chunk_mask

    def prepare_mple_regressors(
            self,
            observed_network: np.ndarray | None = None,
            edge_indices_lims: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if self.requires_graph:
            G1 = connectivity_matrix_to_G(observed_network[:self.n_nodes, :self.n_nodes], directed=self.is_directed)

        data_chunk_mask = self._get_mple_data_chunk_mask(edge_indices_lims)
        Xs = np.zeros((data_chunk_mask.sum(), self.num_of_features))

        feature_idx = 0
        for metric in self.metrics:
            if observed_network is not None:
                if metric.requires_graph:
                    input = G1
                else:
                    if metric._metric_type in ['binary_edge', 'non_binary_edge']:
                        input = observed_network[:self.n_nodes, :self.n_nodes]
                    elif metric._metric_type == 'node':
                        feature_indices_to_pass = self.node_feature_names.get(metric.metric_node_feature,
                                                                              list(np.arange(self.n_node_features)))
                        feature_indices_to_pass = [i + self.n_nodes for i in feature_indices_to_pass]
                        input = observed_network[:, feature_indices_to_pass]
            else:
                input = None

            n_features_from_metric = metric._get_effective_feature_count()
            metric.calculate_mple_regressors(
                Xs_out=Xs,
                feature_col_indices=np.arange(feature_idx, feature_idx + n_features_from_metric, dtype=int),
                observed_network=input,
                edge_indices_mask=data_chunk_mask,
            )
            feature_idx += n_features_from_metric
        return Xs

    def prepare_mple_labels(
            self,
            observed_networks: np.ndarray,
            edge_indices_lims: tuple[int, int] | None = None,
    ) -> np.ndarray:
        observed_networks = expand_net_dims(observed_networks)
        chunk_mask = self._get_mple_data_chunk_mask(edge_indices_lims)
        ys = np.zeros((chunk_mask.sum(), 1))
        num_nets = observed_networks.shape[-1]
        for net_idx in range(num_nets):
            net = observed_networks[..., net_idx]
            ys += flatten_square_matrix_to_edge_list(net, self.is_directed)[chunk_mask].reshape(-1, 1)
        return ys / num_nets

    def prepare_mple_reciprocity_regressors(self):
        Xs = np.zeros(((self.n_nodes ** 2 - self.n_nodes) // 2, 4, self.calc_num_of_features()))
        idx = 0
        zeros_net = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes - 1):
            for j in range(i + 1, self.n_nodes):
                change_score_i_j = self.calc_change_scores(zeros_net, {'edge': (i, j)})
                net_with_i_j = zeros_net.copy()
                net_with_i_j[i, j] = 1
                Xs[idx, UPPER_IDX] = change_score_i_j
                Xs[idx, LOWER_IDX] = self.calc_change_scores(zeros_net, {'edge': (j, i)})
                Xs[idx, RECIPROCAL_IDX] = self.calc_change_scores(net_with_i_j, {'edge': (j, i)}) + change_score_i_j

                idx += 1
        return Xs

    @staticmethod
    def prepare_mple_reciprocity_labels(observed_networks: np.ndarray):
        ys = convert_connectivity_to_dyad_states(observed_networks[..., 0])
        num_nets = observed_networks.shape[-1]
        for i in range(1, num_nets):
            ys += convert_connectivity_to_dyad_states(observed_networks[..., i])
        return ys / num_nets

    def choose_optimization_scheme(self):
        """
        Automatically select the appropriate optimization scheme for the model.

        Returns
        -------
        str
            One of 'MPLE', 'MPLE_RECIPROCITY', or 'MCMLE' depending on the metrics.

        Notes
        -----
        - 'MPLE': Maximum Pseudo-Likelihood Estimation, for dyadic independent metrics
        - 'MPLE_RECIPROCITY': Extended MPLE for models with only reciprocity dependence
        - 'MCMLE': Monte Carlo Maximum Likelihood Estimation, for complex dependencies
        """
        if self.n_node_features > 0:
            return 'MCMLE'
        if not self._has_dyadic_dependent_metrics:
            return 'MPLE'
        # The only edge dependence comes from reciprocal edges
        if not any(
                [not x._is_dyadic_independent for x in self.metrics if
                 str(x) not in ['total_reciprocity', 'reciprocity']]):
            return 'MPLE_RECIPROCITY'
        return 'MCMLE'

    def get_parameter_names(self):
        """
        Returns the names of the parameters of the metrics in the collection.
        """
        parameter_names = tuple()

        for metric in self.metrics:
            parameter_names += metric._get_metric_names()

        return parameter_names

    def get_ignored_features(self):
        """
        Get the names of features that have been ignored due to collinearity.

        Returns
        -------
        tuple
            Names of ignored features across all metrics in the collection.
        """
        parameter_names = tuple()

        for metric in self.metrics:
            parameter_names += metric._get_ignored_features()

        return parameter_names

    def bootstrap_observed_features(self, observed_network: np.ndarray, num_subsamples: int = 1000,
                                    splitting_method: str = 'uniform'):
        if observed_network.ndim == 3:
            observed_network = observed_network[..., 0]
        observed_connectivity_matrix = observed_network[:self.n_nodes, :self.n_nodes]
        observed_net_size = observed_connectivity_matrix.shape[0]
        second_half_size = observed_net_size // 2
        first_half_size = observed_net_size - second_half_size
        first_halves = np.zeros((first_half_size, first_half_size, num_subsamples))
        second_halves = np.zeros((second_half_size, second_half_size, num_subsamples))
        first_halves_indices = np.zeros((first_half_size, num_subsamples), dtype=int)
        second_halves_indices = np.zeros((second_half_size, num_subsamples), dtype=int)
        for i in range(num_subsamples):
            # TODO: currently we simply split randomly into 2 halves, we want to support other methods for sampling half
            #  of the neurons, and specifically sampling half of the neurons of each if there is a metric with types
            #  involved.
            #  NOTE! This will probably imposes to have another field in `Metric` indicating what is the preferred way
            #  for bootstrapping sampling, and `MetricsCollection` will have to decide somehow what to do based on the
            #  values of all metrics.
            #  NOTE! If there are multiple type metrics, we will probably need to sample according to sub-types defined
            #  as the Cartesian product of all types.
            first_half_indices, second_half_indices = split_network_for_bootstrapping(observed_net_size,
                                                                                      first_half_size,
                                                                                      splitting_method=splitting_method)
            first_halves[:, :, i] = observed_connectivity_matrix[first_half_indices, first_half_indices.T]
            second_halves[:, :, i] = observed_connectivity_matrix[second_half_indices, second_half_indices.T]
            first_halves_indices[:, i] = first_half_indices[:, 0]
            second_halves_indices[:, i] = second_half_indices[:, 0]

        bootstrapped_features = np.zeros((self.num_of_features, num_subsamples))
        # TODO: the next code section is copied from `calculate_for_sample`. The difference is that here we use
        #  `first_halves` and `second_halves` instead of `networks_sample` and a different callable for each metric.
        #  Maybe is can be handled in a single method that gets a callable for metrics and **kwargs or something like
        #  that.
        if self.requires_graph:
            first_halves_as_graphs = [connectivity_matrix_to_G(W, self.is_directed) for W in first_halves]
            second_halves_as_graphs = [connectivity_matrix_to_G(W, self.is_directed) for W in second_halves]

        if self.use_sparse_matrix:
            first_halves_as_sparse_tensor = np_tensor_to_sparse_tensor(first_halves)
            second_halves_as_sparse_tensor = np_tensor_to_sparse_tensor(second_halves)

        feature_idx = 0
        for metric in self.metrics:
            n_features_from_metric = metric._get_effective_feature_count()

            if metric.requires_graph:
                first_halves_to_use = first_halves_as_graphs
                second_halves_to_use = second_halves_as_graphs
            elif self.use_sparse_matrix:
                first_halves_to_use = first_halves_as_sparse_tensor
                second_halves_to_use = second_halves_as_sparse_tensor
            else:
                first_halves_to_use = first_halves
                second_halves_to_use = second_halves

            cur_metric_bootstrapped_features = metric.calculate_bootstrapped_features(
                first_halves_to_use, second_halves_to_use,
                first_halves_indices, second_halves_indices)

            if isinstance(cur_metric_bootstrapped_features, torch.Tensor):
                if cur_metric_bootstrapped_features.is_sparse:
                    cur_metric_bootstrapped_features = cur_metric_bootstrapped_features.to_dense()
                cur_metric_bootstrapped_features = cur_metric_bootstrapped_features.numpy()

            bootstrapped_features[feature_idx:feature_idx + n_features_from_metric] = cur_metric_bootstrapped_features
            feature_idx += n_features_from_metric

        return bootstrapped_features
