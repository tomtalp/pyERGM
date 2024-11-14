import numpy as np
import pandas as pd
from scipy.spatial import distance
#from pyERGM.metrics import Metric
import unittest

print("hi")
# class tSumDistancesConnectedNeurons():
#     """
#     This function calculates the sum of euclidean distances between all pairs of connected neurons.
#     """
#
#     def __init__(self, positions):
#         super().__init__(requires_graph=False)
#         self._is_directed = True
#         self._is_dyadic_independent = True
#         self._positions = positions
#
#     def calculate(self, input_graph: np.ndarray):
#         if isinstance(self._positions, (pd.DataFrame, pd.Series)):
#             self._positions = self._positions.values
#         if len(self._positions.shape) == 1:
#             self._positions = self._positions.reshape(-1, 1)
#         dist_matrix = distance.pdist(self._positions, metric='euclidean')
#         dist_square = distance.squareform(dist_matrix)
#         distances = dist_square[np.where(input_graph)]
#         return sum(distances)
#
#
# class tTestSumDistancesConnectedNeurons(unittest.TestCase):
#     def test_sum_distances(self):
#         positions = pd.DataFrame({"x_pos": [0, 0, 4], "y_pos": [0, 3, 0], "z_pos": [2, 2, 2]})
#         W = np.array([[0, 1, 0],
#                       [1, 0, 1],
#                       [1, 0, 0]])
#
#         # Dataframe with multiple columns
#         metric_1 = tSumDistancesConnectedNeurons(positions)
#         expected_res_1 = 3 + 3 + 5 + 4
#         self.assertTrue(metric_1.calculate(W) == expected_res_1)
#
#         # 2D numpy array
#         metric_2 = tSumDistancesConnectedNeurons(positions.to_numpy())
#         expected_res_2 = 3 + 3 + 5 + 4
#         self.assertTrue(metric_2.calculate(W) == expected_res_2)
#
#         # Series
#         metric_3 = tSumDistancesConnectedNeurons(positions.x_pos)
#         expected_res_3 = 4 + 4
#         self.assertTrue(metric_3.calculate(W) == expected_res_3)
#
#         # 1D numpy array
#         metric_4 = tSumDistancesConnectedNeurons(positions.z_pos.to_numpy())
#         expected_res_4 = 0
#         self.assertTrue(metric_4.calculate(W) == expected_res_4)
#
# if __name__ == '__main__':
#     unittest.main()



positions = pd.DataFrame({"x_pos": [0, 0, 4], "y_pos": [0, 3, 0], "z_pos": [2, 2, 2]})
W = np.array([[0, 1, 0],
              [1, 0, 1],
              [1, 0, 0]])