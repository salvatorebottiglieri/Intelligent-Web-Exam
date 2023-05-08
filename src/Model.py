import numpy
import numpy as np
from math import floor, ceil

"""
This class represents the graph of users' similarities in the datasets. It's implemented
through adjacency matrix. But, because of the dataset's large size, it use 16-bit floating
point values and the the number of rows (and also columns) is half of number of users.
"""


class SimilarityGraph:
    def __init__(self, v: int) -> None:
        if v != 2:
            self.first_half = floor(v / 2)
            self.second_half = ceil(v / 2)
            self.graph = np.full(
                shape=(self.first_half, self.second_half),
                fill_value=numpy.NINF,
                dtype=np.float16,
            )
        else:
            self.graph = np.full(shape=(2, 2), fill_value=numpy.NINF, dtype=np.float16)
            self.first_half = self.second_half = 2

    def are_connected(self, a: int, b: int) -> bool:
        if a <= self.first_half:
            return self.graph[a][b] is not np.NINF
        else:
            return self.graph[b][a] is not np.NINF

    def get_edge_value(self, a: int, b: int) -> np.float16:
        return self.graph[a][b]

    def add_edge(self, a: int, b: int, value: np.float16) -> None:
        if self.are_connected(a, b):
            raise Exception(f"There is already an edge between {a} and {b}\n")

        if a <= self.first_half:
            self.graph[a][b] = value
        else:
            self.graph[b][a] = value

    def print_graph(self):
        print(self.graph)


"""
This class represents the user-item matrix. It's implemented through a numpy array.
Because of the ratings' values are between 1 and 5, it uses 16-bit floating point values with
zeros as value for users that didn't rate an item.
"""


class UserItemMatrix:
    def __init__(self, first_dimension: int, second_dimension: int) -> None:
        self.first_dimension = first_dimension
        self.second_dimension = second_dimension
        self.matrix = np.zeros(
            shape=(first_dimension, second_dimension), dtype=np.float16
        )

    def add_value(self, first_index: int, second_index: int, value: np.float16):
        self.matrix[first_index][second_index] = value

    def get_value(self, first_index: int, second_index: int):
        return self.matrix[first_index][second_index]

    def get_row(self, first_index: int):
        return self.matrix[first_index]

    def get_column(self, second_index: int):
        return self.matrix[:, second_index]

    def print_matrix(self):
        print(self.matrix)
