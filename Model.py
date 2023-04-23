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
            self.graph = np.full(shape=(self.first_half, self.second_half), fill_value=numpy.NINF, dtype=np.float16)
        else:
            self.graph = np.full(shape=(2, 2), fill_value=numpy.NINF, dtype=np.float16)
            self.first_half = self.second_half = 2

    def are_connected(self, a: int, b: int) -> bool:
        return self.graph[a][b] is not np.NINF

    def add_edge(self, s: int, d: int, value: np.float16) -> None:
        if self.are_connected(s, d):
            raise Exception(f"There is already an edge between {s} and {d}\n")

        if s <= self.first_half:
            self.graph[s][d] = value
        else:
            self.graph[d][s] = value

    def print_graph(self):
        print(self.graph)
