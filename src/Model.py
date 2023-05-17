
import numpy as np

from src.Utils import read_dataset


"""
This class represents the graph of users' similarities in the datasets. It's implemented
through adjacency matrix. But, because of the dataset's large size, it use 16-bit floating
point values. The value of an edge between two users is the similarity between them. It encode
both the connections between users and the similarity between them: a value different from -inf
means that there is an edge between two users and the value of the edge is the similarity between.
"""
 

class SimilarityMatrix:
    def __init__(self, v: int) -> None:
        self.graph = np.full(
            shape=(v, v),
            fill_value=np.NINF,
            dtype=np.float16,
        )


    def are_connected(self, a: int, b: int) -> bool:
        return self.graph[a-1][b-1] != np.NINF


    def get_edge_value(self, a: int, b: int) -> np.float16:
        return self.graph[a-1][b-1]
    
    def get_neighbors(self, node: int) -> list:
        return [i+1 for i,x in enumerate(self.graph[node-1]) if x != np.NINF]
    
    def remove_edge(self, a: int, b: int) -> None:
        if not self.are_connected(a, b):
            raise Exception(f"Users {a} and {b} not have items in common\n")

        self.graph[a-1][b-1] = np.NINF
        self.graph[b-1][a-1] = np.NINF
    

    def add_edge(self, a: int, b: int, value: np.float16) -> None:
        self.graph[a-1][b-1] = value
        self.graph[b-1][a-1] = value    
  



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

    def add_value(self, user: int, item: int, value: np.float16):
        self.matrix[user][item] = value

    def get_value(self, user: int, item: int):
        return self.matrix[user][item]

    def get_items_rated_by(self, user: int):
        return self.matrix[user]

    def get_users_who_rated(self, item: int):
        return self.matrix[:, item]

    def print_matrix(self):
        print(self.matrix)
