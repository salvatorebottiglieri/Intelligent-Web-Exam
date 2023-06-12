
import numpy as np
import pandas as pd
from definitions import ROOT_DIR

from src.Utils import get_users, read_dataset
from src.WeightsComputation import weight


"""
This class represents the graph of users' similarities in the datasets. It's implemented
through adjacency matrix. But, because of the dataset's large size, it use 16-bit floating
point values. The value of an edge between two users is the similarity between them. It encode
both the connections between users and the similarity between them: a value different from -inf
means that there is an edge between two users and the value of the edge is the similarity between.
"""
 

class SimilarityMatrix:
    def __init__(self, v: int,dataset_name:str) -> None:
        self.graph = np.full(
            shape=(v, v),
            fill_value=np.NINF,
            dtype=np.float16,
        )
        self.dataset_name = dataset_name


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
  

    def save(self) -> None:
        np.save(f"{ROOT_DIR}/output/similarity_matrix.npy", self.graph)

    def add_direct_similarities_for(self, active_user: int, alpha, eot, eor,dataset) -> None:
        users = get_users(dataset=dataset)
        for user in users:
            if user != active_user:
                self.add_edge(active_user, user,weight(user1=active_user,user2=user,
                                                       alpha=alpha,eot=eot,eor=eor,
                                                       dataset=dataset))

    def populate_matrix_from_dataset(self,alpha, eot, eor)-> None:
        dataset = read_dataset(dataset_name=self.dataset_name)
        users = get_users(dataset=dataset)

        for user in users:
            self.add_direct_similarities_for(user,alpha, eot, eor,dataset)

    
        
        
            





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
        self.users = list()
        self.items = list()
        self.mapper = {}

        
    def set_items(self, items: list):
        self.items = items
    
    def set_users(self,users:list):
        self.users = users

    def get_items(self):
        return self.items

    def add_value(self, user: int, item: int, value: np.float16):
        self.matrix[user-1][item-1] = value

    def get_value(self, user: int, item: int):
        return self.matrix[user-1][item-1]

    def get_items_rated_by(self, user: int):
        return self.matrix[user-1]

    def get_users_who_rated(self, item: int):
        return self.matrix[:, item-1]

    def print_matrix(self):
        print(self.matrix)

    def map_movie_id_to_index(self, elems: list) -> int:
        [self.add_to_mapper(elem, i) for i,elem in enumerate(elems)]

    def add_to_mapper(self,key,value):
        self.mapper[key] = value

    def populate_matrix(self, dataset):
        if len(self.items) == 0:
            self.set_items(dataset["movieId"].unique())
        self.map_movie_id_to_index(self.get_items())
        dataset.apply(lambda x: self.add_value(int(x["userId"]), int(self.mapper[x["movieId"]]), x["rating"]), axis=1)
