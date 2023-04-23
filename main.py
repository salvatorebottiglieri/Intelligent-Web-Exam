from Model import Graph
from Utils import read_dataset
from WeightsComputation import get_items_in_common

dataset = read_dataset()

#print(dataset["rating"].describe())

#dataframe = get_items_in_common(55,54)
users_cadinality =len(dataset["userId"].unique())
g = Graph(users_cadinality)
g.print_graph()

