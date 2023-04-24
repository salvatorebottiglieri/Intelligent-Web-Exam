import numpy as np
from Utils import read_dataset
from WeightsComputation import get_items_in_common

dataset = read_dataset()

print(dataset["rating"].describe())
np.savetxt("ratings.csv", dataset["rating"].unique(), delimiter=",")
#ciao


