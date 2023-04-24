import numpy as np
from Utils import read_dataset
from WeightsComputation import get_items_in_common, priority_list


dataset = read_dataset()

print(dataset["rating"])
#np.savetxt("ratings.csv", dataset["rating"].unique(), delimiter=",")
priority_list(1)


