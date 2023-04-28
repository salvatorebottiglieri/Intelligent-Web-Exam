import numpy as np
from src.Utils import read_dataset
from src.WeightsComputation import  proportion, sopd


dataset = read_dataset()


movies_named_5 = dataset[dataset["movieId"] == 5]
# print(movies_named_5)


value = proportion(4, 8, dataset)

print(value)
