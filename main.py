import numpy as np
from src.Utils import read_dataset
from src.WeightsComputation import  sopd


dataset = read_dataset()


movies_named_5 = dataset[dataset["movieId"] == 5]
# print(movies_named_5)


print(sopd(1, 2, dataset))
