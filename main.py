import numpy as np
from src.Utils import read_dataset
from src.WeightsComputation import  mratediff, n_sord, proportion, sopd


dataset = read_dataset()




value = n_sord(4, 8, dataset)

print(value)
