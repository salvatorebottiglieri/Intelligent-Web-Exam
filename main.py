import numpy as np
from src.Utils import read_dataset
from src.WeightsComputation import  decay_function


dataset = read_dataset()

user1 = 4
user2 = 8
alpha = 1.5
print(decay_function(alpha, user1, user2, dataset))
