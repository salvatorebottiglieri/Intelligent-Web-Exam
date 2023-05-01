import numpy as np
from src.Utils import read_dataset
from src.WeightsComputation import  decay_function, time_factor


dataset = read_dataset()

user1 = 1
user2 = 2
alpha = 0.4
d_alpha = 55
beta = 0.6
result = time_factor(d_alpha,alpha, beta, user1, user2,dataset)
print(result)