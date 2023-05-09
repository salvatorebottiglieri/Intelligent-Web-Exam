import numpy as np
from src.Model import SimilarityMatrix
from src.Utils import read_dataset
from src.WeightsComputation import  decay_function, time_factor


gr = SimilarityMatrix(10)
gr.add_edge(4, 8, 0.5)

print(gr.graph)

assert gr.get_neighbors(8) == [0.5]