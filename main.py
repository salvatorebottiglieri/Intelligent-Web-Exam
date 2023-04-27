import numpy as np
from src.Utils import read_dataset
from src.WeightsComputation import SORTD, get_items_in_common, priority_list


dataset = read_dataset()


movies_named_5 = dataset[dataset["movieId"] == 5]
# print(movies_named_5)


print(SORTD(4, 8, "rating"))
