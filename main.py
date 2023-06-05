import numpy as np

# Define a dictionary to map IDs to integers
id_to_int = {}

# Define a list of IDs
ids = [111, 112, 113, 114, 115]

# Map each ID to a sequential integer
for i, id in enumerate(ids):
    id_to_int[id] = i

# Define a matrix with columns indexed by IDs
matrix = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15]])

# Index the matrix using the mapped integers
col_indices = [id_to_int[113], id_to_int[115], id_to_int[111]]
cols = matrix[:, col_indices]

# Print the indexed columns of the matrix
print(cols)
from src.Model import SimilarityMatrix, UserItemMatrix
from src.Utils import read_dataset


dataset = read_dataset(dataset_name="ml-latest-small")

users = dataset["userId"].unique()
items = dataset["movieId"].unique()

user_item_matrix = UserItemMatrix(len(users),len(items))

user_item_matrix.populate_matrix(dataset)

user_item_matrix.print_matrix()

# Iper-parametri
EOR = 6
alpha = 5
EOT = 0.7

