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

