import numpy as np
import pandas as pd
import os

def read_dataset(dataset_name="ml-latest", filename="ratings.csv") -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), "Datasets", dataset_name, filename)
    if not os.path.exists(file_path):
        raise Exception("Error, Path to file doesn't exists or file is missing")
    return pd.read_csv(file_path)

def generate_matrix(file_path, num_rows, num_cols):
    # Read in the CSV file in chunks
    chunksize = 10000
    reader = pd.read_csv(file_path, chunksize=chunksize)

    # Create empty sets for unique userIds and movieIds
    userIds = set()
    movieIds = set()

    # Iterate over each chunk of the CSV file
    for chunk in reader:
        # Add unique userIds and movieIds to their respective sets
        userIds.update(chunk['userId'].unique())
        movieIds.update(chunk['movieId'].unique())

    # Convert sets to sorted lists
    userIds = sorted(list(userIds))
    movieIds = sorted(list(movieIds))

    # Create empty DataFrame for new matrix
    newMatrix = pd.DataFrame(index=userIds[:num_rows], columns=movieIds[:num_cols])

    # Read in the CSV file again in chunks
    reader = pd.read_csv(file_path, chunksize=chunksize)

    # Iterate over each chunk of the CSV file
    for chunk in reader:
        # Iterate over each row in the chunk
        for index, row in chunk.iterrows():
            # Set the value of the new matrix at the intersection of userId and movieId to the rating
            if row['userId'] in userIds[:num_rows] and row['movieId'] in movieIds[:num_cols]:
                newMatrix.at[row['userId'], row['movieId']] = row['rating']

    # Save new matrix to CSV file
    newMatrix.to_csv('new_matrix.csv')

def rating_time(user:int,item:int,dataset:pd.DataFrame) -> int:
    """
    Returns the timestamp at which a user rated a particular item in the given dataset.

    :param user: An integer representing the user ID.
    :param item: An integer representing the item ID.
    :param dataset: A pandas DataFrame containing the user-item rating data.
    
    :return: An integer representing the timestamp of the user-item rating.
    """
    return dataset[dataset['userId'] == user][dataset['movieId'] == item]['timestamp'].values[0]


def minimum_timestamp(user:int, dataset:pd.DataFrame) -> int:
    """
    Returns the minimum timestamp associated with a given user ID in a pandas DataFrame.

    :param user: An integer representing the user ID.
    :param dataset: A pandas DataFrame containing timestamp and userId columns.
    :return: An integer representing the minimum timestamp associated with the given user ID.
    """
    timestamps = dataset[dataset['userId'] == user]['timestamp'].values
    return min(timestamps)

def max_timestamp(user:int, dataset:pd.DataFrame) -> int:
    """
    Returns the maximum timestamp from a given user in a given pandas dataframe.
    
    :param user: An integer representing the user id.
    :param dataset: A pandas dataframe containing the dataset.
    :return: An integer representing the maximum timestamp for the given user.
    """
    timestamps = dataset[dataset['userId'] == user]['timestamp'].values
    return max(timestamps)

def cut_time (min_time, max_time, d):
    return min_time + ((max_time - min_time) / d)


def get_users(dataset:pd.DataFrame) -> list:
    return dataset['userId'].unique().tolist()

def get_neighbours(user:int, dataset:pd.DataFrame) -> list:
    return dataset[dataset['userId'] == user]['movieId'].unique().tolist()