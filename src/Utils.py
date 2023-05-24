import numpy as np
import pandas as pd
import os


def read_dataset(dataset_name="ml-latest", filename="ratings.csv") -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), "Datasets", dataset_name, filename)
    if not os.path.exists(file_path):
        raise Exception("Error, Path to file doesn't exists or file is missing")
    return pd.read_csv(file_path)

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