import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def k_fold_cross_validation(k:int, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    K-cross validation is a function that performs K-fold cross validation.
    
    :param k: The number of folds
    :param dataset: The dataset to be used for the computation
    :return: The K-fold cross validation
    """
    pass


def split_dataset_in_k_fold(k:int, dataset: pd.DataFrame):
    return KFold(n_splits=k, shuffle=True, random_state=5830382).split(dataset)


def compute_error(true_ratings, predicted_ratings):
    return mean_absolute_error(true_ratings, predicted_ratings)