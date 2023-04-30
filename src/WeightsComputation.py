import numpy as np
import pandas as pd


def sortd(user1: int, user2: int, label: str, dataset: pd.DataFrame) -> float:
    """
    Sortd is a function that computes the sum of the absolute differences between
    the ratings of the items in common between two users.

    :param user1: The first user
    :param user2: The second user
    :param label: The label of the column to be used for the computation
    :param dataset: The dataset to be used for the computation
    :return: The sum of the absolute differences between the ratings of the items in common between two users.
    """
    items = get_items_in_common(user1, user2, dataset)
    if items.empty:
        return 0.0
    diffs = abs(items[f"{label}_x"].values - items[f"{label}_y"].values)
    return np.sum(diffs) / 2


def get_items_in_common(user1: int, user2: int, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Get_items_in_common is a function that returns the items in common between two users.

    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The items in common between two users.

    """
    user1_rows = dataset[dataset["userId"] == user1]
    user2_rows = dataset[dataset["userId"] == user2]
    common_rows = pd.merge(user1_rows, user2_rows, how="inner", on=["movieId"])
    return common_rows


def sopd(user1: int, user2: int, dataset: pd.DataFrame) -> float:
    """
    Sopd is a function that computes the sum of the absolute differences between
    the priorities of the items in common between two users.

    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The sum of the absolute differences between the priorities of the items in common between two users.

    """
    items = pd.merge(
        priority_list(user1, dataset),
        priority_list(user2, dataset),
        how="inner",
        on=["movieId"],
    )
    diffs = abs(items["priority_x"].values - items["priority_y"].values)
    return np.sum(diffs) / 2


def priority_list(user: int, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Priority_list is a function that returns the priority list of a user.

    :param user: The user
    :param dataset: The dataset to be used for the computation
    :return: The priority list of a user.
    """
    user_rows = dataset[dataset["userId"] == user]
    user_rows = user_rows.sort_values(by=["timestamp"], ascending=True)

    user_rows["priority"] = range(1, user_rows.shape[0] + 1)

    return user_rows[["movieId", "priority"]]


def mci(user1: int, user2: int, dataset: pd.DataFrame) -> int:
    """
    MCI is a function that computes the minimum number of items in common between two users.

    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The minimum number of items in common between two users.
    """
    n_user1_items = dataset[dataset["userId"] == user1]["movieId"].shape[0]
    n_user2_items = dataset[dataset["userId"] == user2]["movieId"].shape[0]

    if n_user1_items < n_user2_items:
        return n_user1_items
    return n_user2_items


def proportion(user1: int, user2: int, dataset: pd.DataFrame) -> np.float16:
    '''
    Proportion is a function that computes the proportion of items in common between two users.

    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The proportion of items in common between two users.
    '''
    items = get_items_in_common(user1, user2, dataset)
    if items.empty:
        return 0.0
    return items.shape[0] / mci(user1, user2, dataset)


def mratediff(user1: int, user2: int, n_star: int, dataset: pd.DataFrame) -> np.float16:
    '''
    Mratediff is a function that computes the maximum rating difference between two users.

    :param user1: The first user
    :param user2: The second user
    :param n_star: The maximum rating
    :param dataset: The dataset to be used for the computation
    :return: The maximum rating difference between two users.
    '''
    return (n_star - 1) * get_items_in_common(user1, user2, dataset).shape[0]


def n_sord(user1: int, user2: int, dataset: pd.DataFrame) -> np.float16:
    '''
    N_sord is a function that computes the normalized sortd between two users.

    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The normalized sortd between two users.
    '''
    return sortd(user1, user2, "rating", dataset) / mratediff(
        user1, user2, dataset["rating"].max(), dataset
    )


def m_sopd(items_in_common: int) -> np.float16:
    '''
    M_sopd is a function that computes the maximum sum of the absolute 
    differences between the priorities of the items in common between two users.

    :param items_in_common: The number of items in common between two users
    :return: The maximum sum of the absolute differences between the priorities of the items in common between two users.
    '''
    if items_in_common % 2 == 0:
        result = (items_in_common / 2) ** 2
    else:
        result = items_in_common ** 2 - items_in_common - 2 * (items_in_common / 2) ** 2
    return result


def n_sopd(user1: int, user2: int, dataset: pd.DataFrame) -> np.float16:
    '''
    N_sopd is a function that computes the normalized sopd between two users.

    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The normalized sopd between two users.
    '''
    return sopd(user1, user2, dataset) / m_sopd(mci(user1, user2, dataset))


def decay_function(
    alpha: float, user1: int, user2: int, dataset: pd.DataFrame
) -> np.float16:
    '''
    Decay_function is a function that computes the decay function between two users.
    
    :param alpha: The alpha parameter
    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The decay function between two users.
    '''
    if alpha <= 1:
        raise Exception("Alpha must be greater than 1")
    return alpha / alpha + (
        sortd(user1, user2, "rating", dataset)
        / get_items_in_common(user1, user2, dataset).shape[0]
    )
