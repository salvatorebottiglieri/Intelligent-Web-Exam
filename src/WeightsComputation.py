import numpy as np
import pandas as pd


def sortd(user1: int, user2: int, label: str, dataset: pd.DataFrame) -> float:
    items = get_items_in_common(user1, user2, dataset)
    if items.empty:
        return 0.0
    diffs = abs(items[f"{label}_x"].values - items[f"{label}_y"].values)
    return np.sum(diffs) / 2


def get_items_in_common(user1: int, user2: int, dataset: pd.DataFrame) -> pd.DataFrame:
    user1_rows = dataset[dataset["userId"] == user1]
    user2_rows = dataset[dataset["userId"] == user2]
    common_rows = pd.merge(user1_rows, user2_rows, how="inner", on=["movieId"])
    return common_rows


def sopd(user1: int, user2: int, dataset: pd.DataFrame) -> float:
    items = pd.merge(
        priority_list(user1, dataset),
        priority_list(user2, dataset),
        how="inner",
        on=["movieId"],
    )
    diffs = abs(items["priority_x"].values - items["priority_y"].values) 
    return np.sum(diffs) / 2


def priority_list(user: int, dataset: pd.DataFrame) -> pd.DataFrame:
    user_rows = dataset[dataset["userId"] == user]
    user_rows = user_rows.sort_values(by=["timestamp"], ascending=True)

    user_rows["priority"] = range(1, user_rows.shape[0] + 1)

    return user_rows[["movieId", "priority"]]


def mci(dataset: pd.DataFrame) -> int:
    items = dataset["movieId"].unique()
    return items.shape[0]


def proportion(user1: int, user2: int, dataset: pd.DataFrame) -> np.float16:
    items = get_items_in_common(user1, user2, dataset)
    if items.empty:
        return 0.0
    return items.shape[0] / mci(dataset)

    
def mratediff(user1: int, user2: int, n_star: int, dataset: pd.DataFrame) -> np.float16:
    return (n_star-1) * proportion(user1, user2, dataset)
    


