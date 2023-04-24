import pandas as pd

from Utils import read_dataset

def SORTD(user1,user2,label):
    items = get_items_in_common(user1, user2)
    diffs = []
    diffs.append(abs(items[f"{label}_x"]- items[f"{label}_y"]))
    return float(sum(diffs) / 2)

def priority_list(user):
    dataset = read_dataset()
    user_rows = dataset[dataset["userId"] == user]
    user_rows = user_rows.sort_values(by=["timestamp"], ascending=True)

    user_rows["priority"] = range(1, user_rows.shape[0] + 1)



    return user_rows["movieId"].tolist()

def SOPD(user1,user2):
    raise NotImplementedError

def get_items_in_common(user1, user2):
    dataset = read_dataset()
    user1_rows = dataset[dataset["userId"] == user1]
    user2_rows = dataset[dataset["userId"] == user2]
    common_rows = pd.merge(user1_rows, user2_rows, how="inner", on=["movieId"])
    return common_rows
