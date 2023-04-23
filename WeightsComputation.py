import pandas as pd

def SORD(user1, user2):
    items = get_items_in_common(user1, user2)

    diffs = []
    for i, user1 in enumerate(lst):
        for j, user2 in enumerate(lst):
            if i != j:
                diffs.append(abs(user1 - user2))
    return float(sum(diffs) / 2)


def SOTD(user1, user2):
    lst = get_items_in_common(user1, user2)
    diffs = []
    for i, user1 in enumerate(lst):
        for j, user2 in enumerate(lst):
            if i != j:
                diffs.append(abs(user1 - user2))
    return float(sum(diffs) / 2)


def SOPD(user1, user2):
    lst = get_items_in_common(user1, user2)
    diffs = []
    for i, user1 in enumerate(lst):
        for j, user2 in enumerate(lst):
            if i != j:
                diffs.append(abs(user1 - user2))
    return float(sum(diffs) / 2)


def get_items_in_common(user1, user2, dataset):
    user1_rows = dataset[dataset["userId"] == user1]
    user2_rows = dataset[dataset["userId"] == user2]
    common_rows = pd.merge(user1_rows, user2_rows, how="inner", on=["movieId"])
    return common_rows
