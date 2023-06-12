import numpy as np
import pandas as pd
from queue import Queue


def sortd(user1: int, user2: int, label: str, dataset: pd.DataFrame) -> float:
    """
    Sortd is a function that computes the sum of the absolute differences between
    the ratings (or timestamps) of the items in common between two users.

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


def proportion(user1: int, user2: int, dataset: pd.DataFrame) -> float:
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


def mratediff(user1: int, user2: int, n_star: int, dataset: pd.DataFrame) -> float:
    '''
    Mratediff is a function that computes the maximum rating difference between two users.

    :param user1: The first user
    :param user2: The second user
    :param n_star: The maximum rating
    :param dataset: The dataset to be used for the computation
    :return: The maximum rating difference between two users.
    '''
    return (n_star - 1) * get_items_in_common(user1, user2, dataset).shape[0]


def n_sord(user1: int, user2: int, dataset: pd.DataFrame) -> float:
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


def m_sopd(items_in_common: int) -> float:
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


def n_sopd(user1: int, user2: int, dataset: pd.DataFrame) -> float:
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
) -> float:
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
    try:
        sortd_fraction = sortd(user1, user2, "timestamp", dataset) / get_items_in_common(user1, user2, dataset).shape[0]
    except ZeroDivisionError:
        sortd_fraction = 0
    finally:         
        denominator = alpha + sortd_fraction
        return alpha / denominator

def time_factor(d_alpha:float,t_alpha:float,user1:int,user2:int,dataset:pd.DataFrame) -> float:
    '''
    Time_factor is a function that computes the time factor between two users.

    :param d_alpha: The alpha parameter for decay function
    :param t_alpha: The alpha parameter for time factor
    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The time factor between two users.
    '''
    beta = 1 - t_alpha
    
    first_addend = t_alpha * decay_function(d_alpha, user1, user2, dataset)
    second_addend = beta * (1- n_sord(user1, user2, dataset))

    return first_addend + second_addend

def base_weight(eor:float,user1:int,user2:int,dataset:pd.DataFrame) -> float:
    '''
    Base_weight is a function that computes the similarity of two users that have at least
    one item in common, ignoring time factor.

    :param eor: says how much the user rating affects the base weight
    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The base weight between two users.
    '''
    try:
        result =  proportion(user1, user2, dataset) * (1 - (sortd(user1, user2, "rating", dataset) / mratediff(user1, user2, 5, dataset))**(1/eor) )
    except ZeroDivisionError:
        result = 0.0

    return result

def weight(user1: int, user2:int, dataset:pd.DataFrame, eor=6.0, eot=0.7, alpha=5.0) -> float:
    '''
    weight is a function that computes the similarity of two users that have at least
    one item in common, considering time factor.

    :param eor: says how much the rating factor affects the weight
    :param eot: says how much the time factor affects the weight
    :param user1: The first user
    :param user2: The second user
    :param dataset: The dataset to be used for the computation
    :return: The weight between two users.
    '''
    
    return base_weight(eor, user1, user2, dataset) * (eot + ((1 - eot) * time_factor(2, alpha, user1, user2, dataset)) )



'''
Rispetto all'implementazione fornita dallo pseudocodice all'interno del paper, 
questa versione è stata modificata per ovviare ad un errore concettualmente presente
nella prima. Nello pseudocodice, infatti, il controllo che permette di aggiungere
un nodo alla coda Qa viene effettuato all'interno primo if, il quale, durante la prima iterazione,
non viene mai eseguito. Questo comporta che i nodi collegati direttamente al nodo attivo non vengano
mai aggiunti alla coda Qa, e quindi non vengano mai esplorati. Per ovviare a questo problema,
il controllo è stato spostato fuori dal primo if, in modo tale che venga eseguito ad ogni iterazione.

Inoltre, tale versione deve essere richiamata su un grafo in cui siano già stati calcolati i vicini diretti,
per questioni di ottimizzazioni, abbiamo deciso di fare quest'operazione una sola volta e non ogni volta che
l'algoritmo veniva richiamato, cosa che nel paper di riferimento viene però fatta.
'''

def compute_similarity(user_item_graph, active_user, mu):
    Qa = Queue()
    Qb = Queue()
    Qa.put(active_user)
    MaxSim = {active_user: 0}
    similarities = {}

    while not Qa.empty():
        C = Qa.get()
        Qb.put(C)

        for Dn in user_item_graph.get_neighbors(C):
            if user_item_graph.get_edge_value(Dn, active_user) < (
                user_item_graph.get_edge_value(C, active_user) * user_item_graph.get_edge_value(C, Dn)
            ):
                user_item_graph.add_edge(
                    Dn,
                    active_user,
                    user_item_graph.get_edge_value(C, active_user) * user_item_graph.get_edge_value(C, Dn),
                )
                MaxSim[active_user] = (
                    user_item_graph.get_edge_value(C, active_user) * user_item_graph.get_edge_value(C, Dn)
                )
            if (
                Dn not in Qb.queue
                and Dn not in Qa.queue
                and user_item_graph.get_edge_value(Dn, active_user) > MaxSim[active_user] / mu
                ):
                    Qa.put(Dn)

        similarities[C] = user_item_graph.get_edge_value(C, active_user)

    return similarities