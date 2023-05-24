from math import e
import pandas as pd
from src.Utils import max_timestamp, minimum_timestamp, rating_time

def time_impact(user:int, item:int, dataset:pd.Dataframe)-> float:
    e * ( (minimum_timestamp(user, dataset) + rating_time(user, item, dataset)) / (max_timestamp(user, dataset) - minimum_timestamp(user, dataset)) )