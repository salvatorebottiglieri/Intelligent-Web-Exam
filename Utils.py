import pandas as pd
import os

def read_dataset(dataset_name="ml-latest", filename="ratings.csv") -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(),"Datasets", dataset_name, filename)
    if not os.path.exists(file_path):
        raise Exception("Error, Path to file doesn't exists or file is missing")
    return pd.read_csv(file_path)
