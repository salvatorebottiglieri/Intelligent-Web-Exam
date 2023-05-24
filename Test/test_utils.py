import pytest
from src.Utils import rating_time, read_dataset



def test_rating_time_should_return_correct_timestamp():
    user=1
    item=235
    expected = 964980908
    dataset = read_dataset(dataset_name="ml-latest-small", filename="ratings.csv")

    actual = rating_time(user, item,dataset)

    assert actual == expected
