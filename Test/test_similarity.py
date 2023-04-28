import pytest

from src.Utils import read_dataset
from src.WeightsComputation import sopd, sortd, get_items_in_common


@pytest.fixture(scope="module")
def dataset():
    dataset = read_dataset()
    yield dataset
    del dataset


def test_should_get_items_in_common_return_nothing(dataset):
    userId1 = 1
    userId2 = 2
    dataframe = get_items_in_common(userId1, userId2, dataset)

    assert dataframe.empty is True


def test_should_sortd_return_a_float_greater_than_zero(dataset):
    user1 = 4
    user2 = 8
    label = "rating"
    result = sortd(user1, user2, label, dataset)
    assert isinstance(result, float) is True
    assert result > 0.0


def test_should_sortd_return_zero_when_users_not_have_items_in_common(dataset):
    user1 = 1
    user2 = 2
    label = "rating"
    result = sortd(user1, user2, label, dataset)
    assert isinstance(result, float) is True
    assert result == 0.0


def test_should_sopd_return_float_greater_than_zero(dataset):
    user1 = 4
    user2 = 8
    result = sopd(user1, user2, dataset)
    assert isinstance(result, float) is True
    assert result > 0.0


def test_should_sopd_return_zero_when_users_not_have_items_in_common(dataset):
    user1 = 1
    user2 = 2
    result = sopd(user1, user2, dataset)
    assert isinstance(result, float) is True
    assert result == 0.0
