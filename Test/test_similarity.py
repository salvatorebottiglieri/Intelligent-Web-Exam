import pytest

from src.Utils import read_dataset
from src.WeightsComputation import (
    mci,
    mratediff,
    n_sord,
    proportion,
    sopd,
    sortd,
    get_items_in_common,
    decay_function,
)


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


def test_should_proportion_return_float_greater_than_zero(dataset):
    user1 = 4
    user2 = 8
    result = proportion(user1, user2, dataset)
    assert isinstance(result, float) is True
    assert result > 0.0


def test_should_proportion_return_zero_when_users_not_have_items_in_common(dataset):
    user1 = 1
    user2 = 2
    result = proportion(user1, user2, dataset)
    assert isinstance(result, float) is True
    assert result == 0.0


def test_should_mratediff_return_zero_when_users_not_have_items_in_common(dataset):
    user1 = 1
    user2 = 2
    n_star = 5
    result = mratediff(user1, user2, n_star, dataset)
    assert result == 0.0


def test_should_mratediff_return_float_greater_than_zero(dataset):
    user1 = 4
    user2 = 8
    n_star = 5
    result = mratediff(user1, user2, n_star, dataset)
    assert result > 0.0


def test_should_n_sord_return_float_greater_than_zero(dataset):
    user1 = 4
    user2 = 8
    result = n_sord(user1, user2, dataset)
    assert isinstance(result, float) is True
    assert result > 0.0


def test_should_mci_return_right_number(dataset):
    user1 = 4
    user2 = 8
    result = mci(user1, user2, dataset)
    assert isinstance(result, int) is True
    assert result == 31


def test_should_decay_function_raise_error_when_alpha_is_less_than_equal_one(dataset):
    user1 = 4
    user2 = 8
    alpha = 0.5
    with pytest.raises(Exception) as ex:
        decay_function(alpha, user1, user2, dataset)
        assert str(ex.value) == "Alpha must be greater than 1"


def test_should_decay_function_return_float(dataset):
    user1 = 4
    user2 = 8
    alpha = 1.5
    result = decay_function(alpha, user1, user2, dataset)
    assert isinstance(result, float) is True
    assert result > 0.0
