import math
import numpy as np
import pytest
from src.Model import SimilarityMatrix
from src.Utils import read_dataset
from src.WeightsComputation import (
    compute_similarity,
    mci,
    mratediff,
    n_sord,
    proportion,
    sopd,
    sortd,
    get_items_in_common,
    decay_function,
    time_factor,
    base_weight,
    weight,
)
    
    
@pytest.fixture(scope = "module")
def similarity_graph():
    sm =  SimilarityMatrix(5)
    sm.add_edge(1, 2, 0.9)
    sm.add_edge(1, 4, 0.2)
    sm.add_edge(2, 3, 0.5)
    sm.add_edge(3, 4, 0.6)
    sm.add_edge(3, 5, 0.3)
    yield sm

    del sm
    

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
    assert math.isclose(result,0.0)


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
    assert math.isclose(result,0.0)


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
    assert math.isclose(result,0.0)


def test_should_mratediff_return_zero_when_users_not_have_items_in_common(dataset):
    user1 = 1
    user2 = 2
    n_star = 5
    result = mratediff(user1, user2, n_star, dataset)
    assert math.isclose(result,0.0)


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


def test_should_decay_function_return_one_when_users_not_have_items_in_common(dataset):
    user1 = 1
    user2 = 2
    alpha = 1.5
    result = decay_function(alpha, user1, user2, dataset)
    assert isinstance(result, float) is True
    assert math.isclose(result,1.0)


def test_should_time_factor_function_raise_exception_if_alpha_plus_beta_not_equal_one(
    dataset,
):
    user1 = 4
    user2 = 8
    alpha = 0.5
    beta = 1.5
    d_alpha = 1.5
    with pytest.raises(Exception) as ex:
        time_factor(d_alpha, alpha, user1, user2, dataset)
        assert str(ex.value) == "Alpha plus Beta must be equal to 1"


def test_should_time_factor_function_return_float(dataset):
    user1 = 4
    user2 = 8
    alpha = 0.5
    d_alpha = 1.5
    beta = 0.5
    result = time_factor(d_alpha, alpha, user1, user2, dataset)
    print(result)
    assert isinstance(result, float) is True
    assert result > 0.0


def test_should_base_weight_return_zero_when_users_not_have_items_in_common(dataset):
    user1 = 1
    user2 = 2
    eor = 1
    result = base_weight(eor, user1, user2, dataset)
    assert isinstance(result, float) is True
    assert math.isclose(result,0.0)


def test_should_base_weight_return_value_between_zero_and_one_when_users_have_at_least_one_item_in_common(
    dataset
):
    user1 = 4
    user2 = 8
    eor = 1.0
    result = base_weight(eor, user1, user2, dataset)
    assert isinstance(result, float) is True
    assert result > 0.0
    assert result < 1.0

def test_should_weight_return_value_between_zero_and_one_when_users_have_at_least_one_item_in_common(
        dataset
):
    user1 = 4
    user2 = 8
    eot = 1.0
    result = weight(user1=user1, user2=user2,eot=eot, dataset=dataset)
    assert isinstance(result, float) is True
    assert result > 0.0
    assert result < 1.0





