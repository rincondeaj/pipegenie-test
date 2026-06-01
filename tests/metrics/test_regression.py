import pytest
import numpy as np
from pipegenie.metrics._regression import mean_squared_error, root_mean_squared_error, r2_score

def test_mse_perfect_prediction():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert mean_squared_error(y_true, y_pred) == 0.0

def test_mse_partial_error():
    y_true = [1, 2, 3]
    y_pred = [2, 2, 4]
    np.testing.assert_almost_equal(mean_squared_error(y_true, y_pred), 0.6666667, decimal=6)

def test_mse_empty_input():
    y_true = []
    y_pred = []
    assert np.isnan(mean_squared_error(y_true, y_pred))

def test_rmse_perfect_prediction():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert root_mean_squared_error(y_true, y_pred) == 0.0

def test_rmse_partial_error():
    y_true = [1, 2, 3]
    y_pred = [2, 2, 4]
    np.testing.assert_almost_equal(root_mean_squared_error(y_true, y_pred), 0.81649658, decimal=6)

def test_r2_perfect_prediction():
    y_true = [3, -0.5, 2, 7]
    y_pred = [3, -0.5, 2, 7]
    assert r2_score(y_true, y_pred) == 1.0

def test_r2_partial_prediction():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    np.testing.assert_almost_equal(r2_score(y_true, y_pred), 0.9486081370449679, decimal=6)

def test_r2_worse_than_mean():
    y_true = [3, -0.5, 2, 7]
    y_pred = [4, 4, 4, 4]
    np.testing.assert_almost_equal(r2_score(y_true, y_pred), -0.17344753747323338, decimal=6)

def test_r2_zero_variance():
    y_true = [5, 5, 5]
    y_pred = [5, 5, 5]
    result = r2_score(y_true, y_pred)
    assert np.isnan(result) or result == 1.0 
