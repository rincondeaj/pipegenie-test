import pytest
import numpy as np
from pipegenie.metrics._classification import accuracy_score, balanced_accuracy_score, roc_auc_score

def test_accuracy_score_perfect():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]

    assert accuracy_score(y_true, y_pred) == 1.0

def test_accuracy_score_partial():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 0]

    assert accuracy_score(y_true, y_pred) == 0.75

def test_accuracy_score_empty():
    y_true = []
    y_pred = []

    assert np.isnan(accuracy_score(y_true, y_pred)) or accuracy_score(y_true, y_pred) == 0.0

def test_balanced_accuracy_perfect():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    assert balanced_accuracy_score(y_true, y_pred) == 1.0

def test_balanced_accuracy_imbalanced():
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]

    np.testing.assert_almost_equal(balanced_accuracy_score(y_true, y_pred), 0.8333333, decimal=6)

def test_roc_auc_simple():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]

    np.testing.assert_almost_equal(roc_auc_score(y_true, y_score), 1.0, decimal=6)

def test_roc_auc_partial():
    y_true = [0, 1, 0, 1, 0, 0]
    y_score = [0.1, 0.9, 0.2, 0.3, 0.8, 0.4]

    np.testing.assert_almost_equal(roc_auc_score(y_true, y_score), 0.75, decimal=6)

def test_roc_auc_all_same_score():
    y_true = [0, 1, 0, 1]
    y_score = [0.95, 0.95, 0.95, 0.95]

    np.testing.assert_almost_equal(roc_auc_score(y_true, y_score), 0.5, decimal=6)

def test_input_types():
    y_true_list = [0, 1, 0]
    y_pred_list = [0, 0, 0]
    y_true_array = np.array(y_true_list)
    y_pred_array = np.array(y_pred_list)
    
    assert accuracy_score(y_true_list, y_pred_list) == accuracy_score(y_true_array, y_pred_array)
    assert balanced_accuracy_score(y_true_list, y_pred_list) == balanced_accuracy_score(y_true_array, y_pred_array)
