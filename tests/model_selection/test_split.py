import pytest
from pipegenie.model_selection import KFold, StratifiedKFold, BootstrapSplit, train_test_split
import numpy as np
import math
from unittest.mock import patch

# pylint: disable=redefined-outer-name

def _calculate_thresholds(n_samples, y_counts, n_folds):
    group_size = n_samples / n_folds
    thresholds = {}
    
    for i, count in enumerate(y_counts):
        p_target = count / n_samples
        sigma_target = math.sqrt(p_target * (1 - p_target) / group_size)
        percentage_threshold = sigma_target * 100
        absolute_deviation = sigma_target * group_size
        thresholds[i] = (percentage_threshold, absolute_deviation)

    return thresholds

@pytest.fixture
def setup_kfold():
    X = np.arange(50).reshape((25, 2))
    y = np.arange(50)
    n_splits = 5
    return X, y, n_splits

def test_kfold_fold_sizes(setup_kfold):
    X, y, n_splits = setup_kfold
    kfold = KFold(n_splits=n_splits)
    assert kfold.n_splits == n_splits
    assert len(list(kfold.split(X))) == n_splits

    for train_indices, test_indices in kfold.split(X):
        assert len(train_indices) == 20
        assert len(test_indices) == 5

def test_kfold_shuffle(setup_kfold):
    X, y, n_splits = setup_kfold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    first_run = list(kfold.split(X))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    second_run = list(kfold.split(X))
    
    # same seed
    for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, second_run):
        assert np.array_equal(train_indices1, train_indices2)
        assert np.array_equal(test_indices1, test_indices2)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    third_run = list(kfold.split(X))
    
    # different seeds
    for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, third_run):
        assert not np.array_equal(train_indices1, train_indices2)
        assert not np.array_equal(test_indices1, test_indices2)

@pytest.fixture
def setup_stratifiedkfold():
    X = np.arange(30).reshape((15, 2))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    n_splits = 3
    return X, y, n_splits

def test_stratifiedkfold_fold_sizes(setup_stratifiedkfold):
    X, y, n_splits = setup_stratifiedkfold
    skfold = StratifiedKFold(n_splits=n_splits)
    assert skfold.n_splits == n_splits
    assert len(list(skfold.split(X, y))) == n_splits

    for train_indices, test_indices in skfold.split(X, y):
        assert len(train_indices) == 10
        assert len(test_indices) == 5

def test_stratifiedkfold_shuffle(setup_stratifiedkfold):
    X, y, n_splits = setup_stratifiedkfold
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    first_run = list(skfold.split(X, y))
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    second_run = list(skfold.split(X, y))
    
    # same seed
    for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, second_run):
        assert np.array_equal(train_indices1, train_indices2)
        assert np.array_equal(test_indices1, test_indices2)

    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    third_run = list(skfold.split(X, y))
    
    # different seeds
    for (train_indices1, test_indices1), (train_indices2, test_indices2) in zip(first_run, third_run):
        assert not np.array_equal(train_indices1, train_indices2)
        assert not np.array_equal(test_indices1, test_indices2)

def test_stratifiedkfold_stratification(setup_stratifiedkfold):
    X, y, n_splits = setup_stratifiedkfold
    skfold = StratifiedKFold(n_splits=n_splits)
    y_counts = np.bincount(y)
    y_counts = np.sort(y_counts)
    class_samples_per_test_fold = np.round(y_counts / n_splits).astype(int)
    thresholds = _calculate_thresholds(len(y), y_counts, n_splits)
    
    for train_indices, test_indices in skfold.split(X, y):
        train_y = y[train_indices]
        test_y = y[test_indices]
        train_y_counts = np.bincount(train_y)
        train_y_counts = np.sort(train_y_counts)
        test_y_counts = np.bincount(test_y)
        test_y_counts = np.sort(test_y_counts)

        for i, (train_count, test_count) in enumerate(zip(train_y_counts, test_y_counts)):
            test_class_samples = class_samples_per_test_fold[i]
            train_class_samples = test_class_samples * (n_splits - 1)
            threshold = thresholds[i][1]
            assert abs(train_count - train_class_samples) <= threshold
            assert abs(test_count - test_class_samples) <= threshold

def test_stratifiedkfold_stratification_shuffle(setup_stratifiedkfold):
    X, y, n_splits = setup_stratifiedkfold
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    y_counts = np.bincount(y)
    y_counts = np.sort(y_counts)
    class_samples_per_test_fold = np.round(y_counts / n_splits).astype(int)
    thresholds = _calculate_thresholds(len(y), y_counts, n_splits)
    
    for train_indices, test_indices in skfold.split(X, y):
        train_y = y[train_indices]
        test_y = y[test_indices]
        train_y_counts = np.bincount(train_y)
        train_y_counts = np.sort(train_y_counts)
        test_y_counts = np.bincount(test_y)
        test_y_counts = np.sort(test_y_counts)

        for i, (train_count, test_count) in enumerate(zip(train_y_counts, test_y_counts)):
            test_class_samples = class_samples_per_test_fold[i]
            train_class_samples = test_class_samples * (n_splits - 1)
            threshold = thresholds[i][1]
            assert abs(train_count - train_class_samples) <= threshold
            assert abs(test_count - test_class_samples) <= threshold

class MockRandomState:
    def __init__(self, seed=None):
        pass
    def shuffle(self, x):
        x[:] = x[::-1]

@pytest.fixture
def setup_testtrainsplit():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 0, 1, 1])
    return X, y

def test_testtrainsplit_split_sizes(setup_testtrainsplit):
    X, y = setup_testtrainsplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    assert len(X_train) == 4
    assert len(X_test) == 2
    assert len(y_train) == 4
    assert len(y_test) == 2

def test_testtrainsplit_no_shuffle(setup_testtrainsplit):
    X, y = setup_testtrainsplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    assert np.array_equal(X_train, [[1, 2], [3, 4], [5, 6], [7, 8]])
    assert np.array_equal(X_test, [[9, 10], [11, 12]])
    assert np.array_equal(y_train, [0, 0, 0, 0])
    assert np.array_equal(y_test, [1, 1])

def test_testtrainsplit_shuffle(setup_testtrainsplit):
    X, y = setup_testtrainsplit
    
    # We cannot use the shuffle parameter directly as RandomState is inmutable
    # We need to mock the RandomState class to simulate the shuffle
    with patch('numpy.random.RandomState', MockRandomState):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True)
        assert np.array_equal(X_train, [[11, 12], [9, 10], [7, 8], [5, 6]])
        assert np.array_equal(X_test, [[3, 4], [1, 2]])
        assert np.array_equal(y_train, [1, 1, 0, 0])
        assert np.array_equal(y_test, [0, 0])

def test_testtrainsplit_stratify_no_shuffle(setup_testtrainsplit):
    X, y = setup_testtrainsplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0, shuffle=False)
    assert np.array_equal(X_train, [[1, 2], [3, 4], [5, 6], [9, 10]])
    assert np.array_equal(X_test, [[7, 8], [11, 12]])
    assert np.array_equal(y_train, [0, 0, 0, 1])
    assert np.array_equal(y_test, [0, 1])

def test_testtrainsplit_stratify_shuffle(setup_testtrainsplit):
    X, y = setup_testtrainsplit
    
    # We cannot use the shuffle parameter directly as RandomState is inmutable
    # We need to mock the RandomState class to simulate
    with patch('numpy.random.RandomState', MockRandomState):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=None, shuffle=True)
        assert np.array_equal(X_train, [[11, 12], [3, 4], [5, 6], [7, 8]])
        assert np.array_equal(X_test, [[9, 10], [1, 2]])
        assert np.array_equal(y_train, [1, 0, 0, 0])
        assert np.array_equal(y_test, [1, 0])

def test_testtrainsplit_invalid_test_size(setup_testtrainsplit):
    X, y = setup_testtrainsplit
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=0, stratify=y)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1, stratify=y)

def test_testtrainsplit_reproducibility(setup_testtrainsplit):
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.25, random_state=0)
    assert np.array_equal(X_train, X_train2)
    assert np.array_equal(X_test, X_test2)
    assert np.array_equal(y_train, y_train2)
    assert np.array_equal(y_test, y_test2)

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.25, random_state=1)
    assert not np.array_equal(X_train, X_train3)
    assert not np.array_equal(X_test, X_test3)
    assert not np.array_equal(y_train, y_train3)
    assert not np.array_equal(y_test, y_test3)

def test_testtrainsplit_stratify_reproducibility(setup_testtrainsplit):
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    assert np.array_equal(X_train, X_train2)
    assert np.array_equal(X_test, X_test2)
    assert np.array_equal(y_train, y_train2)
    assert np.array_equal(y_test, y_test2)

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    assert not np.array_equal(X_train, X_train3)
    assert not np.array_equal(X_test, X_test3)
    assert not np.array_equal(y_train, y_train3)
    assert not np.array_equal(y_test, y_test3)

@pytest.fixture
def X():
    return np.arange(20).reshape(10, 2)


def test_BootstrapSplit_number_of_splits(X):
    splitter = BootstrapSplit(n_splits=5, random_state=0)
    splits = list(splitter.split(X))

    assert len(splits) <= 5   # some splits may be skipped if no OOB


def test_BootstrapSplit_train_size_equals_n_samples(X):
    splitter = BootstrapSplit(n_splits=3, random_state=0)

    for train_idx, val_idx in splitter.split(X):
        assert len(train_idx) == len(X)


def test_BootstrapSplit_val_is_out_of_bag(X):
    splitter = BootstrapSplit(n_splits=3, random_state=0)

    for train_idx, val_idx in splitter.split(X):
        train_set = set(train_idx)
        val_set = set(val_idx)

        assert train_set.isdisjoint(val_set)


def test_BootstrapSplit_indices_within_range(X):
    splitter = BootstrapSplit(n_splits=3, random_state=0)

    for train_idx, val_idx in splitter.split(X):
        assert train_idx.min() >= 0
        assert train_idx.max() < len(X)
        assert val_idx.min() >= 0
        assert val_idx.max() < len(X)


def test_BootstrapSplit_reproducibility(X):
    splitter1 = BootstrapSplit(n_splits=3, random_state=42)
    splitter2 = BootstrapSplit(n_splits=3, random_state=42)

    splits1 = list(splitter1.split(X))
    splits2 = list(splitter2.split(X))

    for (t1, v1), (t2, v2) in zip(splits1, splits2):
        assert np.array_equal(t1, t2)
        assert np.array_equal(v1, v2)


def test_BootstrapSplit_str():
    splitter = BootstrapSplit(n_splits=4, shuffle=False, random_state=1)

    assert str(splitter) == (
        "BootstrapSplit(iterations=4, shuffle=False, random_state=1)"
    )


def test_BootstrapSplit_small_dataset():
    X = np.arange(2).reshape(2, 1)
    splitter = BootstrapSplit(n_splits=5, random_state=0)

    splits = list(splitter.split(X))

    # Should not crash even if some splits skipped
    assert isinstance(splits, list)