import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from pipegenie.voting import VotingClassifier, VotingRegressor

@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    return X, y

def test_votingclassifier_fit(classification_data):
    X, y = classification_data

    estimators = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier(random_state=0)),
    ]

    clf = VotingClassifier(estimators, n_jobs=1)
    clf.fit(X, y)

    for _, est in clf.estimators:
        assert hasattr(est, "classes_")

def test_votingclassifier_predict(classification_data):
    X, y = classification_data

    estimators = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier(random_state=0)),
        ("knn", KNeighborsClassifier()),
    ]

    clf = VotingClassifier(estimators, n_jobs=1)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)).issubset({0, 1})

def test_votingclassifier_weighted_voting(classification_data):
    X, y = classification_data

    estimators = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier(random_state=0)),
    ]

    weights = np.array([10.0, 1.0])

    clf = VotingClassifier(estimators, weights=weights, n_jobs=1)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    lr_pred = estimators[0][1].fit(X, y).predict(X)

    agreement = np.mean(y_pred == lr_pred)
    assert agreement > 0.9

def test_votingclassifier_predict_proba_success(classification_data):
    X, y = classification_data

    estimators = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("knn", KNeighborsClassifier()),
    ]

    clf = VotingClassifier(estimators, n_jobs=1)
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_votingclassifier_predict_proba_error():
    X, y = make_classification(random_state=0)

    estimators = [
        ("svc", LinearSVC()),
        ("dt", DecisionTreeClassifier()),
    ]

    clf = VotingClassifier(estimators)
    clf.fit(X, y)

    with pytest.raises(AttributeError):
        clf.predict_proba(X)

def test_votingclassifier_score(classification_data):
    X, y = classification_data

    estimators = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier(random_state=0)),
    ]

    clf = VotingClassifier(estimators, n_jobs=1)
    clf.fit(X, y)

    score = clf.score(X, y)

    assert 0.0 <= score <= 1.0

def test_votingclassifier_weight_length_error(classification_data):
    X, y = classification_data

    estimators = [
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier()),
    ]

    clf = VotingClassifier(
        estimators=estimators,
        weights=[1.0],  # wrong length
    )

    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_votingclassifier_fit_parallel(classification_data):
    X, y = classification_data

    estimators = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier(random_state=0)),
    ]

    clf = VotingClassifier(
        estimators=estimators,
        n_jobs=2,
    )

    clf.fit(X, y)

    y_pred = clf.predict(X)

    assert y_pred.shape == y.shape
    assert np.isfinite(y_pred).all()

@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=200,
        n_features=6,
        noise=0.1,
        random_state=42,
    )
    return X, y

def test_votingregressor_fit(regression_data):
    X, y = regression_data

    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge(alpha=1.0)),
    ]

    reg = VotingRegressor(estimators, n_jobs=1)
    reg.fit(X, y)

    for _, est in reg.estimators:
        assert hasattr(est, "coef_")

def test_votingregressor_predict(regression_data):
    X, y = regression_data

    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge()),
    ]

    reg = VotingRegressor(estimators, n_jobs=1)
    reg.fit(X, y)

    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape
    assert np.issubdtype(y_pred.dtype, np.floating)

def test_votingregressor_unweighted_average(regression_data):
    X, y = regression_data

    est1 = LinearRegression()
    est2 = DecisionTreeRegressor(random_state=0)

    estimators = [
        ("lr", est1),
        ("dt", est2),
    ]

    reg = VotingRegressor(estimators, n_jobs=1)
    reg.fit(X, y)

    y_pred = reg.predict(X)

    y_manual = np.mean(
        [est1.fit(X, y).predict(X),
         est2.fit(X, y).predict(X)],
        axis=0,
    )

    assert np.allclose(y_pred, y_manual)

def test_votingregressor_weighted_average(regression_data):
    X, y = regression_data

    est1 = LinearRegression()
    est2 = DecisionTreeRegressor(random_state=0)

    weights = np.array([0.9, 0.1])

    estimators = [
        ("lr", est1),
        ("dt", est2),
    ]

    reg = VotingRegressor(estimators, weights=weights, n_jobs=1)
    reg.fit(X, y)

    y_pred = reg.predict(X)

    y_manual = np.average(
        [est1.fit(X, y).predict(X),
         est2.fit(X, y).predict(X)],
        axis=0,
        weights=weights,
    )

    assert np.allclose(y_pred, y_manual)

def test_votingregressor_score(regression_data):
    X, y = regression_data

    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge()),
    ]

    reg = VotingRegressor(estimators, n_jobs=1)
    reg.fit(X, y)

    score = reg.score(X, y)

    assert np.isfinite(score)

def test_votingregressor_weight_length_mismatch(regression_data):
    X, y = regression_data

    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge()),
    ]

    reg = VotingRegressor(
        estimators=estimators,
        weights=[1.0],  # wrong length
    )

    with pytest.raises(ValueError):
        reg.fit(X, y)

def test_votingregressor_fit_parallel(regression_data):
    X, y = regression_data

    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge()),
    ]

    reg = VotingRegressor(
        estimators=estimators,
        n_jobs=2,
    )

    reg.fit(X, y)

    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape
    assert np.isfinite(y_pred).all()