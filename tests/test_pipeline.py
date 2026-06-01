import pytest
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from pipegenie.pipeline import (
    Pipeline,
    make_pipeline,
    _final_estimator_has,
)

def test_pipeline_fit_predict():
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        random_state=0
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    assert y_pred.shape == (50,)

def test_makepipeline_estimators():
    X, y = make_classification(n_samples=20, n_features=4, random_state=1)

    pipe = make_pipeline([
        StandardScaler(),
        LogisticRegression(max_iter=100),
    ])

    assert isinstance(pipe, Pipeline)
    assert "standardscaler" in pipe.named_steps
    assert "logisticregression" in pipe.named_steps

    pipe.fit(X, y)
    assert pipe.predict(X).shape == (20,)

def test_makepipeline_unique_names_do_not_crash():
    pipe = make_pipeline([
        StandardScaler(),
        LogisticRegression(),
    ])

    names = list(pipe.named_steps.keys())
    assert names == ["standardscaler", "logisticregression"]

def test_makepipeline_duplicate_estimators():
    pipe = make_pipeline([
        StandardScaler(),
        StandardScaler(),
        LogisticRegression(max_iter=100),
    ])

    names = list(pipe.named_steps.keys())

    assert names == [
        "standardscaler-1",
        "standardscaler-2",
        "logisticregression",
    ]

def test_pipeline_score():
    X, y = make_classification(n_samples=40, n_features=5, random_state=0)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    pipe.fit(X, y)
    score = pipe.score(X, y)

    assert 0.0 <= score <= 1.0

def test_pipeline_predict_proba():
    X, y = make_classification(
        n_samples=30,
        n_features=6,
        random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (30, 2)

def test_pipeline_predict_proba_not_available():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("cluster", KMeans(n_clusters=2)),
    ])

    with pytest.raises(AttributeError):
        pipe.predict_proba([[0, 0]])

def test_finalestimator_error():
    check = _final_estimator_has("predict")

    with pytest.raises(ValueError, match="The object is not a Pipeline"):
        check(object())

def test_pipeline_error_duplicate_names():

    steps = [
        ("scaler", StandardScaler()),
        ("scaler", KMeans(n_clusters=2)),
    ]

    with pytest.raises(ValueError, match="Step names are not unique"):
        Pipeline(steps)

def test_pipeline_error_invalid_intermediate_step():
    class BadStep:
        """Does not implement any required methods"""
        pass

    steps = [
        ("badstep", BadStep()),       # intermediate
        ("cluster", KMeans(n_clusters=2)),
    ]

    with pytest.raises(ValueError, match="All intermediate steps should implement at least"):
        Pipeline(steps)

def test_pipeline_error_invalid_final_step():
    class DummyEstimator:
        """Does not implement any required methods"""
        pass

    steps = [
        ("scaler", StandardScaler()),
        ("model", DummyEstimator()),  # final estimator
    ]

    with pytest.raises(ValueError, match="The final estimator should implement 'fit' and a prediction"):
        Pipeline(steps)

def test_pipeline_get_item():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    assert isinstance(pipe[0], StandardScaler)
    assert isinstance(pipe[1], LogisticRegression)

def test_pipeline_print():
    steps = [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ]
    pipe = Pipeline(steps)

    assert repr(pipe) == f"Pipeline(steps={steps})"

def test_pipeline_string():
    steps = [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ]
    pipe = Pipeline(steps)

    assert str(pipe) == "Pipeline(\n\tscaler: StandardScaler()\n\tclf: LogisticRegression(max_iter=200)\n)"

def test_pipeline_fit_branches():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    # Branch 1: fit_transform
    class FitTransformStep:
        def fit_transform(self, X, y=None):
            self.fitted = True
            return X + 1

        def transform(self, X):
            return X + 1  

    # Branch 2: transform
    class TransformStep:
        def fit(self, X, y=None):
            self.fitted = True
        def transform(self, X):
            return X * 2

    # Branch 3: fit_resample
    class FitResampleStep:
        def fit_resample(self, X, y=None):
            self.fitted = True
            return X, y 

    class DummyEstimator:
        def fit(self, X, y):
            self.fitted = True
            self.X_ = X
            self.y_ = y
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

    steps = [
        ("fit_transform", FitTransformStep()),
        ("transform", TransformStep()),
        ("resample", FitResampleStep()),
        ("estimator", DummyEstimator()),
    ]

    pipe = Pipeline(steps)
    pipe.fit(X, y)

    assert steps[0][1].fitted is True
    assert steps[1][1].fitted is True
    assert steps[2][1].fitted is True
    assert steps[3][1].fitted is True

    assert np.array_equal(steps[3][1].X_, ((X + 1) * 2)) 
    assert np.array_equal(steps[3][1].y_, y)

def test_pipeline_predict_log_proba():
    X, y = make_classification(
        n_samples=20,
        n_features=4,
        n_classes=2,
        random_state=1
    )

    steps = [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=100, random_state=1))
    ]

    pipe = Pipeline(steps)
    pipe.fit(X, y)

    y_log_proba = pipe.predict_log_proba(X)

    assert isinstance(y_log_proba, np.ndarray)
    assert y_log_proba.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(y_log_proba))

def test_pipeline_decision_function():
    X, y = make_classification(
        n_samples=20,
        n_features=4,
        n_classes=2,
        random_state=1
    )

    steps = [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=100, random_state=1))
    ]

    pipe = Pipeline(steps)
    pipe.fit(X, y)

    y_decision = pipe.decision_function(X)

    assert isinstance(y_decision, np.ndarray)
    assert y_decision.shape == (X.shape[0],) or y_decision.shape == (X.shape[0], 1)
    assert np.all(np.isfinite(y_decision))