import numpy as np
import pytest
from pathlib import Path
import random

from unittest.mock import MagicMock
from pipegenie.regression.pipegenie_regressor import PipegenieRegressor
from pipegenie.elite._elite import DiverseElite
from pipegenie.voting import VotingRegressor
from pipegenie.pipeline import Pipeline
from pipegenie.evolutionary._individual import Individual, Fitness
from pipegenie.syntax_tree._encoding import SyntaxTreeSchema
from pipegenie.grammar import parse_pipe_grammar


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml

@pytest.fixture
def small_dataset():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0.5, 1.5, 2.0, 0.0])
    return X, y

def test_pipegenieregressor_create_ensemble_object():
    reg = PipegenieRegressor()

    steps1 = [("scaler", StandardScaler()), ("lr", LinearRegression())]
    pipe1 = Pipeline(steps1)

    steps2 = [("scaler", StandardScaler()), ("lr2", LinearRegression())]
    pipe2 = Pipeline(steps2)

    estimators = [("a", pipe1), ("b", pipe2)]
    weights = [1, 2]

    ensemble = reg._create_ensemble_object(estimators, weights)

    assert isinstance(ensemble, VotingRegressor)
    assert ensemble.weights == weights

@pytest.fixture
def setup_tree(request):
    grammar_file = request.param
    test_dir = Path(__file__).parent
    grammar_path = test_dir.parent / "data" / grammar_file
    grammar_type = "evoflow-xml"
    seed = 0

    root, terms, non_terms, _, _ =  parse_pipe_grammar(
        grammar_path,
        grammar_type,
        seed,
    )

    nderiv = 10
    schema = SyntaxTreeSchema(root, nderiv, terms, non_terms)
    tree = schema.create_syntax_tree()
    return tree

@pytest.mark.parametrize("setup_tree", ["test_grammar_regression.xml"], indirect=True)
def test_pipegenieregressor_evaluate_pipeline_predict(setup_tree):
    reg = PipegenieRegressor()
    
    tree = setup_tree
    fitness = Fitness((1.0,))
    fitness.values = (1.0,)
    
    ind = Individual(tree, fitness)
    
    result = reg._evaluate_pipeline(
        ind,
        X_train=np.zeros((2, 1)),
        y_train=np.array([1.0, 2.0]),
        X_test=np.zeros((2, 1)),
        pid_conn=None,
        memory_limit=1024
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 2

def test_pipegenieregressor_evaluate_pipeline_error():
    reg = PipegenieRegressor()
    
    ind = MagicMock()
    ind.pipeline = MagicMock()
    ind.pipeline.fit.side_effect = RuntimeError("error")

    random.seed(0)
    np.random.seed(0)

    result = reg._evaluate_pipeline(
        ind,
        X_train=[[0]],
        y_train=[0],
        X_test=[[0]],
        pid_conn=None,
        memory_limit=1024,
    )

    assert isinstance(result, str)
    assert result.startswith("eval_error")

@pytest.mark.slow
def test_pipegenieregressor_predict_calls_super():
    
    data = fetch_openml(name='diabetes', version=1, as_frame=False)
    X = data.data
    y = data.target

    test_dir = Path(__file__).parent
    grammar_file = test_dir.parent / "data" / "test_grammar_fixed.xml"

    model = PipegenieRegressor(
        # grammar=grammar_file,
        generations=5,
        pop_size=10,
        elite_size=5,
        n_jobs=1,
        seed=42,
        outdir= test_dir.parent / "test_results" / "sample-results",
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert len(y_pred) == len(X)
    assert all(item in [0, 1] for item in y_pred) 

# def test_fit_predict_score(small_dataset):
#     X, y = small_dataset
#     reg = PipegenieRegressor(generations=0)  # skip evolution for speed
    
#     reg.fit(X, y)
#     y_pred = reg.predict(X)
    
#     assert y_pred.shape == y.shape
    
#     score = reg.score(X, y)
#     assert isinstance(score, float)