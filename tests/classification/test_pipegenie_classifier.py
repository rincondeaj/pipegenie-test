import numpy as np
import pytest
from pathlib import Path

from unittest.mock import MagicMock
from pipegenie.classification.pipegenie_classifier import PipegenieClassifier
from pipegenie.elite._elite import DiverseElite
from pipegenie.voting import VotingClassifier
from pipegenie.pipeline import Pipeline
from pipegenie.evolutionary._individual import Individual, Fitness
from pipegenie.syntax_tree._encoding import SyntaxTreeSchema
from pipegenie.grammar import parse_pipe_grammar


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

@pytest.fixture
def small_dataset():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    return X, y

def test_pipegenieclassifier_init():
    clf = PipegenieClassifier()

    assert clf.use_predict_proba is False
    assert clf.div_weight == 0.2
    assert clf.pop_size == 100
    assert clf.generations == 10

def test_pipegenieclassifier_create_ensemble_object():
    clf = PipegenieClassifier()

    steps1 = [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=100, random_state=1))
    ]
    pipe1 = Pipeline(steps1)

    steps2 = [
        ("scaler", StandardScaler()),
        ("knn", KMeans(n_clusters=2)),
    ]
    pipe2 = Pipeline(steps2)

    estimators = [("a", pipe1), ("b", pipe2)]
    weights = [1, 2]

    ensemble = clf._create_ensemble_object(estimators, weights)

    assert isinstance(ensemble, VotingClassifier)
    assert ensemble.weights == weights

def test_pipegenieclassifier_create_elite_object():
    clf = PipegenieClassifier(elite_size=5, diversity_weight=0.8)

    elite = clf._create_elite_object()

    assert isinstance(elite, DiverseElite)
    assert elite.max_size == 5
    assert elite.div_weight == 0.8

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

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_pipegenieclassifier_evaluate_pipeline_predict(setup_tree):
    clf = PipegenieClassifier(use_predict_proba=False)

    tree = setup_tree
    fitness = Fitness((1.0,))
    fitness.values = (1.0,)
    ind = Individual(tree, fitness)

    X_train = np.array([[0], [1]])
    y_train = np.array([0, 1])
    X_test = np.array([[1], [0]])

    result = clf._evaluate_pipeline(
        ind,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        pid_conn=None,
        memory_limit=1024,
    )

    result = result.tolist()

    assert len(result) == len(X_test) 
    assert all(item in [0, 1] for item in result) 

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_pipegenieclassifier_evaluate_pipeline_predict_proba(setup_tree):
    clf = PipegenieClassifier(use_predict_proba=True)

    tree = setup_tree
    fitness = Fitness((1.0,))
    fitness.values = (1.0,)
    ind = Individual(tree, fitness)

    X_train = np.array([[0], [1]])
    y_train = np.array([0, 1])
    X_test = np.array([[1], [0]])

    result = clf._evaluate_pipeline(
        ind,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        pid_conn=None,
        memory_limit=1024,
    )

    result = result.tolist()

    assert len(result) == len(X_test)
    assert all(len(item) == 2 for item in result)
    for inst_proba in result:
        assert all(0 <= item <= 1 for item in inst_proba)

def test_evaluate_pipeline_error():
    clf = PipegenieClassifier()

    ind = MagicMock()
    ind.pipeline = MagicMock()
    ind.pipeline.fit.side_effect = RuntimeError("error")

    result = clf._evaluate_pipeline(
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
def test_predict_calls_super():
    
    data = fetch_openml(name='iris', version=1, as_frame=False)
    X = data.data

    le = LabelEncoder()
    y = le.fit_transform(data.target)

    test_dir = Path(__file__).parent
    grammar_file = test_dir.parent / "data" / "test_grammar_fixed.xml"

    model = PipegenieClassifier(
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

@pytest.mark.slow
def test_predict_proba_uses_ensemble():
    
    data = fetch_openml(name='iris', version=1, as_frame=False)
    X = data.data

    le = LabelEncoder()
    y = le.fit_transform(data.target)

    test_dir = Path(__file__).parent
    grammar_file = test_dir.parent / "data" / "test_grammar_fixed.xml"

    model = PipegenieClassifier(
        # grammar=grammar_file,
        generations=5,
        pop_size=10,
        elite_size=5,
        n_jobs=5,
        seed=42,
        outdir= test_dir.parent / "test_results" / "sample-results",
        use_predict_proba=True,
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    
    assert len(y_pred) == len(X)
    assert all(len(item) == 2 for item in y_pred)
    for inst_proba in y_pred:
        assert all(0 <= item <= 1 for item in inst_proba)

def test_score(small_dataset):
    clf = PipegenieClassifier()

    X, y = small_dataset
    score = clf.score(X, y)

    assert score == 1.0
