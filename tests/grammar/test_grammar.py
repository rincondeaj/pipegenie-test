from typing import TYPE_CHECKING
from pathlib import Path
import types
import pytest
from pipegenie.grammar._grammar import TerminalCode, parse_pipe_grammar, load_productions, get_ephemeral_function
from pipegenie.grammar.parser import EvoflowXML
from pipegenie.syntax_tree._encoding import NonTerminalNode, TerminalNode
from pipegenie.syntax_tree._primitive_set import PrimitiveSet

from sklearn.tree import DecisionTreeClassifier

# pylint: disable=redefined-outer-name

def test_terminalcode():
    terminal_code = TerminalCode("increment", ["n"], seed = 0)

    assert terminal_code.terminal_code == "increment"
    assert isinstance(terminal_code.hp_list, list)
    assert len(terminal_code.hp_list) == 1

@pytest.fixture
def fake_estimators_module():
    class GoodEstimator:
        def __init__(self):
            self.max_depth = None
            self.min_samples = None
            self.random_state = None

    class NeedsArgsEstimator:
        def __init__(self, x):
            pass

    class ExplodingEstimator:
        def __init__(self):
            raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(
        GoodEstimator=GoodEstimator,
        NeedsArgsEstimator=NeedsArgsEstimator,
        ExplodingEstimator=ExplodingEstimator,
    )

    return fake_module

@pytest.fixture
def patch_import(monkeypatch, fake_estimators_module):
    def fake_import(name):
        if name == "fake":
            return fake_estimators_module
        raise ImportError(name)

    monkeypatch.setattr(
        "importlib.import_module",
        fake_import
    )

def test_terminalcode_generate_term_code_success(patch_import):
    terminal_code = TerminalCode("fake.GoodEstimator",
        ["hp::max_depth", "hp::min_samples"], seed = 0)
    code = terminal_code.generate_term_code()
    estimator = code(3, 5)

    assert estimator.max_depth == 3
    assert estimator.min_samples == 5
    assert estimator.random_state == 0
 
def test_terminalcode_generate_term_code_importerror(monkeypatch):
    monkeypatch.setattr(
        "importlib.import_module",
        lambda _: (_ for _ in ()).throw(ImportError("nope"))
    )

    terminal_code = TerminalCode("missing.Class", [], seed=0)
    code = terminal_code.generate_term_code()

    with pytest.raises(ImportError, match="Error importing module"):
        code()
 
def test_terminalcode_generate_term_code_attributeerror(patch_import):
    terminal_code = TerminalCode("fake.DoesNotExist", [], seed=0)
    code = terminal_code.generate_term_code()

    with pytest.raises(AttributeError, match="Error importing class"):
        code()
 
def test_terminalcode_generate_term_code_typeerror(patch_import):
    terminal_code = TerminalCode("fake.NeedsArgsEstimator", [], seed=0)
    code = terminal_code.generate_term_code()

    with pytest.raises(TypeError, match="Error instantiating class"):
        code()
 
def test_terminalcode_generate_term_code_exception(patch_import):
    terminal_code = TerminalCode("fake.ExplodingEstimator", [], seed=0)
    code = terminal_code.generate_term_code()

    with pytest.raises(Exception):
        code()

def test_get_ephemeral_function_categorical():
    hyperparameter_term = {"name": "dt::criterion","type": "categorical","values": "gini;entropy","default": "gini","children": None}
    results = ["gini", "entropy"]
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert val in results

def test_get_ephemeral_function_bool():
    hyperparameter_term = {"name": "dt::criterion","type": "bool","default": "False","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, bool)

def test_get_ephemeral_function_fix_bool():
    hyperparameter_term = {"name": "dt::criterion","type": "fix_bool","value": "False","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, bool)

def test_get_ephemeral_function_int():
    hyperparameter_term = {"name": "dt::min_samples_split","type": "int","lower": "2","upper": "20","default": "2","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, int)
    assert val in range(2, 21)

def test_get_ephemeral_function_int_step():
    hyperparameter_term = {"name": "dt::min_samples_split","type": "int","lower": "2","upper": "20","default": "2","step": "2","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, int)
    assert val in range(2, 21)

def test_get_ephemeral_function_fix_int():
    hyperparameter_term = {"name": "dt::min_samples_split","type": "fix_int","value": "3","lower": "2","upper": "20","default": "2","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, int)
    assert val in range(2, 21)

def test_get_ephemeral_function_log_int():
    hyperparameter_term = {"name": "dt::min_samples_split","type": "int","log":"True","lower": "2","upper": "20","default": "2","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, int)
    assert val in range(2, 21)

def test_get_ephemeral_function_log_int_step():
    hyperparameter_term = {"name": "dt::min_samples_split","type": "int","log":"True","lower": "2","upper": "20","default": "2","step": "2","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, int)
    assert val in range(2, 21)

def test_get_ephemeral_function_float():
    hyperparameter_term = {"name": "dt::max_features","type": "float","lower": "0","upper": "1","default": "0.5","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, float)
    assert 0 <= val <= 1

def test_get_ephemeral_function_fix_float():
    hyperparameter_term = {"name": "dt::max_features","type": "fix_float","value":"0.5","lower": "0","upper": "1","default": "0.5","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, float)
    assert 0 <= val <= 1

def test_get_ephemeral_function_float_step():
    hyperparameter_term = {"name": "dt::max_features","type": "float","value":"0.5","lower": "0","upper": "1","default": "0.5","step": "0.1","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, float)
    assert 0 <= val <= 1

def test_get_ephemeral_function_log_float():
    hyperparameter_term = {"name": "dt::max_features","type": "float","log":"True","lower": "0","upper": "1","default": "0.5","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, float)
    assert 0 <= val <= 1

def test_get_ephemeral_function_log_float_step():
    hyperparameter_term = {"name": "dt::max_features","type": "float","log":"True","lower": "0","upper": "1","default": "0.5","step": "0.1","children": None}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, float)
    assert 0 <= val <= 1

def test_get_ephemeral_function_tuple():
    children = [
        {"type": "int","lower": "1","upper": "3","default": "2","children": None},
        {"type": "int","lower": "10","upper": "20","default": "15","children": None},
    ]

    hyperparameter_term = {"name":"paramTuple","type": "tuple","children": children}
    hparam = get_ephemeral_function(hyperparameter_term)
    val = hparam()

    assert isinstance(val, tuple)
    assert len(val) == 2
    assert 1 <= val[0] <= 3
    assert 10 <= val[1] <= 20

def test_get_ephemeral_function_union():
    children = [
        {"type": "categorical","values":"sqrt;log2","default": None,"children": None},
        {"type": "int","lower": "10","upper": "20","default": "15","children": None},
    ]

    hyperparameter_term = {"name":"paramUnion","type": "union","children": children}
    func = get_ephemeral_function(hyperparameter_term)

    val = func()

    assert val in ["sqrt","log2"] if isinstance(val, str) else 10 <= val <= 20

def test_get_ephemeral_function_warning():
    hyperparameter_term = {"name": "dt::criterion","type": "NonExistant","default": "False","children": None}

    with pytest.warns(UserWarning, match="Unknown hyper-parameter type"):
        func = get_ephemeral_function(hyperparameter_term)

@pytest.fixture
def setup_grammar(request):
    grammar_file = request.param
    test_dir = Path(__file__).parent
    return test_dir.parent / "data" / grammar_file

@pytest.mark.parametrize("setup_grammar", ["test_grammar.xml"], indirect=True)
def test_load_productions(setup_grammar):
    grammar_path = setup_grammar
    parser = EvoflowXML()

    _, _, _, non_terms = parser.parse(grammar_path)
    non_terms = load_productions(non_terms)

    assert isinstance(non_terms, list)
    assert all(isinstance(non_term, NonTerminalNode) for non_term in non_terms)

@pytest.mark.parametrize("setup_grammar", ["test_grammar.xml"], indirect=True)
def test_parse_pipe_grammar_default(setup_grammar):
    """
    """
    grammar_path = setup_grammar
    root, terms, non_terms, pset, terms_families = parse_pipe_grammar(grammar_path, "evoflow-xml", seed = 0)

    assert root == "workflow"
    assert isinstance(terms, list)
    assert all(isinstance(term, TerminalNode) for term in terms)
    assert isinstance(non_terms, list)
    assert all(isinstance(non_term, NonTerminalNode) for non_term in non_terms)
    assert isinstance(pset, PrimitiveSet)
    assert isinstance(terms_families, dict)

@pytest.mark.parametrize("setup_grammar", ["test_grammar_no_hp.xml"], indirect=True)
def test_parse_pipe_grammar_no_hp(setup_grammar):
    """
    """
    grammar_path = setup_grammar
    root, terms, non_terms, pset, terms_families = parse_pipe_grammar(grammar_path, "evoflow-xml", seed = 0)

    assert root == "workflow"
    assert isinstance(terms, list)
    assert all(isinstance(term, TerminalNode) for term in terms)
    assert isinstance(non_terms, list)
    assert all(isinstance(non_term, NonTerminalNode) for non_term in non_terms)
    assert isinstance(pset, PrimitiveSet)
    assert isinstance(terms_families, dict)

@pytest.mark.parametrize("setup_grammar", ["test_grammar_no_hp_production.xml"], indirect=True)
def test_parse_pipe_grammar_no_hp_production(setup_grammar):
    """
    """
    grammar_path = setup_grammar
    root, terms, non_terms, pset, terms_families = parse_pipe_grammar(grammar_path, "evoflow-xml", seed = 0)

    assert root == "workflow"
    assert isinstance(terms, list)
    assert all(isinstance(term, TerminalNode) for term in terms)
    assert isinstance(non_terms, list)
    assert all(isinstance(non_term, NonTerminalNode) for non_term in non_terms)
    assert isinstance(pset, PrimitiveSet)
    assert isinstance(terms_families, dict)

@pytest.mark.parametrize("setup_grammar", ["test_grammar_adaboost.xml"], indirect=True)
def test_parse_pipe_grammar_adaboost(setup_grammar):
    """
    """
    grammar_path = setup_grammar
    root, terms, non_terms, pset, terms_families = parse_pipe_grammar(grammar_path, "evoflow-xml", seed = 0)

    assert root == "workflow"
    assert isinstance(terms, list)
    assert all(isinstance(term, TerminalNode) for term in terms)
    assert isinstance(non_terms, list)
    assert all(isinstance(non_term, NonTerminalNode) for non_term in non_terms)
    assert isinstance(pset, PrimitiveSet)
    assert isinstance(terms_families, dict)

@pytest.mark.parametrize("setup_grammar", ["test_grammar_adaboost_warning.xml"], indirect=True)
def test_parse_pipe_grammar_adaboost_warning(setup_grammar):
    """
    """
    grammar_path = setup_grammar

    with pytest.warns(UserWarning, match="Non terminal node"):
        root, terms, non_terms, pset, terms_families = parse_pipe_grammar(grammar_path, "evoflow-xml", seed = 0)