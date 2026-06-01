import pytest
from pathlib import Path
import random
from pipegenie.evolutionary.mutation import MultiMutation, HyperparameterMutation, BranchMutation, NodeMutation, NonTerminalNode, TerminalNode
from pipegenie.evolutionary._individual import Individual
from pipegenie.syntax_tree._encoding import SyntaxTreeSchema
from pipegenie.syntax_tree._encoding import Primitive
from pipegenie.grammar import parse_pipe_grammar

import numpy as np

# pylint: disable=redefined-outer-name

@pytest.fixture
def setup_individual(request):
    grammar_file = request.param
    test_dir = Path(__file__).parent
    grammar_path = test_dir.parent / "data" / grammar_file
    grammar_type = "evoflow-xml"
    seed = 0

    root, terms, non_terms, pset, _ =  parse_pipe_grammar(
        grammar_path,
        grammar_type,
        seed,
    )

    nderiv = 10
    schema = SyntaxTreeSchema(root, nderiv, terms, non_terms)
    tree = schema.create_syntax_tree()
    return schema, Individual(tree)

@pytest.mark.parametrize("setup_individual", ["test_grammar.xml"], indirect=True)
def test_hyperparametermutation_no_mutation(setup_individual):
    schema, ind = setup_individual
    mutation = HyperparameterMutation(0) # 0% chance of mutation

    son, changed = mutation.mutate(ind, schema)
    
    for son_node, parent_node in zip(son, ind):
        assert son_node.symbol == parent_node.symbol

    assert not changed

@pytest.mark.parametrize("setup_individual", ["test_grammar_hp_classifier_mutation.xml"], indirect=True)
def test_hyperparametermutation_hyperparameter_mutation_classifier(setup_individual):
    schema, ind = setup_individual
    mutation = HyperparameterMutation(1) # 100% chance of mutation
    son = ind.clone() # TODO: clone ind inside operation?

    son, changed = mutation.mutate(son, schema)
    
    assert son[0] == ind[0]
    assert son[1] == ind[1]
    assert son[2] == ind[2]
    assert son[3] == ind[3]
    assert son[4] == ind[4]
    assert son[5] == ind[5]
    assert son[6].symbol == ind[6].symbol
    assert son[6].code != ind[6].code
    assert changed

@pytest.mark.parametrize("setup_individual", ["test_grammar_hp_preprocess_mutation.xml"], indirect=True)
def test_hyperparametermutation_hyperparameter_mutation_preprocess(setup_individual):
    schema, ind = setup_individual
    mutation = HyperparameterMutation(1) # 100% chance of mutation
    son = ind.clone() # TODO: clone ind inside operation?

    random.seed(12345)
    np.random.seed(12345)

    son, changed = mutation.mutate(son, schema)
    
    assert changed
    assert son[0] == ind[0]
    assert son[1] == ind[1]
    assert son[2] == ind[2]
    assert son[3] == ind[3]
    assert son[4].symbol == ind[4].symbol
    assert son[4].code != ind[4].code

@pytest.fixture
def setup_individual_seed(request):
    grammar_file, seed = request.param
    test_dir = Path(__file__).parent
    grammar_path = test_dir.parent / "data" / grammar_file
    grammar_type = "evoflow-xml"
    seed = request.param[1]

    root, terms, non_terms, pset, _ =  parse_pipe_grammar(
        grammar_path,
        grammar_type,
        seed,
    )

    nderiv = 100
    schema = SyntaxTreeSchema(root, nderiv, terms, non_terms)
    tree = schema.create_syntax_tree()
    return schema, Individual(tree)

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_hp_mutation_adaboost.xml", 0)], indirect=True)
def test_hyperparametermutation_hyperparameter_mutation_adaboost_inner_hp(setup_individual_seed):
    """
    Test to modify the adaboost estimator hiperparameter
    E.g.
        adaboost(decisionTree(10)) -> adaboost(decisionTree(3))
    """
    schema, ind = setup_individual_seed
    mutation = HyperparameterMutation(1) # 100% chance of mutation
    son = ind.clone() # TODO: clone ind inside operation?

    random.seed(0)
    np.random.seed(12345)

    son, changed = mutation.mutate(son, schema)
    
    assert changed
    assert son[0] == ind[0]
    assert son[1] == ind[1]
    assert son[2] == ind[2]
    assert son[3] == ind[3]
    assert son[4] == ind[4]
    assert son[5] == ind[5]
    assert son[6] == ind[6]
    assert son[7] == ind[7] 
    assert son[8] == ind[8]
    assert son[9].symbol == ind[9].symbol
    assert son[9].code != ind[9].code

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_hp_mutation_adaboost.xml", 0)], indirect=True)
def test_hyperparametermutation_hyperparameter_mutation_adaboost_hp(setup_individual_seed):
    """
    Test to modify the adaboost estimator hiperparameter
    E.g.
        adaboost(decisionTree(10)) -> adaboost(kNN(3))
    """
    schema, ind = setup_individual_seed
    mutation = HyperparameterMutation(1) # 100% chance of mutation
    son = ind.clone() # TODO: clone ind inside operation?

    random.seed(12345)
    np.random.seed(12345)

    son, changed = mutation.mutate(son, schema)
    
    assert changed
    assert son[0] == ind[0]
    assert son[1] == ind[1]
    assert son[2] == ind[2]
    assert son[3] == ind[3]
    assert son[4] == ind[4]
    assert son[5] == ind[5]
    assert son[6] != ind[6]
    assert son[6].symbol == ind[6].symbol
    assert son[7].code != ind[7].code
    assert son[7].symbol != ind[7].symbol

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_hp_mutation_adaboost_preprocess.xml", 0)], indirect=True)
def test_hyperparametermutation_preprocess_mutation_adaboost_hp(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = HyperparameterMutation(1) # 100% chance of mutation
    son = ind.clone() # TODO: clone ind inside operation?

    random.seed(12345)
    np.random.seed(12345)

    son, changed = mutation.mutate(son, schema)
    
    assert changed
    assert son[0] == ind[0]
    assert son[1] == ind[1]
    assert son[2] == ind[2]
    assert son[3] == ind[3]
    assert son[4].symbol == ind[4].symbol
    assert son[4].code != ind[4].code

def test_hyperparametermutation_string():
    mprob = 1
    mutation = HyperparameterMutation(mprob) 

    assert str(mutation) == f"HyperparameterMutation(p_mutation={mprob})"

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar.xml", 0)], indirect=True)
def test_branchmutation_no_mutation(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = BranchMutation(0) # 0% chance of mutation

    son, changed = mutation.mutate(ind, schema)
    
    for son_node, parent_node in zip(son, ind):
        assert son_node.symbol == parent_node.symbol

    assert not changed

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_branch_classifier_mutation.xml", 0)], indirect=True)
def test_branchmutation_branch_mutation(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = BranchMutation(1) # 100% chance of mutation
    son = ind.clone() # TODO: clone ind inside operation?

    random.seed(0)
    np.random.seed(12345)

    son, changed = mutation.mutate(son, schema)
    
    assert ind[0].symbol == son[0].symbol
    assert ind[1].symbol == son[1].symbol
    assert ind[2].symbol == son[2].symbol
    assert ind[3].symbol == son[3].symbol
    assert ind[4].symbol == son[4].symbol
    assert ind[5].symbol != son[5].symbol
    assert changed

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar.xml", 0)], indirect=True)
def test_nodemutation_no_mutation(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = NodeMutation(0) # 0% chance of mutation

    son, changed = mutation.mutate(ind, schema)
    
    for son_node, parent_node in zip(son, ind):
        assert son_node.symbol == parent_node.symbol

    assert not changed


@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_node_classifier_mutation.xml", 0)], indirect=True)
def test_nodemutation_hyperparameter(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = NodeMutation(1, 1) # 100% chance of mutation and 100% chance of changing a hyperparameter
    son = ind.clone() # TODO: clone ind inside operation?

    son, changed = mutation.mutate(son, schema)
    
    assert ind[0].symbol == son[0].symbol
    assert ind[1].symbol == son[1].symbol
    assert ind[2].symbol == son[2].symbol
    assert ind[3].symbol == son[3].symbol
    assert ind[4].symbol == son[4].symbol   # same hyperparameter
    assert ind[4].code != son[4].code       # different value
    assert changed

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_branch_classifier_mutation.xml", 0)], indirect=True)
def test_nodemutation_algorithm(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = NodeMutation(1, 0) # 100% chance of mutation and 0% chance of changing a hyperparameter
    son = ind.clone()

    random.seed(0)
    np.random.seed(12345)

    son, changed = mutation.mutate(son, schema)
    
    assert ind[0].symbol == son[0].symbol
    assert ind[1].symbol == son[1].symbol
    assert ind[2].symbol == son[2].symbol
    assert ind[3].symbol == son[3].symbol
    assert ind[4].symbol == son[4].symbol
    assert ind[5].symbol != son[5].symbol
    assert changed

def test_nodemutation_str_00():
    mutation = NodeMutation(0, 0)
    assert str(mutation) == "NodeMutation(p_mutation=0, p_hyper=0)"

def test_nodemutation_str_01():
    mutation = NodeMutation(0, 1)
    assert str(mutation) == "NodeMutation(p_mutation=0, p_hyper=1)"

def test_nodemutation_str_10():
    mutation = NodeMutation(1, 0)
    assert str(mutation) == "NodeMutation(p_mutation=1, p_hyper=0)"

def test_nodemutation_str_11():
    mutation = NodeMutation(1, 1)
    assert str(mutation) == "NodeMutation(p_mutation=1, p_hyper=1)"

def test_nodemutation_3():
    mutation = NodeMutation(3, 3)
    assert str(mutation) == "NodeMutation(p_mutation=3, p_hyper=3)"

def test_nodemutation_neg():
    mutation = NodeMutation(-1, -1)
    assert str(mutation) == "NodeMutation(p_mutation=-1, p_hyper=-1)"

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar.xml", 0)], indirect=True)
def test_multimutation_no_mutation(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = MultiMutation(0) # 0% chance of mutation

    son, changed = mutation.mutate(ind, schema)
    
    for son_node, parent_node in zip(son, ind):
        assert son_node.symbol == parent_node.symbol

    assert not changed

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_node_classifier_mutation.xml", 0)], indirect=True)
def test_multimutation_hyperparameter(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = MultiMutation(1, 1) # 100% chance of mutation and 100% chance of changing a hyperparameter
    son = ind.clone()

    son, changed = mutation.mutate(son, schema)

    
    assert ind[0].symbol == son[0].symbol
    assert ind[1].symbol == son[1].symbol
    assert ind[2].symbol == son[2].symbol
    assert ind[3].symbol == son[3].symbol
    assert ind[4].symbol == son[4].symbol   # same hyperparameter
    assert ind[4].code != son[4].code       # different value
    assert changed

@pytest.mark.parametrize("setup_individual_seed", [("test_grammar_branch_classifier_mutation.xml", 0)], indirect=True)
def test_multimutation_algorithm(setup_individual_seed):
    schema, ind = setup_individual_seed
    mutation = MultiMutation(1, 0) # 100% chance of mutation and 0% chance of changing a hyperparameter
    son = ind.clone()

    random.seed(0)
    np.random.seed(12345)

    son, changed = mutation.mutate(son, schema)
    
    assert ind[0].symbol == son[0].symbol
    assert ind[1].symbol == son[1].symbol
    assert ind[2].symbol == son[2].symbol
    assert ind[3].symbol == son[3].symbol
    assert ind[4].symbol == son[4].symbol
    assert ind[5].symbol != son[5].symbol
    assert changed