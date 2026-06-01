import pytest
import random
from abc import ABC
from pathlib import Path

from pipegenie.evolutionary.selection import SelectionBase, RandomSelection, TournamentSelection
from pipegenie.evolutionary._individual import Individual, Fitness
from pipegenie.syntax_tree._encoding import SyntaxTreeSchema
from pipegenie.grammar import parse_pipe_grammar

def test_selectionbase_is_abstract():
    with pytest.raises(TypeError):
        SelectionBase()

def test_selectionbase_select_calls_perform_selection():
    class DummySelection(SelectionBase):
        def _perform_selection(self, population, k=1):
            return population[:k]
    
    population = [1, 2, 3, 4]
    selector = DummySelection()

    result = selector.select(population, k=2)

    assert result == [1, 2]

def test_randomselection_returns_k_individuals():
    population = [1, 2, 3, 4, 5]
    selector = RandomSelection()

    result = selector.select(population, k=3)

    assert len(result) == 3

def test_randomselection_returns_individuals_from_population():
    population = ["a", "b", "c"]
    selector = RandomSelection()

    result = selector.select(population, k=10)

    for individual in result:
        assert individual in population

def test_randomselection_is_reproducible_with_seed():
    population = [1, 2, 3, 4]
    selector = RandomSelection()

    random.seed(42)
    result1 = selector.select(population, k=3)

    random.seed(42)
    result2 = selector.select(population, k=3)

    assert result1 == result2

def test_randomselection_str():
    selector = RandomSelection()
    assert str(selector) == "RandomSelection"

def test_randomselection_empty_population():
    selector = RandomSelection()

    with pytest.raises(IndexError):
        selector.select([], k=1)

def test_randomselection_zero_k():
    population = [1, 2, 3]
    selector = RandomSelection()

    result = selector.select(population, k=0)

    assert result == []

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
def test_tournament_selection_returns_k_individuals(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((1.0,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((1.0,))
    fitness2.values = (2.0,)
    fitness3 = Fitness((1.0,))
    fitness3.values = (3.0,)
    
    population = [
        Individual(tree, fitness1),
        Individual(tree, fitness2),
        Individual(tree, fitness3),
    ]

    selector = TournamentSelection(tournament_size=2)

    result = selector.select(population, k=5) # it does not care about duplicity

    assert len(result) == 5

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_tournament_selection_returns_individuals_from_population(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((1.0,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((1.0,))
    fitness2.values = (2.0,)
    fitness3 = Fitness((1.0,))
    fitness3.values = (3.0,)
    
    population = [
        Individual(tree, fitness1),
        Individual(tree, fitness2),
        Individual(tree, fitness3),
    ]

    selector = TournamentSelection(tournament_size=2)

    result = selector.select(population, k=10)

    for individual in result:
        assert individual in population

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_tournament_selection_picks_best_from_tournament(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((1.0,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((5.0,))
    fitness2.values = (1.0,)
    fitness3 = Fitness((2.0,))
    fitness3.values = (1.0,)
    
    population = [
        Individual(tree, fitness1),
        Individual(tree, fitness2),
        Individual(tree, fitness3),
    ]

    selector = TournamentSelection(tournament_size=3)

    random.seed(0)
    result = selector.select(population, k=1)

    assert result[0].fitness.weighted_value == 5.0

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_tournament_selection_is_reproducible_with_seed(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((1.0,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((5.0,))
    fitness2.values = (1.0,)
    fitness3 = Fitness((2.0,))
    fitness3.values = (1.0,)
    fitness4 = Fitness((4.0,))
    fitness4.values = (1.0,)
    
    population = [
        Individual(tree, fitness1),
        Individual(tree, fitness2),
        Individual(tree, fitness3),
        Individual(tree, fitness4),
    ]

    selector = TournamentSelection(tournament_size=2)

    random.seed(42)
    result1 = selector.select(population, k=4)

    random.seed(42)
    result2 = selector.select(population, k=4)

    assert result1 == result2

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_tournament_selection_size_one(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((0.1,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((10.0,))
    fitness2.values = (1.0,)
    
    population = [
        Individual(tree, fitness1),
        Individual(tree, fitness2),
    ]

    selector = TournamentSelection(tournament_size=1)

    random.seed(1)
    result = selector.select(population, k=3)

    assert len(result) == 3
    for individual in result:
        assert individual in population

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_tournament_selection_size_larger_than_population(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((1.0,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((2.0,))
    fitness2.values = (1.0,)
    
    population = [
        Individual(tree, fitness1),
        Individual(tree, fitness2),
    ]

    selector = TournamentSelection(tournament_size=10)

    result = selector.select(population, k=3)

    assert len(result) == 3

def test_tournament_selection_str():
    selector = TournamentSelection(tournament_size=5)
    assert str(selector) == "TournamentSelection(tournament_size=5)"
