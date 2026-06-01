import pytest
from pathlib import Path
import numpy as np

from pipegenie.elite._elite import DiverseElite
from pipegenie.evolutionary._individual import Individual, Fitness
from pipegenie.syntax_tree._encoding import SyntaxTreeSchema
from pipegenie.grammar import parse_pipe_grammar

# pylint: disable=redefined-outer-name

def test_diverse_elite_invalid_size():
    with pytest.raises(ValueError):
        DiverseElite(max_size=0)

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
def test_diverseelite_add(setup_tree):
    tree = setup_tree
    fitness = Fitness((1.0,))
    fitness.values = (1.0,)
    ind = Individual(tree, fitness)

    elite = DiverseElite(max_size=1)
    elite.add(ind)

    assert len(elite) == 1
    assert elite[0].fitness.weighted_value == 1.0

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_diverseelite_update(setup_tree):
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

    elite = DiverseElite(max_size=2, div_weight=0.5)
    elite.update(population)

    assert len(elite) == 2
    assert all(ind.diversity > 0 for ind in elite)

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_diverseelite_multiple_update(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((0.9,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((1.0,))
    fitness2.values = (1.0,)
    
    a = Individual(tree, fitness1)
    b = Individual(tree, fitness2)
    a.prediction = [1, 1, 1]
    b.prediction = [0, 0, 0]

    elite = DiverseElite(max_size=2)
    elite.update([a])
    elite.update([b])

    assert a.fitness.valid
    assert b.fitness.valid
    assert len(elite) == 2
    assert all(ind.diversity > 0 for ind in elite)

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_diverseelite_same_update(setup_tree):
    tree = setup_tree
    fitness = Fitness((0.9,))
    fitness.values = (1.0,)
    
    ind = Individual(tree, fitness)
    ind.prediction = [1, 1, 1]

    elite = DiverseElite(max_size=2)
    elite.update([ind])
    elite.update([ind])

    assert ind.fitness.valid
    assert len(elite) == 1

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_diverseelite_diversity_influence(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((1.0,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((1.0,))
    fitness2.values = (2.0,)
    fitness3 = Fitness((1.0,))
    fitness3.values = (3.0,)

    a = Individual(tree, fitness1)
    b = Individual(tree, fitness2)
    c = Individual(tree, fitness3)

    a.prediction = [1, 1, 1]
    b.prediction = [1, 1, 1]
    c.prediction = [0, 0, 0]

    elite = DiverseElite(max_size=2, div_weight=0.9)
    elite.update([a, b, c])

    preds = [tuple(ind.prediction) for ind in elite]

    assert (0, 0, 0) in preds
    assert len(preds) == 2

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_diverseelite_invalid_fitness(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((0.9,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((1.0,))
    fitness2.values = (np.nan,)
    
    valid = Individual(tree, fitness1)
    invalid = Individual(tree, fitness2)
    valid.prediction = [1, 1, 1]
    invalid.prediction = [0, 0, 0]

    elite = DiverseElite(max_size=2)
    elite.update([valid])
    elite.update([invalid])


    assert not invalid.fitness.valid
    assert len(elite) == 1
    assert elite[0].fitness.weighted_value == 0.9

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_diverseelite_best_ind(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((0.5,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((0.7,))
    fitness2.values = (1.0,)
    fitness3 = Fitness((0.9,))
    fitness3.values = (1.0,)
    
    a = Individual(tree, fitness1)
    b = Individual(tree, fitness2)
    c = Individual(tree, fitness3)

    a.prediction = [1, 0]
    b.prediction = [0, 1]
    c.prediction = [1, 1]

    elite = DiverseElite(max_size=3)
    elite.update([a, b, c])

    best = elite.best_ind()

    assert best.fitness.weighted_value == 0.9
    assert best is not elite[0] 
    assert best is not elite[1] 
    assert best is not elite[2] 

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_diverseelite_remove_and_clear(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((0.9,))
    fitness1.values = (1.0,)
    fitness2 = Fitness((0.8,))
    fitness2.values = (1.0,)
    
    a = Individual(tree, fitness1)
    b = Individual(tree, fitness2)

    a.prediction = [1, 0]
    b.prediction = [0, 1]

    elite = DiverseElite(max_size=2)

    elite.add(a)
    elite.add(b)
    assert len(elite) == 2

    elite.remove(0)
    assert len(elite) == 1

    elite.clear()
    assert len(elite) == 0

@pytest.mark.parametrize("setup_tree", ["test_grammar_fixed.xml"], indirect=True)
def test_repr_and_str(setup_tree):
    tree = setup_tree
    fitness1 = Fitness((0.9,))
    fitness1.values = (1.0,)

    ind = Individual(tree, fitness1)
    ind.prediction = [1, 0]

    elite = DiverseElite(max_size=1)
    elite.add(ind)

    assert "DiverseElite" in repr(elite)
    assert str(elite)