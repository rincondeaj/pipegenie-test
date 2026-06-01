from pipegenie.evolutionary._individual import Fitness, Individual
from pipegenie.syntax_tree._encoding import TerminalNode
from pipegenie.syntax_tree._primitive_set import Primitive
import math
import numpy as np
import pytest

def test_fitness_init_with_valid_weights():
    f = Fitness(weights=(1.0, -1.0))
    assert f.weights == (1.0, -1.0)
    assert f.values == ()

def test_fitness_init_with_none_weights_raises():
    with pytest.raises(ValueError, match="Weights cannot be None or empty"):
        Fitness(weights=None)

def test_fitness_init_with_empty_weights_raises():
    with pytest.raises(ValueError, match="Weights cannot be None or empty"):
        Fitness(weights=())

def test_fitness_set_and_get_values():
    f = Fitness(weights=(1.0, 2.0))
    f.values = (3.0, 4.0)
    assert f.values == (3.0, 4.0)

def test_fitness_set_values_none_raises():
    f = Fitness(weights=(1.0,))
    with pytest.raises(ValueError, match="Values cannot be None or empty"):
        f.values = None

def test_fitness_set_values_empty_raises():
    f = Fitness(weights=(1.0,))
    with pytest.raises(ValueError, match="Values cannot be None or empty"):
        f.values = ()

def test_fitness_set_values_wrong_length_raises():
    f = Fitness(weights=(1.0, 2.0))
    with pytest.raises(ValueError, match="Values and weights must have the same length"):
        f.values = (1.0,)

def test_fitness_delete_values():
    f = Fitness(weights=(1.0,))
    f.values = (5.0,)
    del f.values
    assert f.values == ()

def test_fitness_valid_true():
    f = Fitness(weights=(1.0, 1.0))
    f.values = (2.0, 3.0)
    assert f.valid

def test_fitness_valid_false_when_nan_weighted_value():
    f = Fitness(weights=(1.0,))
    f.values = (np.nan,)
    assert not f.valid

def test_fitness_valid_false_when_no_values():
    f = Fitness(weights=(1.0,))
    assert not f.valid

def test_fitness_weighted_value():
    f = Fitness(weights=(1.0, -2.0))
    f.values = (3.0, 4.0)
    assert f.weighted_value == pytest.approx(3.0 - 8.0)

def test_fitness_weighted_value_error():
    f = Fitness(weights=(1.0, -2.0))
    
    with pytest.raises(ValueError, match="Fitness values have not been set"):
        f.weighted_value

def test_fitness_weighted_value_with_nan():
    f = Fitness(weights=(1.0,))
    f.values = (np.nan,)
    assert math.isnan(f.weighted_value)

def test_fitness_invalidate():
    f = Fitness(weights=(1.0,))
    f.values = (10.0,)
    f.invalidate()
    assert f.values == ()
    assert not f.valid

def test_fitness_equal_true():
    f1 = Fitness(weights=(1.0, 2.0))
    f2 = Fitness(weights=(1.0, 2.0))
    f1.values = (3.0, 4.0)
    f2.values = (3.0, 4.0)

    assert f1 == f2

def test_fitness_equal_false_values():
    f1 = Fitness(weights=(1.0,))
    f2 = Fitness(weights=(1.0,))
    f1.values = (1.0,)
    f2.values = (2.0,)

    assert f1 != f2

def test_fitness_equal_false_weights():
    f1 = Fitness(weights=(1.0,))
    f2 = Fitness(weights=(2.0,))
    f1.values = (1.0,)
    f2.values = (1.0,)

    assert f1 != f2

def test_fitness_equal_error():
    f = Fitness(weights=(1.0,))
    with pytest.raises(ValueError, match="Cannot compare Fitness with"):
        f == 42

def test_individual_init_fitness():
    fitness = Fitness((1.0,))
    content = ["node1", "node2"]

    ind = Individual(content=content, fitness=fitness)

    assert ind.content == content
    assert ind.fitness is fitness
    assert np.isnan(ind.diversity)
    assert ind.prediction.size == 0

def test_individual_init_default_fitness():
    ind = Individual(content=["node"])

    assert isinstance(ind.fitness, Fitness)
    assert ind.fitness.weights == (1.0,)

def test_individual_clone_primitive():
    primitive_node = Primitive("example", ["arg"]*3, object)
    ind = Individual(content=[primitive_node])
    ind.fitness.values = (5.0,)

    clone = ind.clone()

    assert clone == ind
    assert clone is not ind
    assert clone.fitness is not ind.fitness

def test_individual_clone_terminalnode():
    primitive_node = Primitive("example", ["arg"]*3, object)
    terminal_node = TerminalNode("example", primitive_node, "test")
    ind = Individual(content=[terminal_node])
    ind.fitness.values = (5.0,)

    clone = ind.clone()

    assert clone == ind
    assert clone is not ind
    assert clone.fitness is not ind.fitness

def test_individual_reset():
    terminal_node = TerminalNode("example", "example.example", "test")
    ind = Individual(content=[terminal_node])
    ind.fitness.values = (1.0,)

    ind.reset()

    assert ind.fitness.values == ()
    assert not ind.fitness.valid

def test_individual_equal_true():
    primitive_node = Primitive("example", ["arg"]*3, object)
    terminal_node = TerminalNode("example", primitive_node, "test")
    ind1 = Individual(content=[terminal_node])
    ind2 = Individual(content=[terminal_node])

    ind1.fitness.values = (1.0,)
    ind2.fitness.values = (1.0,)

    assert ind1 == ind2

def test_individual_equal_false_fitness():
    primitive_node = Primitive("example", ["arg"]*3, object)
    terminal_node = TerminalNode("example", primitive_node, "test")
    ind1 = Individual(content=[terminal_node])
    ind2 = Individual(content=[terminal_node])

    ind1.fitness.values = (1.0,)
    ind2.fitness.values = (5.0,)

    assert not (ind1 == ind2)

def test_individual_equal_false_pipeline():
    primitive_node1 = Primitive("example", ["arg"]*3, object)
    terminal_node1 = TerminalNode("example", primitive_node1, "test")
    primitive_node2 = Primitive("example", ["arg"]*4, object)
    terminal_node2 = TerminalNode("example", primitive_node2, "test")
    ind1 = Individual(content=[terminal_node1])
    ind2 = Individual(content=[terminal_node2])

    ind1.fitness.values = (1.0,)
    ind2.fitness.values = (1.0,)

    assert ind1 != ind2

def test_individual_equal_error():
    primitive_node = Primitive("example", ["arg"]*3, object)
    terminal_node = TerminalNode("example", primitive_node, "test")
    ind = Individual(content=[terminal_node])

    ind.fitness.values = (1.0,)

    with pytest.raises(ValueError, match="Cannot compare Individual with"):
        ind == 3
