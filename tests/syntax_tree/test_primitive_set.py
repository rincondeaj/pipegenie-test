from pipegenie.syntax_tree._primitive_set import Primitive, Terminal, Ephemeral, PrimitiveSet
import pytest

def test_primitive():
    arity = 3
    args = [object] * arity
    primitive = Primitive("name", args, object)

    assert primitive.name == "name"
    assert primitive.arity == arity
    assert primitive.args == [object, object, object]
    assert primitive.ret == object
    assert primitive.seq == "name({0}, {1}, {2})"

def test_primitive_format():
    arity = 2
    args = [object] * arity
    primitive = Primitive("name", args, object)

    assert primitive.format(2, "hparam") == 'name(2, hparam)'

def test_primitive_eq_true():
    arity = 2
    args = [object] * arity
    primitive_1 = Primitive("name", args, object)

    arity = 2
    args = [object] * arity
    primitive_2 = Primitive("name", args, object)

    assert primitive_1 == primitive_2

def test_primitive_eq_false():
    arity = 2
    args = [object] * arity
    primitive_1 = Primitive("name", args, object)

    arity = 3
    args = [object] * arity
    primitive_2 = Primitive("name", args, object)

    assert primitive_1 != primitive_2

def test_primitive_eq_notimplemented():
    arity = 2
    args = [object] * arity
    primitive = Primitive("name", args, object)

    assert primitive.__eq__(3) is NotImplemented

def test_terminal():
    terminal = Terminal(3, object, symbolic=True)

    assert terminal.name == "3"
    assert terminal.arity == 0
    assert terminal.ret == object
    assert terminal.value == 3
    assert terminal.symbolic == True

def test_terminal_format():
    terminal = Terminal(3, object, symbolic=True)

    assert terminal.format() == '3'

def test_terminal_eq_true():
    terminal_1 = Terminal(3, object, symbolic=True)
    terminal_2 = Terminal(3, object, symbolic=True)

    assert terminal_1 == terminal_2

def test_terminal_eq_false():
    terminal_1 = Terminal(2, object, symbolic=True)
    terminal_2 = Terminal(3, object, symbolic=True)

    assert terminal_1 != terminal_2

def test_terminal_eq_notimplemented():
    terminal = Terminal(2, object, symbolic=True)

    assert terminal.__eq__(3) is NotImplemented

def test_primitiveset_arity_0():
    primitive_set = PrimitiveSet("name")

    assert not primitive_set.terminals
    assert not primitive_set.primitives
    assert not primitive_set.arguments
    assert primitive_set.context == {"__builtins__": None}
    assert not primitive_set.mapping 
    assert primitive_set.terms_count == 0
    assert primitive_set.prims_count == 0
    assert not primitive_set.ephemerals
    assert primitive_set.name == "name"
    assert primitive_set.ret == object
    assert not primitive_set.ins

def test_primitiveset_arity_not_0():
    arity = 3
    primitive_set = PrimitiveSet("name", arity)

    mapping = {}
    for i in range(arity):
        arg_str = f"ARG{i}"
        mapping[arg_str] = Terminal(arg_str, object, symbolic=True)

    assert len(primitive_set.terminals) == arity
    assert len(primitive_set.primitives) == arity
    assert len(primitive_set.arguments) == arity
    assert primitive_set.context == {"__builtins__": None}
    assert primitive_set.mapping == mapping
    assert primitive_set.terms_count == arity
    assert primitive_set.prims_count == 0
    assert not primitive_set.ephemerals
    assert primitive_set.name == "name"
    assert primitive_set.ret == object
    assert primitive_set.ins == [object, object, object]

def test_primitiveset_add_primitive_success():
    arity = 3
    args = [object] * arity
    primitive = Primitive("name", args, object)

    primitive_set = PrimitiveSet("name")
    primitive_set.add_primitive(primitive, arity, primitive.name)

    assert primitive_set.context == {"__builtins__": None,"name": primitive}
    assert primitive_set.prims_count == 1

def test_primitiveset_add_primitive_error_arity():
    arity = 3
    args = [object] * arity
    primitive = Primitive("name", args, object)

    primitive_set = PrimitiveSet("name")

    with pytest.raises(ValueError, match="Arity should be >= 1."):
        primitive_set.add_primitive(primitive, 0, primitive.name)

def test_primitiveset_add_primitive_error_context():
    arity = 3
    args = [object] * arity
    primitive = Primitive("name", args, object)

    primitive_set = PrimitiveSet("name")
    primitive_set.context["name"] = 3

    with pytest.raises(ValueError, match="Primitives are required to"):
        primitive_set.add_primitive(primitive, arity, primitive.name)
    

def test_primitiveset_add_ephemeral_success():
    def increment(a):
        return a + 1

    primitive_set = PrimitiveSet("name")
    primitive_set.add_ephemeral_constant("increment", increment)

    assert issubclass(primitive_set.ephemerals["increment"], Ephemeral)
    assert primitive_set.terms_count == 1

def test_primitiveset_add_ephemeral_warning():
    def increment(a):
        return a + 1

    primitive_set = PrimitiveSet("name")
    primitive_set.add_ephemeral_constant("increment", increment)
    with pytest.warns(UserWarning, match="Ephemeral constant"):
        primitive_set.add_ephemeral_constant("increment", increment)