import pytest
from pipegenie.syntax_tree._encoding import TerminalNode, NonTerminalNode, SyntaxTreeSchema, SyntaxTree, SyntaxTreePipeline
from pipegenie.syntax_tree._primitive_set import Primitive
from pipegenie.grammar import parse_pipe_grammar
from pathlib import Path

# pylint: disable=redefined-outer-name

def test_terminal_node():
    terminal_node = TerminalNode("example", "example.example")

    assert isinstance(terminal_node, TerminalNode)
    assert terminal_node.symbol == "example"
    assert terminal_node.code == "example.example"
    assert terminal_node.family is None
    assert terminal_node.arity == 0

def test_terminal_node_family():
    terminal_node = TerminalNode("example", "example.example", "test")

    assert isinstance(terminal_node, TerminalNode)
    assert terminal_node.symbol == "example"
    assert terminal_node.code == "example.example"
    assert terminal_node.family == "test"
    assert terminal_node.arity == 0

def test_non_terminal_node():
    non_terminal_node = NonTerminalNode("example", "a;b;c")

    assert isinstance(non_terminal_node, NonTerminalNode)
    assert non_terminal_node.symbol == "example"
    assert non_terminal_node.production == "a;b;c"
    assert non_terminal_node.prod_list ==  ["a", "b", "c"]
    assert non_terminal_node.arity == 3

@pytest.fixture
def setup_sintaxtree(request):
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

    nderiv = 5
    return SyntaxTreeSchema(root, nderiv, terms, non_terms).create_syntax_tree(), nderiv

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar.xml"], indirect=True)
def test_sintaxtree(setup_sintaxtree):
    tree, _ = setup_sintaxtree

    assert isinstance(tree, list)
    assert all(isinstance(x, (TerminalNode, NonTerminalNode)) for x in tree)

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar.xml"], indirect=True)
def test_sintaxtree_search_subtree_slice(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    subtree_slice = tree.search_subtree(2)

    assert isinstance(subtree_slice, slice)
    assert subtree_slice.start <= subtree_slice.stop
    assert subtree_slice.stop <= len(tree)

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar.xml"], indirect=True)
def test_sintaxtree_getitem_index(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    subtree = tree[0]

    assert isinstance(subtree, (TerminalNode, NonTerminalNode))

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar.xml"], indirect=True)
def test_sintaxtree_getitem_slice(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    subtree = tree[tree.search_subtree(2)]

    assert isinstance(subtree, SyntaxTree)

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar.xml"], indirect=True)
def test_sintaxtree_getitem_typeerror(setup_sintaxtree):
    tree, _ = setup_sintaxtree

    with pytest.raises(TypeError, match="Invalid key type:"):
        _ = tree["1"]

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar.xml"], indirect=True)
def test_sintaxtree_to_graph(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    nodes, edges = tree.to_graph()

    assert all(u in nodes and v in nodes for u, v in edges)
    assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)
    assert all(u != v for u, v in edges)

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar.xml"], indirect=True)
def test_sintaxtree_to_graph_no_self_loops(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    _, edges = tree.to_graph()

    assert all(u != v for u, v in edges)

@pytest.fixture
def setup_sintaxtreeschema(request):
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

    nderiv = 5
    return SyntaxTreeSchema(root, nderiv, terms, non_terms), nderiv

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar.xml"], indirect=True)
def test_sintaxtreeschema(setup_sintaxtreeschema):
    schema, nderiv = setup_sintaxtreeschema

    assert isinstance(schema.terminals_map, dict)
    assert list(schema.terminals_map.keys()).sort() == ["minMaxScaler", "maxAbsScaler", "decisionTree", "dt::criterion", "dt::max_depth", "dt::min_samples_split", "dt::max_features", "dt::min_samples_leaf", "dt::class_weight", "kNN", "knn::n_neighbors", "knn::weights", "knn::p"].sort()
    assert list(schema.non_terminals_map.keys()).sort() == ["workflow", "preprocessingBranch", "classifier", "preprocess", "kNN_hp", "decisionTree_hp"].sort()
    assert list(schema.cardinality_map.keys()).sort() == ["classifier", "preprocessingBranch;classifier", "preprocess", "preprocessingBranch;preprocess", "kNN(kNN_hp)", "decisionTree(decisionTree_hp)", "minMaxScaler", "maxAbsScaler", "knn::n_neighbors;knn::weights;knn::p", "dt::criterion;dt::max_depth;dt::min_samples_split;dt::max_features;dt::min_samples_leaf;dt::class_weight"].sort()
    assert all(value == [-1] * (nderiv + 1) for value in schema.cardinality_map.values())

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_zero_preprocess.xml"], indirect=True)
def test_sintaxtreeschema_create_sintax_tree_zero_preprocess(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    tree_list = str(schema.create_syntax_tree()).split(" ")

    assert tree_list == ["workflow", "classifier", "kNN", "kNN_hp", "knn::n_neighbors", "knn::weights", "knn::p"]

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_one_preprocess.xml"], indirect=True)
def test_sintaxtreeschema_create_sintax_tree_one_preprocess(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    tree_list = str(schema.create_syntax_tree()).split(" ")

    assert tree_list == ["workflow", "preprocessingBranch", "preprocess", "minMaxScaler", "classifier", "kNN", "kNN_hp", "knn::n_neighbors", "knn::weights", "knn::p"]

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_multiple_preprocesses.xml"], indirect=True)
def test_sintaxtreeschema_create_sintax_tree_multiple_preprocesses(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    tree_list = str(schema.create_syntax_tree()).split(" ")

    assert any(item in ["minMaxScaler", "maxAbsScaler"] for item in tree_list)

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_one_classifier.xml"], indirect=True)
def test_sintaxtreeschema_create_sintax_tree_one_classifier(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    tree_list = str(schema.create_syntax_tree()).split(" ")

    assert all(item in tree_list for item in ["kNN", "kNN_hp", "knn::n_neighbors", "knn::weights", "knn::p"])

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar.xml"], indirect=True)
def test_sintaxtreeschema_create_sintax_tree_multiple_classifier(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    tree_list = str(schema.create_syntax_tree()).split(" ")

    assert any(item in ["decisionTree", "kNN"] for item in tree_list)

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar.xml"], indirect=True)
def test_sintaxtreeschema_fill_tree_branch(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    tree = SyntaxTree([])

    schema.fill_tree_branch(tree, schema.root, number_of_derivs=3)
    tree = str(tree).split(" ")

    assert any(item in ["decisionTree", "kNN"] for item in tree)

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_one_min_deriv.xml"], indirect=True)
def test_sintaxtreeschema_min_derivations_one(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema

    assert schema.min_derivations(schema.root) == 1

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_three_min_deriv.xml"], indirect=True)
def test_sintaxtreeschema_min_derivations_three(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema

    assert schema.min_derivations(schema.root) == 3

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_five_min_deriv.xml"], indirect=True)
def test_sintaxtreeschema_min_derivations_five(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema

    assert schema.min_derivations(schema.root) == 5

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_one_max_deriv.xml"], indirect=True)
def test_sintaxtreeschema_max_derivations_one(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema

    assert schema.max_derivations(schema.root) == 1

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_three_max_deriv.xml"], indirect=True)
def test_sintaxtreeschema_max_derivations_three(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema

    assert schema.max_derivations(schema.root) == 3

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_nderiv_max_deriv.xml"], indirect=True)
def test_sintaxtreeschema_max_derivations_nderiv(setup_sintaxtreeschema):
    schema, nderiv = setup_sintaxtreeschema

    assert schema.max_derivations(schema.root) == nderiv

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_recursive.xml"], indirect=True)
def test_sintaxtreeschema_is_recursive_true(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema

    assert schema.is_recursive(schema.root) == True

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar_non_recursive.xml"], indirect=True)
def test_sintaxtreeschema_is_recursive_false(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema

    assert schema.is_recursive(schema.root) == False

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar.xml"], indirect=True)
def test_sintaxtreeschema_get_parent_symbols_empty(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    
    assert schema.get_parent_symbols(schema.root) == []

@pytest.mark.parametrize("setup_sintaxtreeschema", ["test_grammar.xml"], indirect=True)
def test_sintaxtreeschema_get_parent_symbols_not_empty(setup_sintaxtreeschema):
    schema, _ = setup_sintaxtreeschema
    
    assert schema.get_parent_symbols("classifier") == ["workflow"]

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar_fixed.xml"], indirect=True)
def test_sintaxtreepipeline_str(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    pipeline = SyntaxTreePipeline(tree)

    assert str(pipeline) == "minMaxScaler();kNN(1,'uniform',2)"

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar_fixed.xml"], indirect=True)
def test_sintaxtreepipeline_reset(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    pipeline = SyntaxTreePipeline(tree)

    pipeline.pipeline = object()
    _ = str(pipeline)

    pipeline.reset()

    assert pipeline.pipeline is None

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar_fixed.xml"], indirect=True)
def test_sintaxtreepipeline_equal_true(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    pipeline1 = SyntaxTreePipeline(tree)
    pipeline2 = SyntaxTreePipeline(tree)

    assert pipeline1 == pipeline2

@pytest.mark.parametrize("setup_sintaxtree", ["test_grammar_fixed.xml"], indirect=True)
def test_sintaxtreepipeline_equal_false(setup_sintaxtree):
    tree, _ = setup_sintaxtree
    pipeline1 = SyntaxTreePipeline(tree)
    pipeline2 = SyntaxTreePipeline(tree[tree.search_subtree(1)])

    assert pipeline1 != pipeline2

def test_sintaxtreepipeline_equal_error():
    pipeline = SyntaxTreePipeline([])

    with pytest.raises(ValueError):
        pipeline == 3

@pytest.fixture
def setup_sintaxtreepipeline(request):
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

    nderiv = 5
    return SyntaxTreeSchema(root, nderiv, terms, non_terms).create_syntax_tree(), pset

@pytest.mark.parametrize("setup_sintaxtreepipeline", ["test_grammar_fixed.xml"], indirect=True)
def test_sintaxtreepipeline_create_pipeline_success(setup_sintaxtreepipeline):
    tree, pset = setup_sintaxtreepipeline
    pipeline = SyntaxTreePipeline(tree)

    pipeline.create_pipeline(pset)

    assert pipeline.pipeline is not None
    assert len(pipeline.pipeline.steps) == 2

@pytest.mark.parametrize("setup_sintaxtreepipeline", ["test_grammar_fixed.xml"], indirect=True)
def test_sintaxtreepipeline_create_pipeline_error(setup_sintaxtreepipeline):
    tree, _ = setup_sintaxtreepipeline
    pipeline = SyntaxTreePipeline(tree)

    with pytest.raises(ValueError):
        pipeline.create_pipeline({})
    assert pipeline.pipeline is None