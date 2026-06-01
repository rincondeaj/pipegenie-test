from pipegenie.grammar.parser import EvoflowXML, EvoflowEBNF, ParserFactory, Parser
from pathlib import Path
import importlib
import sys
import pytest

def test_parser_has_abstract_methods():
    abstract_methods = Parser.__abstractmethods__

    assert "parse" in abstract_methods
    assert "validate" in abstract_methods

def test_parser_inheritance():
    class AuxParser(Parser):
        """
        Auxiliar parser.
        """
        def parse(self, file_path):
            return "ROOT", [], [], []
        def validate(self):
            pass

    parser = AuxParser()

    root, algos, hparams, non_terms = parser.parse("fake_path")
    parser.validate()

    assert root == "ROOT"
    assert isinstance(algos, list)
    assert isinstance(hparams, list)
    assert isinstance(non_terms, list)

@pytest.fixture
def setup_grammar(request):
    grammar_file = request.param
    test_dir = Path(__file__).parent
    return test_dir.parent / "data" / grammar_file

@pytest.mark.parametrize("setup_grammar", ["test_grammar.xml"], indirect=True)
def test_evoflowxml_parser(setup_grammar):
    grammar_path = setup_grammar
    parser = EvoflowXML()

    root, agorithm_terms, hyperparameter_terms, non_terms = parser.parse(grammar_path)

    assert root == "workflow"
    assert agorithm_terms == [{"name": "minMaxScaler","code": "sklearn.preprocessing.MinMaxScaler","family": "scaler"},{"name": "maxAbsScaler","code": "sklearn.preprocessing.MaxAbsScaler","family": "scaler"},{"name": "decisionTree","code": "sklearn.tree.DecisionTreeClassifier","family": "tree"},{"name": "kNN","code": "sklearn.neighbors.KNeighborsClassifier","family": "neighbors"}]
    assert hyperparameter_terms == [{"name": "dt::criterion","type": "categorical","values": "gini;entropy","default": "gini","children": None},{"name": "dt::max_depth","type": "int","lower": "1","upper": "30","default": "10","children": None},{"name": "dt::min_samples_split","type": "int","lower": "2","upper": "20","default": "2","children": None},{"name": "dt::max_features","type": "float","lower": "0","upper": "1","default": "0.5","children": None},{"name": "dt::min_samples_leaf","type": "int","lower": "1","upper": "20","default": "1","children": None},{"name": "dt::class_weight","type": "categorical","values": "balanced","default": "None","children": None},{"name": "knn::n_neighbors","type": "int","log": "True","lower": "1","upper": "50","default": "1","children": None},{"name": "knn::weights","type": "categorical","values": "uniform;distance","default": "uniform","children": None},{"name": "knn::p","type": "int","lower": "1","upper": "2","default": "2","children": None}]
    assert non_terms == [{"name": "workflow","production": "classifier"},{"name": "workflow","production": "preprocessingBranch;classifier"},{"name": "preprocessingBranch","production": "preprocess"},{"name": "preprocessingBranch","production": "preprocessingBranch;preprocess"},{"name": "classifier","production": "kNN(kNN_hp)"},{"name": "classifier","production": "decisionTree(decisionTree_hp)"},{"name": "preprocess","production": "minMaxScaler"},{"name": "preprocess","production": "maxAbsScaler"},{"name": "kNN_hp","production": "knn::n_neighbors;knn::weights;knn::p"},{"name": "decisionTree_hp","production": "dt::criterion;dt::max_depth;dt::min_samples_split;dt::max_features;dt::min_samples_leaf;dt::class_weight"}]

@pytest.mark.parametrize("setup_grammar", ["test_grammar.ebnf"], indirect=True)
def test_ebnf_parser(setup_grammar):
    grammar_path = setup_grammar
    parser = EvoflowEBNF()

    root, agorithm_terms, hyperparameter_terms, non_terms = parser.parse(grammar_path)

    assert root == "workflow"
    assert agorithm_terms == [{"name": "minMaxScaler","code": "sklearn.preprocessing.MinMaxScaler","family": "scaler"},{"name": "maxAbsScaler","code": "sklearn.preprocessing.MaxAbsScaler","family": "scaler"},{"name": "decisionTree","code": "sklearn.tree.DecisionTreeClassifier","family": "tree"},{"name": "kNN","code": "sklearn.neighbors.KNeighborsClassifier","family": "neighbors"}]
    assert hyperparameter_terms == [{"name": "dt::criterion","type": "categorical","values": "gini;entropy","default": "gini","children": None},{"name": "dt::max_depth","type": "int","lower": "1","upper": "30","default": "10","children": None},{"name": "dt::min_samples_split","type": "int","lower": "2","upper": "20","default": "2","children": None},{"name": "dt::max_features","type": "float","lower": "0.0","upper": "1.0","default": "0.5","children": None},{"name": "dt::min_samples_leaf","type": "int","lower": "1","upper": "20","default": "1","children": None},{"name": "dt::class_weight","type": "categorical","values": "balanced","default": "None","children": None},{"name": "knn::n_neighbors","type": "int","log": "True","lower": "1","upper": "50","default": "1","children": None},{"name": "knn::weights","type": "categorical","values": "uniform;distance","default": "uniform","children": None},{"name": "knn::p","type": "int","lower": "1","upper": "2","default": "2","children": None}]
    assert non_terms == [{"name": "workflow","production": "classifier"},{"name": "workflow","production": "preprocessingBranch;classifier"},{"name": "preprocessingBranch","production": "preprocess"},{"name": "preprocessingBranch","production": "preprocessingBranch;preprocess"},{"name": "classifier","production": "kNN(kNN_hp)"},{"name": "classifier","production": "decisionTree(decisionTree_hp)"},{"name": "preprocess","production": "minMaxScaler"},{"name": "preprocess","production": "maxAbsScaler"},{"name": "kNN_hp","production": "knn::n_neighbors;knn::weights;knn::p"},{"name": "decisionTree_hp","production": "dt::criterion;dt::max_depth;dt::min_samples_split;dt::max_features;dt::min_samples_leaf;dt::class_weight"}]

def test_grammar_consistency():

    test_dir = Path(__file__).parent
    xml_grammar = test_dir.parent / "data" / "complete_grammar.xml"
    ebnf_grammar = test_dir.parent / "data" / "complete_grammar.ebnf"

    xml_parser = EvoflowXML()
    ebnf_parser = EvoflowEBNF()

    xml_root, xml_agorithm_terms, xml_hyperparameter_terms, xml_non_terms = xml_parser.parse(xml_grammar)
    ebnf_root, ebnf_agorithm_terms, ebnf_hyperparameter_terms, ebnf_non_terms = ebnf_parser.parse(ebnf_grammar)

    assert xml_root == ebnf_root
    assert xml_agorithm_terms == ebnf_agorithm_terms
    assert xml_hyperparameter_terms == ebnf_hyperparameter_terms
    assert xml_non_terms == ebnf_non_terms
    

def test_factory_registered_parsers():
    class AuxParser(Parser):
        """
        Auxiliar parser.
        """
        def parse(self, file_path):
            pass
        def validate(self):
            pass
        
    parser_factory = ParserFactory()
    parser_factory.register_parser("aux", AuxParser)
    assert parser_factory.registered_parsers() == {"evoflow-xml": EvoflowXML,"ebnf": EvoflowEBNF,"aux": AuxParser,}

def test_factory_register_parser_success():
    class AuxParser(Parser):
        """
        Auxiliar parser.
        """
        def parse(self, file_path):
            pass
        def validate(self):
            pass
        
    parser_factory = ParserFactory()
    parser_factory.register_parser("aux", AuxParser)
    aux_parser = parser_factory.get_parser("aux")

    assert isinstance(aux_parser, AuxParser)

def test_factory_register_parser_error():
    class AuxParser():
        """
        Auxiliar parser.
        """
        def parse(self, file_path):
            pass
        def validate(self):
            pass
        
    parser_factory = ParserFactory()

    with pytest.raises(ValueError, match="Parser must be a subclass of Parser"):
        parser_factory.register_parser("aux", AuxParser)

@pytest.mark.parametrize("setup_grammar", ["test_grammar.xml"], indirect=True)
def test_factory_detect_format_from_file(setup_grammar):
    grammar_path = setup_grammar
    parser_factory = ParserFactory()
    file_format = parser_factory.detect_format_from_file(grammar_path)

    assert file_format == "application/xml"

def test_factory_get_parser_xml_success():
    parser_factory = ParserFactory()
    parser = parser_factory.get_parser("evoflow-xml")
    assert isinstance(parser, EvoflowXML)

def test_factory_get_parser_ebnf_success():
    parser_factory = ParserFactory()
    parser = parser_factory.get_parser("ebnf")
    assert isinstance(parser, EvoflowEBNF)

def test_factory_get_parser_error():
    parser_factory = ParserFactory()

    with pytest.raises(ValueError, match="No parser registered for file format"):
        parser = parser_factory.get_parser("none")
