# SPDX-License-Identifier: MIT-HUMAINS-Attribution
#
# Copyright (c) 2024 HUMAINS Research Group (University of Córdoba, Spain).
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License – HUMAINS Research Group Attribution Variant
# For full license text, see the LICENSE file in the repository root.

"""
Parser module to handle the grammar parsing from different file formats.
"""

import sys
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from xml.etree.ElementTree import Element

if sys.version_info < (3, 13):
    # mimetypes.guess_type using filename parameter is deprecated in Python 3.13
    from mimetypes import guess_type as guess_file_type
else:
    from mimetypes import guess_file_type

class Parser(ABC):
    """
    Abstract class for grammar parsers.
    """

    @abstractmethod
    def parse(
        self,
        file_path: str,
    ) -> tuple[str, list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        """
        Parse the file content.

        Parameters
        ----------
        file_path : str
            The path to the file.

        Returns
        -------
        root_symbol : str
            The root symbol of the grammar.

        algorithm_terms : list of dict of str: str
            The algorithm terminals of the grammar.

        hyperparameter_terms : list of dict of str: str
            The hyperparameter terminals of the grammar.

        non_terms : list of dict of str: str
            The non-terminals of the grammar.
        """
        raise NotImplementedError("Method 'parse' must be implemented in subclass")

    @abstractmethod
    def validate(self) -> None:
        """
        Validate the parsed data.
        """
        raise NotImplementedError("Method 'validate' must be implemented in subclass")


class EvoflowEBNF(Parser):
    """
    Grammar parser for EBNF formatted grammars.
    """

    def parse(
        self,
        file_path: str,
    ) -> tuple[str, list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        """
        Parse the EBNF file content.

        Parameters
        ----------
        file_path : str
            The path to the EBNF file.

        Returns
        -------
        root_symbol : str
            The root symbol of the grammar.

        algorithm_terms : list of dict of str: str
            The algorithm terminals of the grammar.

        hyperparameter_terms : list of dict of str: str
            The hyperparameter terminals of the grammar.

        non_terms : list of dict of str: str
            The non-terminals of the grammar.
        """

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        root_symbol = None
        algorithm_terms = []
        hyperparameter_terms = []
        non_terms = []

        non_terminal_mode = False
        current_non_term = None

        for line in lines:
            # Root symbol
            if line.startswith("@root"):
                root_symbol = line.split()[1]
                continue

            # Algorithm terminals
            if line.startswith("@algorithm"):
                parts = line.split()
                algorithm_terms.append({
                    "name": parts[1],
                    "code": parts[2],
                    "family": parts[3]
                })
                continue

            # Hyperparameter terminals
            if line.startswith("@hyperparameter"):
                parts = line.split()
                hp_name = parts[1]
                hp_type = parts[2]
                hp_attrs = {"name": hp_name, "type": hp_type}
                if hp_type in ["tuple", "union"]:
                    children = []
                    childrens = " ".join(parts[3:]).replace("(","").replace(")","").split(",")
                    for child in childrens:
                        child_parts = child.split()
                        child_data = {"type": child_parts[0]}
                        for attr in child_parts[1:]:
                            if "=" in attr:
                                k, v = attr.split("=", 1)
                                if hp_type == "float":
                                    child_data[k] = v
                                else:
                                    child_data[k] = v
                            elif attr.lower() == "log":
                                child_data["log"] = "True"
                        children.append(child_data)
                else:
                    children = None
                    for attr in parts[3:]:
                        if "=" in attr:
                            k, v = attr.split("=", 1)
                            if hp_type == "float":
                                hp_attrs[k] = v
                            else:
                                hp_attrs[k] = v
                        elif attr.lower() == "log":
                            hp_attrs["log"] = "True"
                hp_attrs["children"] = children
                hyperparameter_terms.append(hp_attrs)
                continue

            # Non-terminal rules
            if re.match(r"^\w+\s*=", line):
                non_terminal_mode = True
                current_non_term = line.split("=")[0].strip()
                continue

            if non_terminal_mode and line.endswith(";"):
                # Continuation of non-terminal
                rule_content = line.replace('"', "").replace(" ", "")[:-1]
                non_terms.append({"name": current_non_term, "production": rule_content})
                current_non_term = None
                non_terminal_mode = False
            elif non_terminal_mode:
                rule_content = line.replace("|", "").replace(" ", "")
                non_terms.append({"name": current_non_term, "production": rule_content})

        return root_symbol, algorithm_terms, hyperparameter_terms, non_terms

    def validate(self) -> None:
        """
        Validate parsed EBNF grammar.
        """
        # TODO
        pass


class EvoflowXML(Parser):
    """
    Grammar parser for XML files using the Evoflow format.
    """

    def parse(
        self,
        file_path: str,
    ) -> tuple[str, list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        """
        Parse the XML file content.

        Parameters
        ----------
        file_path : str
            The path to the XML file.

        Returns
        -------
        root_symbol : str
            The root symbol of the grammar.

        algorithm_terms : list of dict of str: str
            The algorithm terminals of the grammar.

        hyperparameter_terms : list of dict of str: str
            The hyperparameter terminals of the grammar.

        non_terms : list of dict of str: str
            The non-terminals of the grammar.
        """
        def element_to_dict(element: 'Element') -> dict[str, object]:
            return {
                "tag": element.tag,
                "attributes": dict(element.attrib.items()),
                "children": [element_to_dict(child) for child in element]
                             if len(element) > 0 else None,
                "content": element.text.strip()
                             if element.text and element.text.strip() else None,
            }

        grammar = ET.parse(file_path).getroot()
        root = grammar.find('root-symbol').text.strip()

        non_terms = [element_to_dict(non_term)
                     for non_term in grammar.findall('non-terminals/non-terminal')]
        non_terms = [{
            "name": non_term["attributes"]["name"],
            "production": production["content"],
        } for non_term in non_terms for production in non_term["children"]]

        terms = grammar.findall('terminals/terminal')
        algorithm_terms = [element_to_dict(term) for term in terms
                           if term.get('code') is not None]
        algorithm_terms = [{
            "name": term["attributes"]["name"],
            "code": term["attributes"]["code"],
            "family": term["attributes"]["type"],
        } for term in algorithm_terms]
        hyperparameter_terms = [element_to_dict(term) for term in terms
                                if term.get('code') is None]
        hyperparameter_terms = [{
            "name": term["attributes"]["name"],
            **term["attributes"],
            "children": [{  # Tuple or union type handling
                **child["attributes"],
            } for child in term["children"]] if term["children"] else None,
        } for term in hyperparameter_terms]

        return root, algorithm_terms, hyperparameter_terms, non_terms

    def validate(self) -> None:
        """
        Validate the parsed XML data.
        """
        # TODO: Implement validation
        pass


class ParserFactory:
    """
    Factory class to create parser instances based on file format.
    """

    # Default parsers
    _registered_parsers: dict[str, type[Parser]] = {
        "evoflow-xml": EvoflowXML,
        "ebnf": EvoflowEBNF,
    }

    @staticmethod
    def registered_parsers() -> dict[str, type[Parser]]:
        """
        Get the registered parsers.

        Returns
        -------
        registered_parsers : dict of str: Parser type
            The registered parsers.
        """
        return ParserFactory._registered_parsers

    @staticmethod
    def register_parser(file_format: str, parser: type[Parser]) -> None:
        """
        Register a parser for a specific file format.

        Parameters
        ----------
        file_format : str
            The file format to register the parser for.
            Use Factory.detect_format_from_file to get the file format.

        parser : Parser type
            The parser class to register
        """
        if not issubclass(parser, Parser):
            raise ValueError("Parser must be a subclass of Parser")

        ParserFactory._registered_parsers[file_format] = parser

    @staticmethod
    def detect_format_from_file(file_path: str) -> 'Optional[str]':
        """
        Detect the file format.

        Parameters
        ----------
        file_path : str
            The path to the file.

        Returns
        -------
        file_format : str or None if not detected
            The detected file format.
        """
        return guess_file_type(file_path)[0]

    @staticmethod
    def get_parser(file_format: str) -> Parser:
        """
        Get a parser instance for the given file format.

        Parameters
        ----------
        file_format : str
            The file format to get the parser for.

        Returns
        -------
        parser : Parser
            The parser instance.
        """
        parser_class = ParserFactory._registered_parsers.get(file_format)

        if not parser_class:
            raise ValueError(f"No parser registered for file format '{file_format}'")

        return parser_class()
