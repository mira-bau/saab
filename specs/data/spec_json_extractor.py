"""Specs for JSONExtractor - happy path only."""

import json

from saab_v3.data.extractors import JSONExtractor


# ============================================================================
# JSONExtractor Basic Functionality
# ============================================================================


def spec_json_extractor_dict_basic(sample_json_dict):
    """Verify JSONExtractor extracts tokens from simple dict."""
    # Arrange
    extractor = JSONExtractor()

    # Act
    tokens = extractor.extract(sample_json_dict)

    # Assert
    assert len(tokens) == 3
    assert tokens[0].value == "Alice"
    assert tokens[0].structure_tag.field == "name"
    assert tokens[0].position == 0
    assert tokens[1].value == "25"
    assert tokens[1].structure_tag.field == "age"
    assert tokens[1].position == 1
    # Verify positions are sequential
    positions = [token.position for token in tokens]
    assert positions == list(range(3))


def spec_json_extractor_nested(sample_json_nested):
    """Verify JSONExtractor handles nested structures and builds paths correctly."""
    # Arrange
    extractor = JSONExtractor()

    # Act
    tokens = extractor.extract(sample_json_nested)

    # Assert
    assert len(tokens) == 3  # name, city, zip
    # Verify path building
    fields = [token.structure_tag.field for token in tokens]
    assert "user.name" in fields
    assert "user.address.city" in fields
    assert "user.address.zip" in fields
    # Verify positions are sequential
    positions = [token.position for token in tokens]
    assert positions == list(range(3))


def spec_json_extractor_list():
    """Verify JSONExtractor handles lists and array indices in paths."""
    # Arrange
    extractor = JSONExtractor()
    data = ["apple", "banana", "cherry"]

    # Act
    tokens = extractor.extract(data)

    # Assert
    assert len(tokens) == 3
    assert tokens[0].structure_tag.field == "[0]"
    assert tokens[1].structure_tag.field == "[1]"
    assert tokens[2].structure_tag.field == "[2]"
    # Verify positions are sequential
    positions = [token.position for token in tokens]
    assert positions == list(range(3))


def spec_json_extractor_json_string(sample_json_dict):
    """Verify JSONExtractor parses and extracts from JSON string."""
    # Arrange
    extractor = JSONExtractor()
    json_string = json.dumps(sample_json_dict)

    # Act
    tokens = extractor.extract(json_string)

    # Assert
    assert len(tokens) == 3
    assert tokens[0].value == "Alice"
    assert tokens[0].structure_tag.field == "name"
    # Verify positions are sequential
    positions = [token.position for token in tokens]
    assert positions == list(range(3))


def spec_json_extractor_can_handle(sample_json_dict):
    """Verify JSONExtractor can_handle() detects correct formats."""
    # Arrange
    extractor = JSONExtractor()

    # Act & Assert
    assert extractor.can_handle(sample_json_dict) is True
    assert extractor.can_handle([1, 2, 3]) is True
    assert extractor.can_handle('{"key": "value"}') is True
    assert extractor.can_handle("not valid json") is False
    assert extractor.can_handle(123) is False
    assert extractor.can_handle(None) is False
