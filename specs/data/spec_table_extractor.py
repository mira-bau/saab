"""Specs for TableExtractor - happy path only."""

from saab_v3.data.extractors import TableExtractor


# ============================================================================
# TableExtractor Basic Functionality
# ============================================================================


def spec_table_extractor_dataframe_basic(sample_dataframe):
    """Verify TableExtractor extracts tokens from simple DataFrame."""
    # Arrange
    extractor = TableExtractor()

    # Act
    tokens = extractor.extract(sample_dataframe)

    # Assert
    assert len(tokens) == 9  # 3 rows × 3 columns
    assert tokens[0].value == "Alice"
    assert tokens[0].structure_tag.field == "name"
    assert tokens[0].position == 0
    assert tokens[1].value == "25"
    assert tokens[1].structure_tag.field == "age"
    assert tokens[1].position == 1
    # Verify positions are sequential
    positions = [token.position for token in tokens]
    assert positions == list(range(9))


def spec_table_extractor_with_nan(sample_dataframe_with_nan):
    """Verify TableExtractor skips NaN values."""
    # Arrange
    extractor = TableExtractor()

    # Act
    tokens = extractor.extract(sample_dataframe_with_nan)

    # Assert
    # Should have fewer tokens than 9 (3 rows × 3 columns) due to NaN skipping
    assert len(tokens) < 9
    # Verify no empty values in tokens
    assert all(token.value for token in tokens)
    # Verify positions are still sequential
    positions = [token.position for token in tokens]
    assert positions == list(range(len(tokens)))


def spec_table_extractor_with_schema(sample_dataframe):
    """Verify TableExtractor handles schema with primary and foreign keys."""
    # Arrange
    extractor = TableExtractor()
    schema = {
        "primary_keys": ["name"],
        "foreign_keys": [{"column": "age", "references": "users.id"}],
    }

    # Act
    tokens = extractor.extract(sample_dataframe, schema=schema)

    # Assert
    # Find primary key token
    pk_token = next(t for t in tokens if t.structure_tag.field == "name")
    assert pk_token.structure_tag.role == "primary_key"
    # Find foreign key token
    fk_token = next(t for t in tokens if t.structure_tag.field == "age")
    assert fk_token.structure_tag.role == "foreign_key"
    assert fk_token.structure_tag.edge == "users.id"


def spec_table_extractor_can_handle(sample_dataframe):
    """Verify TableExtractor can_handle() detects correct formats."""
    # Arrange
    extractor = TableExtractor()

    # Act & Assert
    assert extractor.can_handle(sample_dataframe) is True
    assert extractor.can_handle("not a dataframe") is False
    assert extractor.can_handle(123) is False
    assert extractor.can_handle(None) is False
