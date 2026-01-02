"""Specs for Preprocessor class - happy path only."""

import pytest
import pandas as pd

from saab_v3.training.config import PreprocessingConfig
from saab_v3.training.preprocessor import Preprocessor
from saab_v3.data.structures import TokenizedSequence


def spec_preprocessor_initialization(sample_config):
    """Verify Preprocessor can be initialized with config."""
    # Arrange: sample_config fixture provides PreprocessingConfig

    # Act
    preprocessor = Preprocessor(sample_config)

    # Assert
    assert preprocessor.config == sample_config
    assert preprocessor._is_fitted is False
    assert preprocessor._selected_extractor is None


def spec_preprocessor_fit_table_data(sample_preprocessor, sample_dataframe):
    """Verify Preprocessor can fit on table data."""
    # Arrange: sample_preprocessor and sample_dataframe fixtures

    # Act
    sample_preprocessor.fit(sample_dataframe)

    # Assert
    assert sample_preprocessor._is_fitted is True
    assert sample_preprocessor._selected_extractor is not None
    assert sample_preprocessor.tokenizer._is_built is True
    assert sample_preprocessor.tag_encoder._is_built is True


def spec_preprocessor_transform_table_data(fitted_preprocessor, sample_dataframe):
    """Verify Preprocessor can transform table data."""
    # Arrange: fitted_preprocessor and sample_dataframe

    # Act
    sequences = fitted_preprocessor.transform(sample_dataframe)

    # Assert
    assert len(sequences) > 0
    assert all(isinstance(seq, TokenizedSequence) for seq in sequences)


def spec_preprocessor_encode_sequences(fitted_preprocessor, sample_dataframe):
    """Verify Preprocessor can encode sequences."""
    # Arrange
    sequences = fitted_preprocessor.transform(sample_dataframe)

    # Act
    encoded = fitted_preprocessor.encode(sequences)

    # Assert
    assert len(encoded) == len(sequences)
    assert all(isinstance(item, tuple) and len(item) == 3 for item in encoded)
    # Check tuple structure: (TokenizedSequence, list[int], list[EncodedTag])
    seq, token_ids, encoded_tags = encoded[0]
    assert isinstance(seq, TokenizedSequence)
    assert isinstance(token_ids, list)
    assert all(isinstance(tid, int) for tid in token_ids)
    assert isinstance(encoded_tags, list)
    assert len(encoded_tags) == len(token_ids)


def spec_preprocessor_auto_detect_format(sample_config):
    """Verify Preprocessor can auto-detect format."""
    # Arrange
    preprocessor = Preprocessor(sample_config)
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Act
    preprocessor.fit(df)

    # Assert
    assert preprocessor._selected_extractor is not None
    # Should have selected TableExtractor
    from saab_v3.data.extractors.table import TableExtractor

    assert isinstance(preprocessor._selected_extractor, TableExtractor)


def spec_preprocessor_extractor_type_override(sample_config):
    """Verify extractor_type config can override auto-detection."""
    # Arrange
    config = PreprocessingConfig(
        vocab_size=1000,
        max_seq_len=128,
        extractor_type="table",
    )
    preprocessor = Preprocessor(config)
    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Act
    preprocessor.fit(df)

    # Assert
    from saab_v3.data.extractors.table import TableExtractor

    assert isinstance(preprocessor._selected_extractor, TableExtractor)


def spec_preprocessor_fit_raises_if_already_fitted(
    fitted_preprocessor, sample_dataframe
):
    """Verify fit() raises ValueError if called multiple times."""
    # Arrange: fitted_preprocessor already fitted

    # Act & Assert
    with pytest.raises(ValueError, match="already fitted"):
        fitted_preprocessor.fit(sample_dataframe)


def spec_preprocessor_encode_raises_if_not_fitted(
    sample_preprocessor, sample_dataframe
):
    """Verify encode() raises ValueError if not fitted."""
    # Arrange: sample_preprocessor not fitted
    sequences = sample_preprocessor.transform(sample_dataframe)

    # Act & Assert
    with pytest.raises(ValueError, match="not fitted"):
        sample_preprocessor.encode(sequences)
