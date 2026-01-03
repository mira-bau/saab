"""Specs for model factory functions - happy path only."""

import pytest
import pandas as pd

from saab_v3.models.config import ModelConfig
from saab_v3.models.factory import (
    get_vocab_sizes,
    create_flat_transformer,
    create_scratch_transformer,
    create_saab_transformer,
)
from saab_v3.training.config import PreprocessingConfig
from saab_v3.training.preprocessor import Preprocessor


# ============================================================================
# Factory Function Specs
# ============================================================================


@pytest.fixture
def fitted_preprocessor():
    """Fitted Preprocessor for factory testing."""
    config = PreprocessingConfig(vocab_size=1000, max_seq_len=128)
    preprocessor = Preprocessor(config)
    # Create sample data
    sample_data = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 90.0, 88.5],
        }
    )
    preprocessor.fit(sample_data)
    return preprocessor


def spec_get_vocab_sizes(fitted_preprocessor):
    """Verify get_vocab_sizes extracts vocabulary sizes correctly."""
    # Arrange
    # Act
    vocab_sizes = get_vocab_sizes(fitted_preprocessor)

    # Assert
    assert isinstance(vocab_sizes, dict)
    assert "token_vocab_size" in vocab_sizes
    assert "token_type_vocab_size" in vocab_sizes
    assert "field_vocab_size" in vocab_sizes
    assert "entity_vocab_size" in vocab_sizes
    assert "time_vocab_size" in vocab_sizes

    # Verify all sizes are positive integers
    for key, value in vocab_sizes.items():
        assert isinstance(value, int)
        assert value > 0


def spec_get_vocab_sizes_unfitted_error():
    """Verify get_vocab_sizes raises error for unfitted preprocessor."""
    # Arrange
    config = PreprocessingConfig(vocab_size=1000, max_seq_len=128)
    unfitted_preprocessor = Preprocessor(config)

    # Act & Assert
    with pytest.raises(ValueError, match="Preprocessor must be fitted"):
        get_vocab_sizes(unfitted_preprocessor)


def spec_create_flat_transformer(fitted_preprocessor, sample_model_config):
    """Verify create_flat_transformer creates model correctly."""
    # Arrange
    # Act
    model = create_flat_transformer(fitted_preprocessor, sample_model_config)

    # Assert
    assert model is not None
    assert model.d_model == sample_model_config.d_model
    assert model.num_layers == sample_model_config.num_layers
    assert model.num_heads == sample_model_config.num_heads
    assert model.ffn_dim == sample_model_config.ffn_dim


def spec_create_flat_transformer_device_placement(
    fitted_preprocessor, sample_model_config
):
    """Verify create_flat_transformer places model on correct device."""
    # Arrange
    # Test with CPU
    config_cpu = ModelConfig(
        device="cpu", d_model=128, num_layers=2, num_heads=4, ffn_dim=512
    )
    model_cpu = create_flat_transformer(fitted_preprocessor, config_cpu)

    # Assert
    assert model_cpu.embeddings.token_embedding.embedding.weight.device.type == "cpu"

    # Test with auto (should detect available device)
    config_auto = ModelConfig(
        device="auto", d_model=128, num_layers=2, num_heads=4, ffn_dim=512
    )
    model_auto = create_flat_transformer(fitted_preprocessor, config_auto)

    # Assert: Should be on a valid device
    device_type = model_auto.embeddings.token_embedding.embedding.weight.device.type
    assert device_type in ["cpu", "cuda", "mps"]


def spec_create_scratch_transformer(fitted_preprocessor, sample_model_config):
    """Verify create_scratch_transformer creates model correctly."""
    # Arrange
    # Act
    model = create_scratch_transformer(fitted_preprocessor, sample_model_config)

    # Assert
    assert model is not None
    assert model.d_model == sample_model_config.d_model
    assert model.num_layers == sample_model_config.num_layers
    assert model.num_heads == sample_model_config.num_heads
    assert model.ffn_dim == sample_model_config.ffn_dim


def spec_create_scratch_transformer_device_placement(
    fitted_preprocessor, sample_model_config
):
    """Verify create_scratch_transformer places model on correct device."""
    # Arrange
    config_cpu = ModelConfig(
        device="cpu", d_model=128, num_layers=2, num_heads=4, ffn_dim=512
    )
    model_cpu = create_scratch_transformer(fitted_preprocessor, config_cpu)

    # Assert
    assert model_cpu.embeddings.token_embedding.embedding.weight.device.type == "cpu"


def spec_create_saab_transformer(fitted_preprocessor, sample_model_config):
    """Verify create_saab_transformer creates model correctly."""
    # Arrange
    # Act
    model = create_saab_transformer(fitted_preprocessor, sample_model_config)

    # Assert
    assert model is not None
    assert model.d_model == sample_model_config.d_model
    assert model.num_layers == sample_model_config.num_layers
    assert model.num_heads == sample_model_config.num_heads
    assert model.ffn_dim == sample_model_config.ffn_dim
    assert model.lambda_bias == sample_model_config.lambda_bias


def spec_create_saab_transformer_lambda_override(
    fitted_preprocessor, sample_model_config
):
    """Verify create_saab_transformer can override lambda_bias parameter."""
    # Arrange
    # Act
    model = create_saab_transformer(
        fitted_preprocessor, sample_model_config, lambda_bias=2.5
    )

    # Assert
    assert model.lambda_bias == 2.5
    # Verify it's different from config default
    assert model.lambda_bias != sample_model_config.lambda_bias


def spec_create_saab_transformer_device_placement(
    fitted_preprocessor, sample_model_config
):
    """Verify create_saab_transformer places model on correct device."""
    # Arrange
    config_cpu = ModelConfig(
        device="cpu", d_model=128, num_layers=2, num_heads=4, ffn_dim=512
    )
    model_cpu = create_saab_transformer(fitted_preprocessor, config_cpu)

    # Assert
    assert model_cpu.embeddings.token_embedding.embedding.weight.device.type == "cpu"


def spec_factory_models_forward_pass(fitted_preprocessor, sample_model_config):
    """Verify models created by factory functions can perform forward pass."""
    # Arrange
    # Create a batch with valid token IDs within vocabulary size
    from saab_v3.data.structures import Batch
    import torch

    vocab_size = len(fitted_preprocessor.tokenizer.vocab)
    batch_size, seq_len = 2, 10
    token_ids = torch.randint(
        0, min(vocab_size, 100), (batch_size, seq_len), dtype=torch.long
    )
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    field_ids = torch.randint(
        1,
        min(len(fitted_preprocessor.tag_encoder.tag_vocabs["field"]), 20),
        (batch_size, seq_len),
        dtype=torch.long,
    )
    entity_ids = torch.randint(
        1,
        min(len(fitted_preprocessor.tag_encoder.tag_vocabs["entity"]), 50),
        (batch_size, seq_len),
        dtype=torch.long,
    )
    time_ids = torch.randint(
        1,
        min(len(fitted_preprocessor.tag_encoder.tag_vocabs["time"]), 10),
        (batch_size, seq_len),
        dtype=torch.long,
    )
    token_type_ids = torch.randint(
        0,
        min(len(fitted_preprocessor.tag_encoder.tag_vocabs["token_type"]), 5),
        (batch_size, seq_len),
        dtype=torch.long,
    )

    test_batch = Batch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        field_ids=field_ids,
        entity_ids=entity_ids,
        time_ids=time_ids,
        token_type_ids=token_type_ids,
        sequence_lengths=[seq_len] * batch_size,
    )

    # Create smaller config to avoid issues
    small_config = ModelConfig(
        d_model=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        max_seq_len=128,
        device="cpu",
    )

    flat_model = create_flat_transformer(fitted_preprocessor, small_config)
    scratch_model = create_scratch_transformer(fitted_preprocessor, small_config)
    saab_model = create_saab_transformer(fitted_preprocessor, small_config)

    # Act
    flat_output = flat_model(test_batch)
    scratch_output = scratch_model(test_batch)
    saab_output = saab_model(test_batch)

    # Assert
    d_model = small_config.d_model

    assert flat_output.shape == (batch_size, seq_len, d_model)
    assert scratch_output.shape == (batch_size, seq_len, d_model)
    assert saab_output.shape == (batch_size, seq_len, d_model)

    # Check only non-padding positions (SAAB may have NaN in padding due to -inf bias)
    non_padding_mask = test_batch.field_ids != 0  # PAD_IDX = 0
    if non_padding_mask.any():
        batch_indices, seq_indices = torch.where(non_padding_mask)
        if len(batch_indices) > 0:
            flat_non_pad = flat_output[batch_indices, seq_indices, :]
            scratch_non_pad = scratch_output[batch_indices, seq_indices, :]
            saab_non_pad = saab_output[batch_indices, seq_indices, :]
            assert not torch.isnan(flat_non_pad).any()
            assert not torch.isnan(scratch_non_pad).any()
            assert not torch.isnan(saab_non_pad).any()


def spec_factory_device_consistency(fitted_preprocessor, sample_model_config):
    """Verify all factory functions place models on same device when config matches."""
    # Arrange
    config_cpu = ModelConfig(
        device="cpu", d_model=128, num_layers=2, num_heads=4, ffn_dim=512
    )

    # Act
    flat_model = create_flat_transformer(fitted_preprocessor, config_cpu)
    scratch_model = create_scratch_transformer(fitted_preprocessor, config_cpu)
    saab_model = create_saab_transformer(fitted_preprocessor, config_cpu)

    # Assert: All models should be on CPU
    flat_device = flat_model.embeddings.token_embedding.embedding.weight.device
    scratch_device = scratch_model.embeddings.token_embedding.embedding.weight.device
    saab_device = saab_model.embeddings.token_embedding.embedding.weight.device

    assert flat_device.type == "cpu"
    assert scratch_device.type == "cpu"
    assert saab_device.type == "cpu"
    assert flat_device == scratch_device == saab_device
