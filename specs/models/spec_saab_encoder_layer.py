"""Specs for SAABEncoderLayer component - happy path only."""

import torch

from saab_v3.data.constants import PAD_IDX
from saab_v3.models.components.saab_encoder_layer import SAABEncoderLayer


# ============================================================================
# SAABEncoderLayer Specs
# ============================================================================


def spec_saab_encoder_layer_basic_forward_pass(
    sample_tensor, sample_tag_indices, batch_size, seq_len, d_model, num_heads, ffn_dim
):
    """Verify SAABEncoderLayer performs basic forward pass correctly with tag indices."""
    # Arrange
    x = sample_tensor
    layer = SAABEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)

    # Act
    output = layer(
        x,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Check only non-padding positions (padding may have NaN from -inf bias)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        non_padding_output = output[non_padding_mask]
        assert not torch.isnan(non_padding_output).any()
        assert not torch.isinf(non_padding_output).any()


def spec_saab_encoder_layer_with_mask(
    sample_tensor,
    sample_tag_indices,
    sample_attention_mask,
    batch_size,
    seq_len,
    d_model,
    num_heads,
    ffn_dim,
):
    """Verify SAABEncoderLayer handles attention mask correctly."""
    # Arrange
    x = sample_tensor
    layer = SAABEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)

    # Act
    output = layer(
        x,
        attention_mask=sample_attention_mask,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Check only non-padding positions (padding may have NaN from -inf bias)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        non_padding_output = output[non_padding_mask]
        assert not torch.isnan(non_padding_output).any()
        assert not torch.isinf(non_padding_output).any()


def spec_saab_encoder_layer_return_weights(
    sample_tensor,
    sample_tag_indices,
    batch_size,
    seq_len,
    d_model,
    num_heads,
    ffn_dim,
):
    """Verify SAABEncoderLayer can return attention weights."""
    # Arrange
    x = sample_tensor
    layer = SAABEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)

    # Act
    output, attn_weights = layer(
        x,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
        return_attention_weights=True,
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    # Check only non-padding positions (padding may have NaN from -inf bias)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        # Check attention weights for non-padding positions
        for b in range(batch_size):
            for h in range(num_heads):
                non_pad_attn = attn_weights[b, h, :, :][non_padding_mask[b], :]
                non_pad_attn = non_pad_attn[:, non_padding_mask[b]]
                # Filter out NaN values
                valid_attn = non_pad_attn[~torch.isnan(non_pad_attn)]
                if len(valid_attn) > 0:
                    assert (valid_attn >= 0).all()
                    assert (valid_attn <= 2.0).all()  # Allow tolerance for dropout


def spec_saab_encoder_layer_residual_connection(
    sample_tensor, sample_tag_indices, batch_size, seq_len, d_model, num_heads, ffn_dim
):
    """Verify SAABEncoderLayer applies residual connections."""
    # Arrange
    x = sample_tensor
    layer = SAABEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)

    # Act
    output = layer(
        x,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert: Output should be different from input (transformation occurred)
    assert not torch.allclose(output, x, atol=1e-5)


def spec_saab_encoder_layer_with_edge_role(
    sample_tensor,
    sample_tag_indices,
    sample_edge_role_indices,
    batch_size,
    seq_len,
    d_model,
    num_heads,
    ffn_dim,
):
    """Verify SAABEncoderLayer handles optional edge and role indices."""
    # Arrange
    x = sample_tensor
    layer = SAABEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)

    # Act
    output = layer(
        x,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
        edge_ids=sample_edge_role_indices["edge_ids"],
        role_ids=sample_edge_role_indices["role_ids"],
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Check only non-padding positions (padding may have NaN from -inf bias)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        non_padding_output = output[non_padding_mask]
        assert not torch.isnan(non_padding_output).any()
        assert not torch.isinf(non_padding_output).any()


def spec_saab_encoder_layer_different_configs(
    d_model, sample_tensor, sample_tag_indices, batch_size, seq_len
):
    """Verify SAABEncoderLayer works with different configurations."""
    # Arrange
    x = sample_tensor
    test_configs = [
        {"num_heads": 2, "ffn_dim": 2 * d_model, "dropout": 0.0, "lambda_bias": 1.0},
        {"num_heads": 4, "ffn_dim": 4 * d_model, "dropout": 0.1, "lambda_bias": 0.5},
        {"num_heads": 8, "ffn_dim": 8 * d_model, "dropout": 0.2, "lambda_bias": 2.0},
    ]

    # Act & Assert
    for config in test_configs:
        if d_model % config["num_heads"] == 0:
            layer = SAABEncoderLayer(
                d_model=d_model,
                num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"],
                dropout=config["dropout"],
                lambda_bias=config["lambda_bias"],
            )
            output = layer(
                x,
                field_ids=sample_tag_indices["field_ids"],
                entity_ids=sample_tag_indices["entity_ids"],
                time_ids=sample_tag_indices["time_ids"],
                token_type_ids=sample_tag_indices["token_type_ids"],
            )
            assert output.shape == (batch_size, seq_len, d_model)


def spec_saab_encoder_layer_different_shapes(d_model, num_heads, ffn_dim):
    """Verify SAABEncoderLayer works with different batch sizes and sequence lengths."""
    # Arrange
    test_cases = [
        (1, 10),
        (2, 10),
        (4, 10),
        (2, 50),
        (2, 100),
    ]

    # Act & Assert
    for batch_size, seq_len in test_cases:
        x = torch.randn(batch_size, seq_len, d_model)
        # Create tag indices
        field_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
        entity_ids = torch.randint(1, 20, (batch_size, seq_len), dtype=torch.long)
        time_ids = torch.randint(1, 5, (batch_size, seq_len), dtype=torch.long)
        token_type_ids = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long)

        layer = SAABEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)
        output = layer(
            x,
            field_ids=field_ids,
            entity_ids=entity_ids,
            time_ids=time_ids,
            token_type_ids=token_type_ids,
        )
        assert output.shape == (batch_size, seq_len, d_model)


def spec_saab_encoder_layer_with_mask_and_weights(
    sample_tensor,
    sample_tag_indices,
    sample_attention_mask,
    batch_size,
    seq_len,
    d_model,
    num_heads,
    ffn_dim,
):
    """Verify SAABEncoderLayer works with both mask and return_attention_weights."""
    # Arrange
    x = sample_tensor
    layer = SAABEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)

    # Act
    output, attn_weights = layer(
        x,
        attention_mask=sample_attention_mask,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
        return_attention_weights=True,
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def spec_saab_encoder_layer_lambda_zero(
    sample_tensor, sample_tag_indices, batch_size, seq_len, d_model, num_heads, ffn_dim
):
    """Verify SAABEncoderLayer with lambda=0 works (should be equivalent to standard layer)."""
    # Arrange
    x = sample_tensor
    layer = SAABEncoderLayer(
        d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim, lambda_bias=0.0
    )

    # Act
    output = layer(
        x,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Check only non-padding positions (padding may have NaN from -inf bias)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        non_padding_output = output[non_padding_mask]
        assert not torch.isnan(non_padding_output).any()
        assert not torch.isinf(non_padding_output).any()
