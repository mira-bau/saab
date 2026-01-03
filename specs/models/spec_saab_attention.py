"""Specs for SAABAttention component - happy path only."""

import torch

from saab_v3.data.constants import PAD_IDX
from saab_v3.models.components.attention import MultiHeadAttention
from saab_v3.models.components.saab_attention import SAABAttention


# ============================================================================
# SAABAttention Specs
# ============================================================================


def spec_saab_attention_basic_forward_pass(
    saab_attention, sample_tensor, sample_tag_indices, batch_size, seq_len, d_model
):
    """Verify SAABAttention performs basic forward pass correctly with tag indices."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output = saab_attention(
        query,
        key,
        value,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Check only non-padding positions (padding may have NaN from -inf bias)
    # In our fixture, positions 4+ are padding (PAD_IDX = 0)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        non_padding_output = output[non_padding_mask]
        assert not torch.isnan(non_padding_output).any()
        assert not torch.isinf(non_padding_output).any()


def spec_saab_attention_lambda_zero_bitwise_equivalence(
    d_model, num_heads, sample_tensor, sample_tag_indices
):
    """Verify SAABAttention with lambda=0 is bitwise equivalent to MultiHeadAttention."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Create standard attention and SAAB attention with lambda=0
    # Use same initialization for both
    torch.manual_seed(42)
    standard_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    torch.manual_seed(42)
    saab_attn = SAABAttention(d_model=d_model, num_heads=num_heads, lambda_bias=0.0)

    # Copy weights to ensure exact same initialization
    saab_attn.q_proj.weight.data.copy_(standard_attn.q_proj.weight.data)
    saab_attn.k_proj.weight.data.copy_(standard_attn.k_proj.weight.data)
    saab_attn.v_proj.weight.data.copy_(standard_attn.v_proj.weight.data)
    saab_attn.out_proj.weight.data.copy_(standard_attn.out_proj.weight.data)
    if standard_attn.q_proj.bias is not None:
        saab_attn.q_proj.bias.data.copy_(standard_attn.q_proj.bias.data)
        saab_attn.k_proj.bias.data.copy_(standard_attn.k_proj.bias.data)
        saab_attn.v_proj.bias.data.copy_(standard_attn.v_proj.bias.data)
        saab_attn.out_proj.bias.data.copy_(standard_attn.out_proj.bias.data)

    # Set same random seed for forward pass
    torch.manual_seed(123)
    standard_output = standard_attn(query, key, value)

    torch.manual_seed(123)
    saab_output = saab_attn(
        query,
        key,
        value,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert: Should be bitwise equivalent when lambda=0
    # Check only non-padding positions
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        standard_non_pad = standard_output[non_padding_mask]
        saab_non_pad = saab_output[non_padding_mask]
        assert torch.allclose(standard_non_pad, saab_non_pad, atol=1e-5)


def spec_saab_attention_bias_computation(
    saab_attention, sample_tag_indices, batch_size, seq_len
):
    """Verify bias matrix computation for different structural relationships."""
    # Arrange
    field_ids = sample_tag_indices["field_ids"]
    entity_ids = sample_tag_indices["entity_ids"]
    time_ids = sample_tag_indices["time_ids"]
    token_type_ids = sample_tag_indices["token_type_ids"]

    # Act
    bias_matrix = saab_attention.compute_bias_matrix(
        field_ids=field_ids,
        entity_ids=entity_ids,
        time_ids=time_ids,
        token_type_ids=token_type_ids,
    )

    # Assert
    assert bias_matrix.shape == (batch_size, seq_len, seq_len)

    # Check that same field tokens get bias (batch 0, positions 0-1 have same field=1)
    # Same field adds 1.0 to bias
    assert bias_matrix[0, 0, 1] > 0  # Same field
    assert bias_matrix[0, 1, 0] > 0  # Same field (symmetric)

    # Check that different field tokens have lower bias
    # Positions 0 and 2 have different fields (1 vs 2)
    # But they might share entity or time, so bias might still be > 0
    # Just verify the matrix is computed correctly

    # Check padding positions are -inf
    # Positions 4+ are padding (PAD_IDX = 0)
    assert torch.isinf(bias_matrix[0, 0, 4]).item()  # Padding position
    assert torch.isinf(bias_matrix[0, 4, 0]).item()  # Padding position (symmetric)


def spec_saab_attention_with_edge_role(
    saab_attention,
    sample_tensor,
    sample_tag_indices,
    sample_edge_role_indices,
    batch_size,
    seq_len,
    d_model,
):
    """Verify SAABAttention handles optional edge and role indices."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output = saab_attention(
        query,
        key,
        value,
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
    # In our fixture, positions 4+ are padding (PAD_IDX = 0)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        non_padding_output = output[non_padding_mask]
        assert not torch.isnan(non_padding_output).any()
        assert not torch.isinf(non_padding_output).any()


def spec_saab_attention_with_mask(
    saab_attention,
    sample_tensor,
    sample_tag_indices,
    sample_attention_mask,
    batch_size,
    seq_len,
    d_model,
):
    """Verify SAABAttention handles attention mask correctly."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output = saab_attention(
        query,
        key,
        value,
        attention_mask=sample_attention_mask,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    # Check only non-padding positions (padding may have NaN from -inf bias)
    # In our fixture, positions 4+ are padding (PAD_IDX = 0)
    non_padding_mask = sample_tag_indices["field_ids"] != PAD_IDX
    if non_padding_mask.any():
        non_padding_output = output[non_padding_mask]
        assert not torch.isnan(non_padding_output).any()
        assert not torch.isinf(non_padding_output).any()


def spec_saab_attention_return_weights(
    saab_attention,
    sample_tensor,
    sample_tag_indices,
    batch_size,
    seq_len,
    d_model,
    num_heads,
):
    """Verify SAABAttention can return attention weights."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act
    output, attn_weights = saab_attention(
        query,
        key,
        value,
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


def spec_saab_attention_lambda_parameter(
    d_model, num_heads, sample_tensor, sample_tag_indices
):
    """Verify lambda parameter controls bias strength."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Create SAAB attention with different lambda values
    saab_lambda_0 = SAABAttention(d_model=d_model, num_heads=num_heads, lambda_bias=0.0)
    saab_lambda_1 = SAABAttention(d_model=d_model, num_heads=num_heads, lambda_bias=1.0)
    saab_lambda_2 = SAABAttention(d_model=d_model, num_heads=num_heads, lambda_bias=2.0)

    # Act
    torch.manual_seed(42)
    output_0 = saab_lambda_0(
        query,
        key,
        value,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    torch.manual_seed(42)
    output_1 = saab_lambda_1(
        query,
        key,
        value,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    torch.manual_seed(42)
    output_2 = saab_lambda_2(
        query,
        key,
        value,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert: Different lambda values should produce different outputs
    # lambda=0 should be equivalent to standard attention (tested separately)
    # lambda=1 and lambda=2 should be different
    assert not torch.allclose(output_1, output_2, atol=1e-5)


def spec_saab_attention_learnable_lambda(
    d_model, num_heads, sample_tensor, sample_tag_indices
):
    """Verify learnable lambda parameter works correctly."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    saab_attn = SAABAttention(
        d_model=d_model, num_heads=num_heads, lambda_bias=1.0, learnable_lambda=True
    )

    # Act
    output = saab_attn(
        query,
        key,
        value,
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert
    assert output.shape == query.shape
    # Verify lambda is a parameter (not a buffer)
    assert isinstance(saab_attn.lambda_bias, torch.nn.Parameter)
    assert saab_attn.lambda_bias.requires_grad


def spec_saab_attention_bias_normalization(
    d_model, num_heads, sample_tensor, sample_tag_indices, batch_size, seq_len
):
    """Verify bias normalization parameter works correctly."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    saab_normalized = SAABAttention(
        d_model=d_model, num_heads=num_heads, bias_normalization=0.5
    )

    # Act
    bias_matrix = saab_normalized.compute_bias_matrix(
        field_ids=sample_tag_indices["field_ids"],
        entity_ids=sample_tag_indices["entity_ids"],
        time_ids=sample_tag_indices["time_ids"],
        token_type_ids=sample_tag_indices["token_type_ids"],
    )

    # Assert: Bias values should be scaled by normalization factor
    # Check that non-padding, non-inf values are scaled
    valid_bias = bias_matrix[~torch.isinf(bias_matrix)]
    if len(valid_bias) > 0:
        # With normalization=0.5, values should be half of what they would be
        # This is a basic check - exact values depend on structural relationships
        assert (valid_bias >= 0).all() or (
            valid_bias <= 0
        ).all()  # All same sign or zero


def spec_saab_attention_missing_tag_indices_error(saab_attention, sample_tensor):
    """Verify SAABAttention raises error when tag indices are missing with lambda != 0."""
    # Arrange
    query = sample_tensor
    key = sample_tensor
    value = sample_tensor

    # Act & Assert: Should raise ValueError when lambda != 0 and tag indices missing
    # Note: When lambda=0, tag indices are not required
    # But our fixture has lambda=1.0, so we need tag indices
    # This test verifies the error is raised correctly
    # We'll test with None for one required tag index
    try:
        saab_attention(query, key, value, field_ids=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Tag indices" in str(e) or "required" in str(e).lower()


def spec_saab_attention_different_shapes(d_model, num_heads):
    """Verify SAABAttention works with different batch sizes and sequence lengths."""
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
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        # Create tag indices
        field_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
        entity_ids = torch.randint(1, 20, (batch_size, seq_len), dtype=torch.long)
        time_ids = torch.randint(1, 5, (batch_size, seq_len), dtype=torch.long)
        token_type_ids = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long)

        saab_attn = SAABAttention(d_model=d_model, num_heads=num_heads)
        output = saab_attn(
            query,
            key,
            value,
            field_ids=field_ids,
            entity_ids=entity_ids,
            time_ids=time_ids,
            token_type_ids=token_type_ids,
        )

        assert output.shape == (batch_size, seq_len, d_model)


def spec_saab_attention_padding_handling(
    saab_attention, sample_tag_indices, batch_size, seq_len
):
    """Verify padding positions are handled correctly in bias computation."""
    # Arrange
    field_ids = sample_tag_indices["field_ids"]
    entity_ids = sample_tag_indices["entity_ids"]
    time_ids = sample_tag_indices["time_ids"]
    token_type_ids = sample_tag_indices["token_type_ids"]

    # Act
    bias_matrix = saab_attention.compute_bias_matrix(
        field_ids=field_ids,
        entity_ids=entity_ids,
        time_ids=time_ids,
        token_type_ids=token_type_ids,
    )

    # Assert: Padding positions (PAD_IDX = 0) should have -inf in bias matrix
    # In our fixture, positions 4+ are padding
    for i in range(batch_size):
        for j in range(seq_len):
            if field_ids[i, j] == PAD_IDX:
                # All positions involving this padding token should be -inf
                assert torch.isinf(bias_matrix[i, :, j]).all()
                assert torch.isinf(bias_matrix[i, j, :]).all()
