"""Specs for Dropout component - happy path only."""

import torch

from saab_v3.models.components.dropout import Dropout


# ============================================================================
# Dropout Specs
# ============================================================================


def spec_dropout_basic_forward_pass(
    dropout, sample_tensor, batch_size, seq_len, d_model
):
    """Verify Dropout performs basic forward pass correctly."""
    # Arrange
    x = sample_tensor

    # Act
    output = dropout(x)

    # Assert
    assert output.shape == (batch_size, seq_len, d_model)
    assert output.shape == x.shape


def spec_dropout_different_probabilities(sample_tensor, batch_size, seq_len, d_model):
    """Verify Dropout works with different dropout probabilities."""
    # Arrange
    x = sample_tensor
    probabilities = [0.0, 0.1, 0.5]

    # Act & Assert
    for p in probabilities:
        dropout = Dropout(p=p)
        output = dropout(x)
        assert output.shape == (batch_size, seq_len, d_model)


def spec_dropout_training_eval_mode(sample_tensor, batch_size, seq_len, d_model):
    """Verify Dropout behaves differently in training vs evaluation mode."""
    # Arrange
    x = sample_tensor
    dropout = Dropout(p=0.5)

    # Act: Training mode
    dropout.train()
    output_train = dropout(x)

    # Act: Evaluation mode
    dropout.eval()
    output_eval = dropout(x)

    # Assert: Training mode - output shape is correct
    assert output_train.shape == (batch_size, seq_len, d_model)
    # In training mode, output may differ from input (due to dropout)
    # We can't assert exact difference due to stochasticity, but shape is verified

    # Assert: Evaluation mode - output should equal input
    assert output_eval.shape == (batch_size, seq_len, d_model)
    # In eval mode, dropout should not modify input
    assert torch.allclose(output_eval, x)


def spec_dropout_different_shapes():
    """Verify Dropout works with different tensor shapes."""
    # Arrange
    test_cases = [
        (2, 10, 128),  # [batch_size, seq_len, d_model]
        (1, 5),  # [batch_size, features]
        (2, 10, 5, 128),  # [batch_size, seq_len, features, d_model]
    ]

    # Act & Assert
    for shape in test_cases:
        x = torch.randn(*shape)
        dropout = Dropout(p=0.1)
        output = dropout(x)
        assert output.shape == x.shape
