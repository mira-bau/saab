"""Specs for BaseTaskHead - happy path only."""

import pytest
import torch
import torch.nn as nn

from saab_v3.tasks.base import BaseTaskHead
from saab_v3.tasks.pooling import CLSPooling, MeanPooling


# ============================================================================
# BaseTaskHead Abstract Class Specs
# ============================================================================


def spec_base_task_head_cannot_instantiate(sample_d_model):
    """Verify BaseTaskHead cannot be instantiated directly (abstract)."""
    # Act & Assert
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseTaskHead(d_model=sample_d_model)


# ============================================================================
# Concrete Implementation for Testing
# ============================================================================


class ConcreteTaskHead(BaseTaskHead):
    """Concrete implementation of BaseTaskHead for testing."""

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        pooling: nn.Module | None = None,
    ):
        """Initialize concrete task head."""
        super().__init__(d_model, hidden_dims, dropout, pooling)
        self.output_dim = output_dim

        # Build output layer
        if hidden_dims is not None:
            # MLP mode: output from last hidden layer
            input_dim = hidden_dims[-1]
        else:
            # Simple mode: output from d_model
            input_dim = d_model

        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Pool sequence
        seq_repr = self._pool_sequence(encoder_output, attention_mask)

        # Apply MLP if present
        if self.mlp is not None:
            seq_repr = self.mlp(seq_repr)

        # Apply output layer
        output = self.output_layer(seq_repr)

        return output


# ============================================================================
# BaseTaskHead MLP Builder Specs
# ============================================================================


def spec_base_task_head_mlp_builder_simple(sample_d_model):
    """Verify MLP builder creates correct architecture for simple case."""
    # Arrange
    head = ConcreteTaskHead(d_model=sample_d_model, output_dim=10, hidden_dims=[256])

    # Assert: MLP should have 3 layers: Linear, Dropout, ReLU
    assert head.mlp is not None
    assert len(head.mlp) == 3
    assert isinstance(head.mlp[0], nn.Linear)
    assert head.mlp[0].in_features == sample_d_model
    assert head.mlp[0].out_features == 256
    assert isinstance(head.mlp[1], nn.Dropout)
    assert isinstance(head.mlp[2], nn.ReLU)


def spec_base_task_head_mlp_builder_multi_layer(sample_d_model):
    """Verify MLP builder creates correct architecture for multi-layer case."""
    # Arrange
    head = ConcreteTaskHead(
        d_model=sample_d_model, output_dim=10, hidden_dims=[256, 128, 64]
    )

    # Assert: MLP should have 9 layers: 3 * (Linear, Dropout, ReLU)
    assert head.mlp is not None
    assert len(head.mlp) == 9

    # Check first layer
    assert isinstance(head.mlp[0], nn.Linear)
    assert head.mlp[0].in_features == sample_d_model
    assert head.mlp[0].out_features == 256

    # Check second layer
    assert isinstance(head.mlp[3], nn.Linear)
    assert head.mlp[3].in_features == 256
    assert head.mlp[3].out_features == 128

    # Check third layer
    assert isinstance(head.mlp[6], nn.Linear)
    assert head.mlp[6].in_features == 128
    assert head.mlp[6].out_features == 64


def spec_base_task_head_simple_mode(sample_d_model, sample_encoder_output):
    """Verify BaseTaskHead works in simple mode (hidden_dims=None)."""
    # Arrange
    head = ConcreteTaskHead(d_model=sample_d_model, output_dim=10, hidden_dims=None)

    # Assert: No MLP should be created
    assert head.mlp is None
    assert head.output_layer.in_features == sample_d_model

    # Act: Forward pass should work
    output = head(sample_encoder_output)

    # Assert: Output shape should be correct
    batch_size = sample_encoder_output.shape[0]
    assert output.shape == (batch_size, 10)


def spec_base_task_head_mlp_mode(sample_d_model, sample_encoder_output):
    """Verify BaseTaskHead works in MLP mode (hidden_dims provided)."""
    # Arrange
    head = ConcreteTaskHead(
        d_model=sample_d_model, output_dim=10, hidden_dims=[256, 128]
    )

    # Assert: MLP should be created
    assert head.mlp is not None
    assert head.output_layer.in_features == 128  # Last hidden dim

    # Act: Forward pass should work
    output = head(sample_encoder_output)

    # Assert: Output shape should be correct
    batch_size = sample_encoder_output.shape[0]
    assert output.shape == (batch_size, 10)


def spec_base_task_head_dropout_only_in_mlp_mode(sample_d_model):
    """Verify dropout is only used in MLP mode."""
    # Arrange: Simple mode
    head_simple = ConcreteTaskHead(
        d_model=sample_d_model, output_dim=10, hidden_dims=None, dropout=0.5
    )
    assert head_simple.mlp is None  # No MLP, so dropout not used

    # Arrange: MLP mode
    head_mlp = ConcreteTaskHead(
        d_model=sample_d_model, output_dim=10, hidden_dims=[256], dropout=0.5
    )
    assert head_mlp.mlp is not None
    # Check that dropout is in the MLP
    assert isinstance(head_mlp.mlp[1], nn.Dropout)
    assert head_mlp.mlp[1].p == 0.5


def spec_base_task_head_pooling_integration(
    sample_encoder_output, sample_attention_mask
):
    """Verify BaseTaskHead integrates with pooling strategies."""
    # Arrange: CLS pooling (default)
    head_cls = ConcreteTaskHead(
        d_model=sample_encoder_output.shape[2], output_dim=10, pooling=CLSPooling()
    )

    # Act
    output_cls = head_cls(sample_encoder_output, sample_attention_mask)

    # Arrange: Mean pooling
    head_mean = ConcreteTaskHead(
        d_model=sample_encoder_output.shape[2], output_dim=10, pooling=MeanPooling()
    )

    # Act
    output_mean = head_mean(sample_encoder_output, sample_attention_mask)

    # Assert: Both should produce correct output shapes
    batch_size = sample_encoder_output.shape[0]
    assert output_cls.shape == (batch_size, 10)
    assert output_mean.shape == (batch_size, 10)

    # Outputs should be different (different pooling strategies)
    assert not torch.allclose(output_cls, output_mean)


def spec_base_task_head_default_pooling(sample_encoder_output):
    """Verify BaseTaskHead defaults to CLSPooling if not specified."""
    # Arrange
    head = ConcreteTaskHead(
        d_model=sample_encoder_output.shape[2], output_dim=10, pooling=None
    )

    # Assert: Should default to CLSPooling
    assert isinstance(head.pooling, CLSPooling)


def spec_base_task_head_device_consistency(sample_encoder_output):
    """Verify BaseTaskHead works on different devices."""
    # Arrange
    head = ConcreteTaskHead(d_model=sample_encoder_output.shape[2], output_dim=10)

    # Test CPU
    output_cpu = head(sample_encoder_output)
    assert output_cpu.device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        encoder_output_cuda = sample_encoder_output.cuda()
        head_cuda = head.cuda()
        output_cuda = head_cuda(encoder_output_cuda)
        assert output_cuda.device.type == "cuda"
        assert torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-5)

    # Test MPS if available
    if torch.backends.mps.is_available():
        encoder_output_mps = sample_encoder_output.to("mps")
        head_mps = head.to("mps")
        output_mps = head_mps(encoder_output_mps)
        assert output_mps.device.type == "mps"
        assert torch.allclose(output_cpu, output_mps.cpu(), atol=1e-5)


def spec_base_task_head_pool_sequence_method(
    sample_encoder_output, sample_attention_mask
):
    """Verify _pool_sequence method works correctly."""
    # Arrange
    head = ConcreteTaskHead(d_model=sample_encoder_output.shape[2], output_dim=10)

    # Act
    pooled = head._pool_sequence(sample_encoder_output, sample_attention_mask)

    # Assert: Should use the pooling strategy
    expected = head.pooling(sample_encoder_output, sample_attention_mask)
    assert torch.allclose(pooled, expected)


def spec_base_task_head_forward_with_attention_mask(
    sample_encoder_output, sample_attention_mask
):
    """Verify forward pass works with attention mask."""
    # Arrange
    head = ConcreteTaskHead(d_model=sample_encoder_output.shape[2], output_dim=10)

    # Act
    output = head(sample_encoder_output, attention_mask=sample_attention_mask)

    # Assert: Should produce correct output
    batch_size = sample_encoder_output.shape[0]
    assert output.shape == (batch_size, 10)
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))


def spec_base_task_head_forward_without_attention_mask(sample_encoder_output):
    """Verify forward pass works without attention mask (for CLS pooling)."""
    # Arrange
    head = ConcreteTaskHead(d_model=sample_encoder_output.shape[2], output_dim=10)

    # Act
    output = head(sample_encoder_output, attention_mask=None)

    # Assert: Should produce correct output
    batch_size = sample_encoder_output.shape[0]
    assert output.shape == (batch_size, 10)
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))
