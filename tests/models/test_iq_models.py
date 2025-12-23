"""Tests for IQ models (XCiT1d and related components)."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from torchsig.models.iq_models.xcit import (
    XCiT1d,
    ConvDownSampler,
    Chunker,
    PositionalEncoding1D,
    FocalLoss,
)


class TestConvDownSampler:
    """Tests for ConvDownSampler class."""

    @pytest.mark.parametrize("in_chans,embed_dim,ds_rate", [
        (2, 64, 16),
        (1, 128, 8),
        (4, 256, 32),
    ])
    def test_output_shape(self, in_chans, embed_dim, ds_rate):
        """Test that output shape is correct after downsampling."""
        batch_size = 4
        seq_length = 1024

        model = ConvDownSampler(in_chans=in_chans, embed_dim=embed_dim, ds_rate=ds_rate)
        x = torch.randn(batch_size, in_chans, seq_length)

        output = model(x)

        expected_length = seq_length // ds_rate
        assert output.shape == (batch_size, embed_dim, expected_length), \
            f"Expected shape {(batch_size, embed_dim, expected_length)}, got {output.shape}"

    def test_forward_preserves_batch(self):
        """Test that batch dimension is preserved."""
        model = ConvDownSampler(in_chans=2, embed_dim=64, ds_rate=16)

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 2, 512)
            output = model(x)
            assert output.shape[0] == batch_size

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = ConvDownSampler(in_chans=2, embed_dim=64, ds_rate=16)
        x = torch.randn(4, 2, 512, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestChunker:
    """Tests for Chunker class."""

    @pytest.mark.parametrize("in_chans,embed_dim,ds_rate", [
        (2, 64, 16),
        (1, 128, 8),
        (4, 256, 32),
    ])
    def test_output_shape(self, in_chans, embed_dim, ds_rate):
        """Test that output shape is correct after chunking."""
        batch_size = 4
        seq_length = 1024

        model = Chunker(in_chans=in_chans, embed_dim=embed_dim, ds_rate=ds_rate)
        x = torch.randn(batch_size, in_chans, seq_length)

        output = model(x)

        expected_length = seq_length // ds_rate
        assert output.shape == (batch_size, embed_dim, expected_length), \
            f"Expected shape {(batch_size, embed_dim, expected_length)}, got {output.shape}"

    def test_forward_preserves_batch(self):
        """Test that batch dimension is preserved."""
        model = Chunker(in_chans=2, embed_dim=64, ds_rate=16)

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 2, 512)
            output = model(x)
            assert output.shape[0] == batch_size

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = Chunker(in_chans=2, embed_dim=64, ds_rate=16)
        x = torch.randn(4, 2, 512, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestPositionalEncoding1D:
    """Tests for PositionalEncoding1D class."""

    @pytest.mark.parametrize("embed_dim,seq_length", [
        (64, 128),
        (128, 256),
        (256, 512),
    ])
    def test_output_shape(self, embed_dim, seq_length):
        """Test that output shape matches input shape."""
        batch_size = 4

        model = PositionalEncoding1D(embed_dim=embed_dim)
        x = torch.randn(batch_size, seq_length, embed_dim)

        output = model(x)

        assert output.shape == x.shape, \
            f"Expected shape {x.shape}, got {output.shape}"

    def test_different_sequences_same_encoding(self):
        """Test that positional encodings are consistent."""
        embed_dim = 64
        model = PositionalEncoding1D(embed_dim=embed_dim)

        x1 = torch.randn(2, 100, embed_dim)
        x2 = torch.randn(2, 100, embed_dim)

        pe1 = model(x1)
        pe2 = model(x2)

        # The positional encoding should add the same pattern
        # So (pe1 - x1) should equal (pe2 - x2)
        diff1 = pe1 - x1
        diff2 = pe2 - x2
        assert torch.allclose(diff1, diff2, atol=1e-6)


class TestFocalLoss:
    """Tests for FocalLoss class."""

    def test_basic_loss_computation(self):
        """Test that loss can be computed."""
        loss_fn = FocalLoss(gamma=2.0)

        # Create sample inputs (batch_size=4, num_classes=10)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = loss_fn(inputs, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_perfect_prediction(self):
        """Test that loss is low for perfect predictions."""
        loss_fn = FocalLoss(gamma=2.0)

        # Create inputs that strongly favor the correct class
        inputs = torch.zeros(4, 10)
        targets = torch.tensor([0, 1, 2, 3])
        for i, t in enumerate(targets):
            inputs[i, t] = 10.0  # High logit for correct class

        loss = loss_fn(inputs, targets)

        # Loss should be very low for confident correct predictions
        assert loss < 0.1

    def test_wrong_prediction(self):
        """Test that loss is high for wrong predictions."""
        loss_fn = FocalLoss(gamma=2.0)

        # Create inputs that strongly favor the wrong class
        inputs = torch.zeros(4, 10)
        targets = torch.tensor([0, 1, 2, 3])
        for i, t in enumerate(targets):
            inputs[i, (t + 1) % 10] = 10.0  # High logit for wrong class

        loss = loss_fn(inputs, targets)

        # Loss should be high for confident wrong predictions
        assert loss > 1.0

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_different_gamma_values(self, gamma):
        """Test that different gamma values work."""
        loss_fn = FocalLoss(gamma=gamma)

        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = loss_fn(inputs, targets)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss_mean = FocalLoss(gamma=2.0, reduction='mean')
        loss_sum = FocalLoss(gamma=2.0, reduction='sum')

        mean_val = loss_mean(inputs, targets)
        sum_val = loss_sum(inputs, targets)

        # Sum should be greater than mean for batch > 1
        assert sum_val > mean_val

    def test_gradient_flow(self):
        """Test that gradients flow through focal loss."""
        loss_fn = FocalLoss(gamma=2.0)

        inputs = torch.randn(4, 10, requires_grad=True)
        targets = torch.randint(0, 10, (4,))

        loss = loss_fn(inputs, targets)
        loss.backward()

        assert inputs.grad is not None
        assert not torch.all(inputs.grad == 0)


class TestXCiT1d:
    """Tests for XCiT1d model."""

    @pytest.mark.slow
    @pytest.mark.parametrize("ds_method", ["downsample", "chunk"])
    def test_output_shape(self, ds_method):
        """Test that output shape is correct."""
        batch_size = 2
        input_channels = 2
        n_features = 10
        seq_length = 1024

        model = XCiT1d(
            input_channels=input_channels,
            n_features=n_features,
            xcit_version="nano_12_p16_224",
            ds_method=ds_method,
            ds_rate=16
        )
        model.eval()

        x = torch.randn(batch_size, input_channels, seq_length)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, n_features), \
            f"Expected shape {(batch_size, n_features)}, got {output.shape}"

    @pytest.mark.slow
    def test_different_sequence_lengths(self):
        """Test that model handles different sequence lengths."""
        model = XCiT1d(
            input_channels=2,
            n_features=10,
            xcit_version="nano_12_p16_224",
            ds_rate=16
        )
        model.eval()

        for seq_length in [512, 1024, 2048]:
            x = torch.randn(2, 2, seq_length)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 10)

    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = XCiT1d(
            input_channels=2,
            n_features=10,
            xcit_version="nano_12_p16_224",
            ds_rate=16
        )

        x = torch.randn(2, 2, 1024, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    @pytest.mark.slow
    def test_eval_mode(self):
        """Test model behavior in eval mode."""
        model = XCiT1d(
            input_channels=2,
            n_features=10,
            xcit_version="nano_12_p16_224"
        )

        x = torch.randn(2, 2, 1024)

        # Train mode
        model.train()
        out_train = model(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(x)

        # Outputs may differ due to dropout
        assert out_train.shape == out_eval.shape

    @pytest.mark.slow
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = XCiT1d(
            input_channels=2,
            n_features=10,
            xcit_version="nano_12_p16_224"
        )

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert num_params > 0
        assert num_trainable > 0
        assert num_trainable == num_params  # All params should be trainable


class TestXCiT1dInputValidation:
    """Tests for XCiT1d input validation."""

    @pytest.mark.slow
    def test_single_channel_input(self):
        """Test model with single channel input."""
        model = XCiT1d(
            input_channels=1,
            n_features=5,
            xcit_version="nano_12_p16_224"
        )
        model.eval()

        x = torch.randn(2, 1, 1024)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 5)

    @pytest.mark.slow
    def test_multi_channel_input(self):
        """Test model with multiple channel input."""
        model = XCiT1d(
            input_channels=4,
            n_features=20,
            xcit_version="nano_12_p16_224"
        )
        model.eval()

        x = torch.randn(2, 4, 1024)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 20)
