"""Tests for model utilities (layer_tools, simple_models, general_layers)."""

import pytest
import torch
import torch.nn as nn

from torchsig.models.model_utils.layer_tools import (
    get_layer_list,
    replace_layer,
    is_same_type,
    replace_layers_on_condition,
    replace_layers_of_type,
)
from torchsig.models.model_utils.simple_models import (
    convnet_block_2d,
    convnet_block_1d,
    dense_block,
    simple_convnet_2d,
    simple_convnet_1d,
    simple_densenet,
    double_image_scale_2d,
)
from torchsig.models.model_utils.general_layers import (
    DebugPrintLayer,
    ScalingLayer,
    DropChannel,
    Reshape,
    Mean,
)


class TestLayerTools:
    """Tests for layer manipulation utilities."""

    def test_is_same_type_with_type(self):
        """Test is_same_type with type comparison."""
        conv = nn.Conv2d(3, 16, 3)
        linear = nn.Linear(10, 5)

        assert is_same_type(conv, nn.Conv2d)
        assert not is_same_type(conv, nn.Linear)
        assert is_same_type(linear, nn.Linear)
        assert not is_same_type(linear, nn.Conv2d)

    def test_is_same_type_with_string(self):
        """Test is_same_type with string comparison."""
        conv = nn.Conv2d(3, 16, 3)
        linear = nn.Linear(10, 5)

        assert is_same_type(conv, "Conv2d")
        assert not is_same_type(conv, "Linear")
        assert is_same_type(linear, "Linear")

    def test_is_same_type_with_instance(self):
        """Test is_same_type with instance comparison."""
        conv1 = nn.Conv2d(3, 16, 3)
        conv2 = nn.Conv2d(16, 32, 3)
        linear = nn.Linear(10, 5)

        assert is_same_type(conv1, conv2)
        assert not is_same_type(conv1, linear)

    def test_replace_layer_simple(self):
        """Test replacing a single layer in a model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
        )

        old_layer = model[0]
        new_layer = nn.Conv2d(3, 32, 5)

        result = replace_layer(old_layer, new_layer, model)

        assert result is True
        assert model[0] is new_layer

    def test_replace_layer_not_found(self):
        """Test replacing a layer that doesn't exist."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
        )

        old_layer = nn.Linear(10, 5)  # Not in model
        new_layer = nn.Conv2d(3, 32, 5)

        result = replace_layer(old_layer, new_layer, model)

        assert result is False

    def test_replace_layers_of_type(self):
        """Test replacing all layers of a specific type."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
        )

        def relu_to_gelu(layer):
            return nn.GELU()

        result = replace_layers_of_type(model, nn.ReLU, relu_to_gelu)

        assert result is True
        assert isinstance(model[1], nn.GELU)
        assert isinstance(model[3], nn.GELU)

    def test_replace_layers_on_condition(self):
        """Test replacing layers based on a condition."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Conv2d(16, 32, 3),
            nn.Conv2d(32, 64, 3),
        )

        # Replace convs with > 16 input channels
        def condition(layer):
            return isinstance(layer, nn.Conv2d) and layer.in_channels > 16

        def factory(layer):
            return nn.Conv2d(layer.in_channels, layer.out_channels, 5)

        result = replace_layers_on_condition(model, condition, factory)

        assert result is True
        assert model[0].kernel_size == (3, 3)  # Not replaced
        assert model[1].kernel_size == (5, 5)  # Replaced
        assert model[2].kernel_size == (5, 5)  # Replaced


class TestSimpleModels:
    """Tests for simple model building functions."""

    def test_convnet_block_2d(self):
        """Test 2D convolution block creation."""
        block = convnet_block_2d(in_width=3, out_width=16)

        assert isinstance(block, nn.Sequential)

        x = torch.randn(4, 3, 32, 32)
        output = block(x)

        assert output.shape == (4, 16, 28, 28)  # Default kernel 5x5 reduces size

    def test_convnet_block_2d_custom_kernel(self):
        """Test 2D convolution block with custom kernel."""
        block = convnet_block_2d(in_width=3, out_width=16, kernel_shape=[3, 3])

        x = torch.randn(4, 3, 32, 32)
        output = block(x)

        assert output.shape == (4, 16, 30, 30)  # Kernel 3x3 reduces by 2

    def test_convnet_block_1d(self):
        """Test 1D convolution block creation."""
        block = convnet_block_1d(in_width=2, out_width=16)

        assert isinstance(block, nn.Sequential)

        x = torch.randn(4, 2, 128)
        output = block(x)

        assert output.shape[0] == 4
        assert output.shape[1] == 16

    def test_dense_block(self):
        """Test dense block creation."""
        block = dense_block(in_width=64, out_width=32)

        assert isinstance(block, nn.Sequential)

        x = torch.randn(4, 64)
        output = block(x)

        assert output.shape == (4, 32)

    def test_simple_convnet_2d(self):
        """Test simple 2D CNN creation."""
        model = simple_convnet_2d([3, 16, 32, 64])

        assert isinstance(model, nn.Sequential)

        x = torch.randn(4, 3, 64, 64)
        output = model(x)

        assert output.shape[1] == 64

    def test_simple_convnet_1d(self):
        """Test simple 1D CNN creation."""
        model = simple_convnet_1d([2, 16, 32, 64])

        assert isinstance(model, nn.Sequential)

        x = torch.randn(4, 2, 256)
        output = model(x)

        assert output.shape[1] == 64

    def test_simple_densenet(self):
        """Test simple dense network creation."""
        model = simple_densenet([128, 64, 32, 10])

        assert isinstance(model, nn.Sequential)

        x = torch.randn(4, 128)
        output = model(x)

        assert output.shape == (4, 10)

    def test_double_image_scale_2d(self):
        """Test image upscaling layer."""
        layer = double_image_scale_2d(width=16)

        x = torch.randn(4, 16, 32, 32)
        output = layer(x)

        # Should approximately double the size
        assert output.shape[2] > x.shape[2]
        assert output.shape[3] > x.shape[3]


class TestGeneralLayers:
    """Tests for general-purpose layers."""

    def test_debug_print_layer(self):
        """Test that debug print layer passes through unchanged."""
        layer = DebugPrintLayer()

        x = torch.randn(4, 3, 32, 32)
        output = layer(x)

        assert torch.equal(x, output)

    def test_scaling_layer(self):
        """Test scaling layer."""
        scale = 2.5
        layer = ScalingLayer(scale_val=scale)

        x = torch.randn(4, 3, 32, 32)
        output = layer(x)

        assert torch.allclose(output, x * scale)

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 10.0])
    def test_scaling_layer_values(self, scale):
        """Test scaling layer with various scale values."""
        layer = ScalingLayer(scale_val=scale)

        x = torch.ones(4, 3, 32, 32)
        output = layer(x)

        assert torch.allclose(output, torch.full_like(x, scale))

    def test_drop_channel(self):
        """Test channel dropping layer."""
        layer = DropChannel()

        x = torch.randn(4, 5, 32, 32)
        output = layer(x)

        assert output.shape == (4, 4, 32, 32)
        assert torch.equal(output, x[:, :-1, :, :])

    def test_reshape_keep_batch(self):
        """Test reshape layer keeping batch dimension."""
        layer = Reshape(shape=(16, 16), keep_batch_dim=True)

        x = torch.randn(4, 256)
        output = layer(x)

        assert output.shape == (4, 16, 16)

    def test_reshape_no_batch(self):
        """Test reshape layer without keeping batch dimension."""
        layer = Reshape(shape=(4, 64), keep_batch_dim=False)

        x = torch.randn(4, 256)
        output = layer(x)

        assert output.shape == (4, 64)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_mean_layer(self, dim):
        """Test mean layer along different dimensions."""
        layer = Mean(dim=dim)

        x = torch.randn(4, 8, 16)
        output = layer(x)

        expected_shape = list(x.shape)
        expected_shape[dim] = 1
        # Mean reduces the dimension
        assert output.shape[dim] == 1


class TestGradientFlow:
    """Tests for gradient flow through layers."""

    def test_convnet_block_2d_gradient(self):
        """Test gradient flow through 2D conv block."""
        block = convnet_block_2d(in_width=3, out_width=16)

        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_simple_densenet_gradient(self):
        """Test gradient flow through dense network."""
        model = simple_densenet([128, 64, 32, 10])

        x = torch.randn(4, 128, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_scaling_layer_gradient(self):
        """Test gradient flow through scaling layer."""
        layer = ScalingLayer(scale_val=2.0)

        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.full_like(x, 2.0))


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_layer_list(self):
        """Test simple models with minimal layer counts."""
        model = simple_densenet([64, 32])  # Just 1 layer

        x = torch.randn(4, 64)
        output = model(x)

        assert output.shape == (4, 32)

    def test_single_element_batch(self):
        """Test layers with batch size of 1."""
        block = convnet_block_2d(in_width=3, out_width=16)

        x = torch.randn(1, 3, 32, 32)
        output = block(x)

        assert output.shape[0] == 1

    def test_large_batch(self):
        """Test layers with large batch size."""
        block = dense_block(in_width=64, out_width=32)

        x = torch.randn(128, 64)
        output = block(x)

        assert output.shape == (128, 32)
