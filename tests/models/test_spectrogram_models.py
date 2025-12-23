"""Tests for spectrogram models (EfficientNet2d and DETR)."""

import pytest
import torch
import torch.nn as nn

from torchsig.models.spectrogram_models.efficientnet import EfficientNet2d
from torchsig.models.spectrogram_models.detr import DETR
from torchsig.models.spectrogram_models.detr.modules import (
    HungarianMatcher,
    SetCriterion,
    MLP,
    create_detr,
)


class TestEfficientNet2d:
    """Tests for EfficientNet2d model."""

    @pytest.mark.slow
    @pytest.mark.parametrize("version", ["b0", "b1"])
    def test_output_shape(self, version):
        """Test that output shape is correct for different versions."""
        batch_size = 2
        input_channels = 1
        n_features = 10
        height, width = 224, 224

        model = EfficientNet2d(
            input_channels=input_channels,
            n_features=n_features,
            efficientnet_version=version
        )
        model.eval()

        x = torch.randn(batch_size, input_channels, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, n_features), \
            f"Expected shape {(batch_size, n_features)}, got {output.shape}"

    @pytest.mark.slow
    def test_different_input_channels(self):
        """Test model with different input channel counts."""
        for input_channels in [1, 2, 3]:
            model = EfficientNet2d(
                input_channels=input_channels,
                n_features=10,
                efficientnet_version="b0"
            )
            model.eval()

            x = torch.randn(2, input_channels, 224, 224)
            with torch.no_grad():
                output = model(x)

            assert output.shape == (2, 10)

    @pytest.mark.slow
    def test_different_image_sizes(self):
        """Test model with different input image sizes."""
        model = EfficientNet2d(
            input_channels=1,
            n_features=10,
            efficientnet_version="b0"
        )
        model.eval()

        for size in [128, 224, 256]:
            x = torch.randn(2, 1, size, size)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 10)

    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = EfficientNet2d(
            input_channels=1,
            n_features=10,
            efficientnet_version="b0"
        )

        x = torch.randn(2, 1, 224, 224, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    @pytest.mark.slow
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = EfficientNet2d(
            input_channels=1,
            n_features=10,
            efficientnet_version="b0"
        )

        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0


class TestMLP:
    """Tests for MLP class used in DETR."""

    @pytest.mark.parametrize("input_dim,hidden_dim,output_dim,num_layers", [
        (256, 512, 4, 3),
        (128, 256, 10, 2),
        (512, 1024, 1, 4),
    ])
    def test_output_shape(self, input_dim, hidden_dim, output_dim, num_layers):
        """Test that output shape is correct."""
        batch_size = 4
        seq_length = 50

        model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )

        x = torch.randn(batch_size, seq_length, input_dim)
        output = model(x)

        assert output.shape == (batch_size, seq_length, output_dim)

    def test_gradient_flow(self):
        """Test that gradients flow through MLP."""
        model = MLP(input_dim=256, hidden_dim=512, output_dim=4, num_layers=3)

        x = torch.randn(4, 50, 256, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestHungarianMatcher:
    """Tests for HungarianMatcher class."""

    def test_basic_matching(self):
        """Test that matcher produces valid indices."""
        matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)

        batch_size = 2
        num_queries = 10
        num_classes = 5

        outputs = {
            "pred_logits": torch.randn(batch_size, num_queries, num_classes + 1),
            "pred_boxes": torch.rand(batch_size, num_queries, 4)
        }

        targets = [
            {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([2, 3, 4]), "boxes": torch.rand(3, 4)},
        ]

        indices = matcher(outputs, targets)

        assert len(indices) == batch_size

        # Check that indices are valid
        for i, (src_idx, tgt_idx) in enumerate(indices):
            assert len(src_idx) == len(targets[i]["labels"])
            assert len(tgt_idx) == len(targets[i]["labels"])
            assert all(0 <= idx < num_queries for idx in src_idx)
            assert all(0 <= idx < len(targets[i]["labels"]) for idx in tgt_idx)

    def test_empty_targets(self):
        """Test matcher with empty targets."""
        matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)

        outputs = {
            "pred_logits": torch.randn(2, 10, 5),
            "pred_boxes": torch.rand(2, 10, 4)
        }

        targets = [
            {"labels": torch.tensor([]), "boxes": torch.zeros(0, 4)},
            {"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)},
        ]

        indices = matcher(outputs, targets)

        assert len(indices) == 2
        assert len(indices[0][0]) == 0  # Empty match for empty target
        assert len(indices[1][0]) == 1


class TestSetCriterion:
    """Tests for SetCriterion class (DETR loss)."""

    def test_loss_computation(self):
        """Test that loss can be computed."""
        num_classes = 5
        criterion = SetCriterion(num_classes=num_classes)

        batch_size = 2
        num_queries = 10

        outputs = {
            "pred_logits": torch.randn(batch_size, num_queries, num_classes + 1),
            "pred_boxes": torch.rand(batch_size, num_queries, 4)
        }

        targets = [
            {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([2]), "boxes": torch.rand(1, 4)},
        ]

        losses = criterion(outputs, targets)

        assert "loss_ce" in losses
        assert "loss_bbox" in losses
        assert "loss_giou" in losses

        for name, loss in losses.items():
            if "loss" in name:
                assert not torch.isnan(loss), f"{name} is NaN"
                assert not torch.isinf(loss), f"{name} is infinite"

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        criterion = SetCriterion(num_classes=5)

        outputs = {
            "pred_logits": torch.randn(2, 10, 6, requires_grad=True),
            "pred_boxes": torch.rand(2, 10, 4, requires_grad=True)
        }

        targets = [
            {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([2]), "boxes": torch.rand(1, 4)},
        ]

        losses = criterion(outputs, targets)
        total_loss = sum(losses[k] for k in losses if "loss" in k)
        total_loss.backward()

        assert outputs["pred_logits"].grad is not None
        assert outputs["pred_boxes"].grad is not None


class TestDETR:
    """Tests for DETR factory function and model."""

    @pytest.mark.slow
    @pytest.mark.parametrize("version", ["detr_b0_nano"])
    def test_output_structure(self, version):
        """Test that output has correct structure."""
        num_classes = 5

        model = DETR(version=version, num_classes=num_classes)
        model.eval()

        batch_size = 2
        height, width = 224, 224
        x = torch.randn(batch_size, 2, height, width)

        with torch.no_grad():
            output = model(x)

        assert "pred_logits" in output
        assert "pred_boxes" in output

        # Check shapes
        assert output["pred_logits"].shape[0] == batch_size
        assert output["pred_logits"].shape[2] == num_classes + 1  # +1 for no-object class
        assert output["pred_boxes"].shape[0] == batch_size
        assert output["pred_boxes"].shape[2] == 4  # (cx, cy, w, h)

    @pytest.mark.slow
    def test_box_predictions_normalized(self):
        """Test that box predictions are in [0, 1] range."""
        model = DETR(version="detr_b0_nano", num_classes=5)
        model.eval()

        x = torch.randn(2, 2, 224, 224)

        with torch.no_grad():
            output = model(x)

        boxes = output["pred_boxes"]

        # After sigmoid, boxes should be in [0, 1]
        assert torch.all(boxes >= 0)
        assert torch.all(boxes <= 1)

    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test that gradients flow through DETR."""
        model = DETR(version="detr_b0_nano", num_classes=5)

        x = torch.randn(2, 2, 224, 224, requires_grad=True)

        output = model(x)
        loss = output["pred_logits"].sum() + output["pred_boxes"].sum()
        loss.backward()

        assert x.grad is not None


class TestCreateDetr:
    """Tests for create_detr factory function."""

    @pytest.mark.slow
    def test_create_detr_default(self):
        """Test creating DETR with default parameters."""
        model = create_detr()

        assert model is not None
        assert isinstance(model, nn.Module)

    @pytest.mark.slow
    def test_create_detr_custom_params(self):
        """Test creating DETR with custom parameters."""
        model = create_detr(
            backbone="efficientnet_b0",
            transformer="xcit-nano",
            num_classes=10,
            num_objects=25,
            hidden_dim=128
        )

        model.eval()
        x = torch.randn(2, 2, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output["pred_logits"].shape[1] == 25  # num_objects
        assert output["pred_logits"].shape[2] == 11  # num_classes + 1
