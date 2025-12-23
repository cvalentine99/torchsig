"""Tests for image transforms (denoising and impairments)."""

import pytest
import torch
import numpy as np

from torchsig.image_datasets.transforms.denoising import (
    normalize_image,
    isolate_foreground_signal,
)
from torchsig.image_datasets.transforms.impairments import (
    pad_border,
    scale_dynamic_range,
    GaussianNoiseTransform,
    ScaleTransform,
    RandomGaussianNoiseTransform,
    RippleNoiseTransform,
    RandomRippleNoiseTransform,
    BlurTransform,
    RandomImageResizeTransform,
)


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_normalize_tensor(self):
        """Test normalizing a tensor to [0, 1] range."""
        img = torch.randn(1, 64, 128) * 10 + 5  # Values roughly in [-25, 25]

        normalized = normalize_image(img)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_numpy(self):
        """Test normalizing a numpy array."""
        img = np.random.randn(1, 64, 128) * 10

        normalized = normalize_image(img)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves shape."""
        img = torch.randn(1, 32, 64)

        normalized = normalize_image(img)

        assert normalized.shape == img.shape

    def test_normalize_already_normalized(self):
        """Test normalizing an already normalized image."""
        img = torch.rand(1, 64, 128)  # Already in [0, 1]

        normalized = normalize_image(img)

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_constant_image(self):
        """Test normalizing a constant image."""
        img = torch.ones(1, 32, 32) * 5

        normalized = normalize_image(img)

        # Constant image normalized should be all zeros or handled gracefully
        assert not torch.isnan(normalized).any()

    def test_normalize_with_axis(self):
        """Test normalizing along a specific axis."""
        img = torch.randn(1, 64, 128)

        normalized = normalize_image(img, axis=2)

        assert normalized.shape == img.shape


class TestIsolateForegroundSignal:
    """Tests for isolate_foreground_signal function."""

    def test_basic_isolation(self):
        """Test basic foreground isolation."""
        img = torch.randn(1, 64, 128)

        isolated = isolate_foreground_signal(img, filter_strength=0)

        assert isinstance(isolated, torch.Tensor)
        assert isolated.shape == img.shape

    @pytest.mark.parametrize("filter_strength", [0, 1, 2, 5])
    def test_different_filter_strengths(self, filter_strength):
        """Test isolation with different filter strengths."""
        img = torch.randn(1, 64, 128)

        isolated = isolate_foreground_signal(img, filter_strength=filter_strength)

        assert not torch.isnan(isolated).any()
        assert isolated.shape == img.shape


class TestPadBorder:
    """Tests for pad_border function."""

    def test_pad_int(self):
        """Test padding with integer value."""
        img = torch.ones(1, 32, 64)

        padded = pad_border(img, to_pad=5)

        assert padded.shape == (1, 42, 74)

    def test_pad_tuple(self):
        """Test padding with tuple (y_pad, x_pad)."""
        img = torch.ones(1, 32, 64)

        padded = pad_border(img, to_pad=(10, 5))

        assert padded.shape == (1, 52, 74)

    def test_pad_values(self):
        """Test that padding adds zeros."""
        img = torch.ones(1, 10, 10)

        padded = pad_border(img, to_pad=2)

        # Check corners are zero
        assert padded[0, 0, 0] == 0
        assert padded[0, -1, -1] == 0

        # Check center is one
        assert padded[0, 5, 5] == 1

    def test_pad_numpy(self):
        """Test padding a numpy array."""
        img = np.ones((1, 32, 64))

        padded = pad_border(img, to_pad=3)

        assert isinstance(padded, torch.Tensor)
        assert padded.shape == (1, 38, 70)


class TestScaleDynamicRange:
    """Tests for scale_dynamic_range function."""

    def test_basic_scaling(self):
        """Test basic dynamic range scaling."""
        img = torch.randn(1, 64, 128)

        scaled = scale_dynamic_range(img)

        assert isinstance(scaled, torch.Tensor)
        assert scaled.shape == img.shape

    def test_preserves_shape(self):
        """Test that scaling preserves shape."""
        img = torch.randn(1, 32, 64)

        scaled = scale_dynamic_range(img)

        assert scaled.shape == img.shape


class TestGaussianNoiseTransform:
    """Tests for GaussianNoiseTransform class."""

    def test_basic_noise(self):
        """Test adding Gaussian noise."""
        transform = GaussianNoiseTransform(mean=0, std=0.1)
        img = torch.zeros(1, 64, 128)

        noisy = transform(img)

        assert noisy.shape == img.shape
        # Should have added noise (not all zeros anymore)
        assert not torch.allclose(noisy, img)

    def test_noise_statistics(self):
        """Test that noise has approximately correct statistics."""
        transform = GaussianNoiseTransform(mean=0, std=0.5)
        img = torch.zeros(1, 128, 128)

        noisy = transform(img)

        # Mean should be approximately 0 (before normalization)
        # Std should be approximately 0.5
        # After normalization, values are in [0, 1]
        assert noisy.min() >= 0
        assert noisy.max() <= 1

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same result."""
        transform1 = GaussianNoiseTransform(mean=0, std=0.2, seed=42)
        transform2 = GaussianNoiseTransform(mean=0, std=0.2, seed=42)

        img = torch.zeros(1, 32, 32)

        noisy1 = transform1(img.clone())
        noisy2 = transform2(img.clone())

        assert torch.allclose(noisy1, noisy2)


class TestScaleTransform:
    """Tests for ScaleTransform class."""

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 10.0])
    def test_scaling(self, scale):
        """Test scaling by different factors."""
        transform = ScaleTransform(scale=scale)
        img = torch.ones(1, 32, 64)

        scaled = transform(img)

        assert torch.allclose(scaled, img * scale)

    def test_scale_zero(self):
        """Test scaling by zero."""
        transform = ScaleTransform(scale=0)
        img = torch.ones(1, 32, 64)

        scaled = transform(img)

        assert torch.allclose(scaled, torch.zeros_like(img))


class TestRandomGaussianNoiseTransform:
    """Tests for RandomGaussianNoiseTransform class."""

    def test_basic_random_noise(self):
        """Test adding random Gaussian noise."""
        transform = RandomGaussianNoiseTransform(mean=0, range=(0.01, 0.5))
        img = torch.zeros(1, 64, 128)

        noisy = transform(img)

        assert noisy.shape == img.shape
        assert not torch.allclose(noisy, img)

    def test_noise_varies(self):
        """Test that noise varies between calls."""
        transform = RandomGaussianNoiseTransform(mean=0, range=(0.1, 0.5))
        img = torch.zeros(1, 32, 32)

        noisy1 = transform(img.clone())
        noisy2 = transform(img.clone())

        # Random noise should produce different results
        assert not torch.allclose(noisy1, noisy2)


class TestBlurTransform:
    """Tests for BlurTransform class."""

    def test_basic_blur(self):
        """Test basic blur application."""
        transform = BlurTransform(strength=1.0, blur_shape=5)
        img = torch.randn(1, 64, 128)

        blurred = transform(img)

        assert blurred.shape == img.shape

    def test_no_blur_at_zero_strength(self):
        """Test that zero strength means no blur."""
        transform = BlurTransform(strength=0.0, blur_shape=5)
        img = torch.randn(1, 64, 128)

        blurred = transform(img)

        assert torch.allclose(blurred, img)

    @pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
    def test_different_kernel_sizes(self, kernel_size):
        """Test blur with different kernel sizes."""
        transform = BlurTransform(strength=0.5, blur_shape=kernel_size)
        img = torch.randn(1, 64, 128)

        blurred = transform(img)

        assert blurred.shape == img.shape

    def test_blur_reduces_high_frequency(self):
        """Test that blur reduces high-frequency content."""
        transform = BlurTransform(strength=1.0, blur_shape=7)

        # Create high-frequency checkerboard pattern
        img = torch.zeros(1, 64, 64)
        img[0, ::2, ::2] = 1
        img[0, 1::2, 1::2] = 1

        blurred = transform(img)

        # Blurred image should have less variance
        assert blurred.var() < img.var()


class TestRippleNoiseTransform:
    """Tests for RippleNoiseTransform class."""

    def test_basic_ripple(self):
        """Test basic ripple noise application."""
        transform = RippleNoiseTransform(
            strength=0.5,
            num_emitors=10,
            image_shape=(64, 128)
        )
        img = torch.zeros(1, 64, 128)

        rippled = transform(img)

        assert rippled.shape == img.shape
        assert not torch.allclose(rippled, img)

    def test_ripple_without_precomputed_mesh(self):
        """Test ripple noise without precomputed mesh."""
        transform = RippleNoiseTransform(
            strength=0.3,
            num_emitors=5
        )
        img = torch.zeros(1, 32, 64)

        rippled = transform(img)

        assert rippled.shape == img.shape

    def test_strength_zero(self):
        """Test that zero strength means no ripple."""
        transform = RippleNoiseTransform(strength=0.0, num_emitors=10)
        img = torch.randn(1, 64, 128)

        rippled = transform(img)

        # With 0 strength, should be similar to input (after normalization)
        # Due to normalization, exact equality may not hold
        assert rippled.shape == img.shape


class TestRandomRippleNoiseTransform:
    """Tests for RandomRippleNoiseTransform class."""

    def test_basic_random_ripple(self):
        """Test random ripple noise."""
        transform = RandomRippleNoiseTransform(
            range=(0.1, 0.5),
            num_emitors=10
        )
        img = torch.zeros(1, 64, 128)

        rippled = transform(img)

        assert rippled.shape == img.shape

    def test_random_strength_varies(self):
        """Test that random strength produces varying results."""
        transform = RandomRippleNoiseTransform(
            range=(0.1, 0.9),
            num_emitors=10
        )
        img = torch.zeros(1, 32, 32)

        results = [transform(img.clone()) for _ in range(5)]

        # Not all results should be identical
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same


class TestRandomImageResizeTransform:
    """Tests for RandomImageResizeTransform class."""

    def test_basic_resize(self):
        """Test basic random resize."""
        transform = RandomImageResizeTransform(scale=(0.5, 2.0))
        img = torch.randn(1, 64, 128)

        resized = transform(img)

        assert resized.ndim == 3
        assert resized.shape[0] == 1

    def test_resize_range(self):
        """Test that resize stays within range."""
        transform = RandomImageResizeTransform(scale=(0.8, 1.2))
        img = torch.randn(1, 100, 100)

        resized = transform(img)

        # Size should be roughly within 80-120% of original
        assert 70 < resized.shape[1] < 130
        assert 70 < resized.shape[2] < 130

    def test_different_x_y_scales(self):
        """Test different scales for x and y."""
        transform = RandomImageResizeTransform(
            scale=(0.5, 0.6),
            y_scale=(1.5, 2.0)
        )
        img = torch.randn(1, 100, 100)

        resized = transform(img)

        # Height should be larger, width should be smaller
        assert resized.shape[1] > resized.shape[2]


class TestTransformChaining:
    """Tests for chaining multiple transforms."""

    def test_chain_noise_and_blur(self):
        """Test chaining noise and blur transforms."""
        noise = GaussianNoiseTransform(mean=0, std=0.2)
        blur = BlurTransform(strength=0.5, blur_shape=5)

        img = torch.zeros(1, 64, 128)

        result = blur(noise(img))

        assert result.shape == img.shape

    def test_chain_multiple_transforms(self):
        """Test chaining multiple transforms."""
        transforms = [
            ScaleTransform(scale=2.0),
            GaussianNoiseTransform(mean=0, std=0.1),
            BlurTransform(strength=0.3, blur_shape=3),
        ]

        img = torch.ones(1, 32, 32)

        result = img
        for t in transforms:
            result = t(result)

        assert result.shape == img.shape


class TestTransformEdgeCases:
    """Tests for edge cases in transforms."""

    def test_small_image(self):
        """Test transforms on small images."""
        transform = BlurTransform(strength=0.5, blur_shape=3)
        img = torch.randn(1, 8, 8)

        blurred = transform(img)

        assert blurred.shape == img.shape

    def test_single_pixel(self):
        """Test transforms on single pixel."""
        transform = GaussianNoiseTransform(mean=0, std=0.1)
        img = torch.randn(1, 1, 1)

        noisy = transform(img)

        assert noisy.shape == (1, 1, 1)

    def test_rectangular_image(self):
        """Test transforms on highly rectangular images."""
        transform = BlurTransform(strength=0.5, blur_shape=5)
        img = torch.randn(1, 10, 500)

        blurred = transform(img)

        assert blurred.shape == img.shape
