"""Tests for synthetic signal generation in image_datasets."""

import pytest
import torch
import numpy as np

from torchsig.image_datasets.datasets.synthetic_signals import (
    GeneratorFunctionDataset,
    generate_tone,
    tone_generator_function,
    generate_chirp,
    chirp_generator_function,
    generate_rectangle_signal,
    rectangle_signal_generator_function,
    generate_repeated_signal,
    repeated_signal_generator_function,
)


class TestGenerateTone:
    """Tests for tone signal generation."""

    def test_basic_generation(self):
        """Test basic tone generation."""
        tone = generate_tone(tone_width=100, max_height=10, min_height=3)

        assert isinstance(tone, torch.Tensor)
        assert tone.ndim == 3  # (1, height, width)
        assert tone.shape[0] == 1  # Single channel
        assert tone.shape[2] == 100  # Specified width

    def test_height_range(self):
        """Test that height is within specified range."""
        for _ in range(10):
            tone = generate_tone(tone_width=50, max_height=20, min_height=5)

            assert 5 <= tone.shape[1] <= 20

    def test_values_binary(self):
        """Test that tone values are binary (0 or 1)."""
        tone = generate_tone(tone_width=100)

        unique_values = torch.unique(tone)
        assert len(unique_values) <= 2
        assert torch.all((tone == 0) | (tone == 1))

    def test_generator_function(self):
        """Test curried generator function."""
        gen_fn = tone_generator_function(tone_width=80, max_height=15, min_height=5)

        tone = gen_fn()

        assert isinstance(tone, torch.Tensor)
        assert tone.shape[2] == 80


class TestGenerateChirp:
    """Tests for chirp signal generation."""

    def test_basic_generation(self):
        """Test basic chirp generation."""
        chirp = generate_chirp(chirp_width=3, height=64, width=128)

        assert isinstance(chirp, torch.Tensor)
        assert chirp.ndim == 3  # (1, height, width)
        assert chirp.shape[0] == 1

    @pytest.mark.parametrize("height,width", [
        (32, 64),
        (64, 128),
        (128, 256),
    ])
    def test_different_dimensions(self, height, width):
        """Test chirp generation with different dimensions."""
        chirp = generate_chirp(chirp_width=2, height=height, width=width)

        # Output should be approximately the specified size
        assert chirp.shape[1] > 0
        assert chirp.shape[2] > 0

    def test_chirp_contains_line(self):
        """Test that chirp contains a drawn line (non-zero values)."""
        chirp = generate_chirp(chirp_width=5, height=64, width=128)

        # Should contain some non-zero values (the chirp line)
        assert torch.sum(chirp > 0) > 0

    def test_generator_function(self):
        """Test curried generator function."""
        gen_fn = chirp_generator_function(chirp_width=3, height=64, width=128)

        chirp = gen_fn()

        assert isinstance(chirp, torch.Tensor)

    def test_random_scaling(self):
        """Test chirp with random scaling."""
        chirp = generate_chirp(
            chirp_width=3,
            height=64,
            width=128,
            random_height_scale=[0.5, 1.5],
            random_width_scale=[0.5, 1.5]
        )

        assert chirp.ndim == 3


class TestGenerateRectangleSignal:
    """Tests for rectangle signal generation."""

    def test_basic_generation(self):
        """Test basic rectangle generation."""
        rect = generate_rectangle_signal(
            min_width=10, max_width=50,
            min_height=5, max_height=20
        )

        assert isinstance(rect, torch.Tensor)
        assert rect.ndim == 3
        assert rect.shape[0] == 1

    def test_dimensions_within_range(self):
        """Test that dimensions are within specified range."""
        for _ in range(10):
            rect = generate_rectangle_signal(
                min_width=20, max_width=40,
                min_height=10, max_height=30
            )

            # Account for padding (+2)
            assert 22 <= rect.shape[2] <= 42
            assert 12 <= rect.shape[1] <= 32

    def test_rectangle_filled(self):
        """Test that rectangle is filled with 1s."""
        rect = generate_rectangle_signal(
            min_width=10, max_width=10,
            min_height=5, max_height=5
        )

        # Core rectangle (excluding padding) should be mostly 1s
        core = rect[0, 1:-1, 1:-1]
        assert torch.all(core == 1)

    def test_generator_function(self):
        """Test curried generator function."""
        gen_fn = rectangle_signal_generator_function(
            min_width=15, max_width=30,
            min_height=8, max_height=16
        )

        rect = gen_fn()

        assert isinstance(rect, torch.Tensor)


class TestGenerateRepeatedSignal:
    """Tests for repeated signal generation."""

    def test_basic_repetition(self):
        """Test basic signal repetition."""
        base_gen = lambda: torch.ones(1, 10, 20)

        repeated = generate_repeated_signal(
            generator_fn=base_gen,
            min_gap=2, max_gap=5,
            min_repeats=3, max_repeats=5
        )

        assert isinstance(repeated, torch.Tensor)
        assert repeated.shape[0] == 1
        assert repeated.shape[1] == 10  # Height preserved

    def test_width_increases_with_repeats(self):
        """Test that width increases with repetitions."""
        base_gen = lambda: torch.ones(1, 10, 20)

        # Multiple trials to account for randomness
        widths = []
        for _ in range(5):
            repeated = generate_repeated_signal(
                generator_fn=base_gen,
                min_gap=2, max_gap=2,
                min_repeats=5, max_repeats=5
            )
            widths.append(repeated.shape[2])

        # With 5 repeats of width 20 + 4 gaps of 2, expect ~108 width
        for w in widths:
            assert w > 20  # Should be wider than single signal

    def test_generator_function(self):
        """Test curried generator function."""
        base_gen = lambda: torch.ones(1, 5, 10)

        gen_fn = repeated_signal_generator_function(
            generator_fn=base_gen,
            min_gap=1, max_gap=3,
            min_repeats=2, max_repeats=4
        )

        repeated = gen_fn()

        assert isinstance(repeated, torch.Tensor)


class TestGeneratorFunctionDataset:
    """Tests for GeneratorFunctionDataset class."""

    def test_basic_usage(self):
        """Test basic dataset usage."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        sample = dataset[0]

        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (1, 32, 64)

    def test_length_is_one(self):
        """Test that dataset length is 1."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        assert len(dataset) == 1

    def test_next_method(self):
        """Test the next() convenience method."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        sample = dataset.next()

        assert isinstance(sample, torch.Tensor)

    def test_with_single_transform(self):
        """Test dataset with a single transform."""
        gen_fn = lambda: torch.ones(1, 32, 64)

        def scale_transform(x):
            return x * 2.0

        dataset = GeneratorFunctionDataset(
            generator_function=gen_fn,
            transforms=scale_transform
        )

        sample = dataset[0]

        assert torch.allclose(sample, torch.full((1, 32, 64), 2.0))

    def test_with_multiple_transforms(self):
        """Test dataset with multiple transforms."""
        gen_fn = lambda: torch.ones(1, 32, 64)

        def add_one(x):
            return x + 1

        def multiply_two(x):
            return x * 2

        dataset = GeneratorFunctionDataset(
            generator_function=gen_fn,
            transforms=[add_one, multiply_two]
        )

        sample = dataset[0]

        # (1 + 1) * 2 = 4
        assert torch.allclose(sample, torch.full((1, 32, 64), 4.0))

    def test_different_samples_each_call(self):
        """Test that each call generates different samples."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        sample1 = dataset[0]
        sample2 = dataset[0]

        # Random samples should be different
        assert not torch.equal(sample1, sample2)

    def test_with_tone_generator(self):
        """Test integration with tone generator."""
        gen_fn = tone_generator_function(tone_width=50, max_height=10, min_height=5)
        dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        sample = dataset[0]

        assert sample.shape[0] == 1
        assert sample.shape[2] == 50

    def test_with_chirp_generator(self):
        """Test integration with chirp generator."""
        gen_fn = chirp_generator_function(chirp_width=3, height=64, width=128)
        dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        sample = dataset[0]

        assert sample.ndim == 3

    def test_with_rectangle_generator(self):
        """Test integration with rectangle generator."""
        gen_fn = rectangle_signal_generator_function(
            min_width=20, max_width=40,
            min_height=10, max_height=20
        )
        dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        sample = dataset[0]

        assert sample.ndim == 3


class TestGeneratorIntegration:
    """Integration tests for generator functions with datasets."""

    def test_chained_generators(self):
        """Test using repeated signal with rectangle base."""
        rect_gen = rectangle_signal_generator_function(
            min_width=10, max_width=15,
            min_height=5, max_height=8
        )

        repeated_gen = repeated_signal_generator_function(
            generator_fn=rect_gen,
            min_gap=2, max_gap=4,
            min_repeats=3, max_repeats=5
        )

        dataset = GeneratorFunctionDataset(generator_function=repeated_gen)

        sample = dataset[0]

        assert sample.ndim == 3
        assert sample.shape[2] > 15  # Should be wider than single rect
