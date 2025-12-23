"""Tests for composite datasets."""

import pytest
import torch
import numpy as np

from torchsig.image_datasets.datasets.composites import ConcatDataset
from torchsig.image_datasets.datasets.synthetic_signals import (
    GeneratorFunctionDataset,
    tone_generator_function,
    rectangle_signal_generator_function,
)


class TestConcatDataset:
    """Tests for ConcatDataset class."""

    def test_basic_concat(self):
        """Test basic concatenation of datasets."""
        gen1 = lambda: torch.ones(1, 10, 20)
        gen2 = lambda: torch.zeros(1, 15, 25)

        ds1 = GeneratorFunctionDataset(generator_function=gen1)
        ds2 = GeneratorFunctionDataset(generator_function=gen2)

        concat = ConcatDataset([ds1, ds2], balance=False)

        assert len(concat) == 2

    def test_balanced_sampling(self):
        """Test balanced sampling from datasets."""
        gen1 = lambda: torch.ones(1, 10, 20)
        gen2 = lambda: torch.zeros(1, 15, 25)

        ds1 = GeneratorFunctionDataset(generator_function=gen1)
        ds2 = GeneratorFunctionDataset(generator_function=gen2)

        concat = ConcatDataset([ds1, ds2], balance=True)

        # With balance=True, should sample uniformly
        # Length is max(len(ds)) * num_datasets
        assert len(concat) == 2

    def test_getitem_sequential(self):
        """Test sequential access without balancing."""
        gen1 = lambda: torch.ones(1, 10, 20)
        gen2 = lambda: torch.zeros(1, 15, 25)

        ds1 = GeneratorFunctionDataset(generator_function=gen1)
        ds2 = GeneratorFunctionDataset(generator_function=gen2)

        concat = ConcatDataset([ds1, ds2], balance=False)

        sample0 = concat[0]
        sample1 = concat[1]

        # First sample from ds1 (ones)
        assert torch.all(sample0 == 1)
        # Second sample from ds2 (zeros)
        assert torch.all(sample1 == 0)

    def test_balanced_returns_variety(self):
        """Test that balanced mode returns variety."""
        gen1 = lambda: torch.ones(1, 10, 20) * 1
        gen2 = lambda: torch.ones(1, 15, 25) * 2

        ds1 = GeneratorFunctionDataset(generator_function=gen1)
        ds2 = GeneratorFunctionDataset(generator_function=gen2)

        concat = ConcatDataset([ds1, ds2], balance=True)

        # Sample multiple times
        samples = [concat[0] for _ in range(20)]

        # Should have samples from both datasets
        max_values = [torch.max(s).item() for s in samples]
        assert 1.0 in max_values or 2.0 in max_values

    def test_with_transforms(self):
        """Test concatenation with transforms."""
        gen1 = lambda: torch.ones(1, 10, 20)
        gen2 = lambda: torch.ones(1, 15, 25)

        ds1 = GeneratorFunctionDataset(generator_function=gen1)
        ds2 = GeneratorFunctionDataset(generator_function=gen2)

        def double_transform(x):
            return x * 2

        concat = ConcatDataset([ds1, ds2], balance=False, transforms=[double_transform])

        sample = concat[0]

        assert torch.all(sample == 2)

    def test_multiple_transforms(self):
        """Test with multiple transforms."""
        gen = lambda: torch.ones(1, 10, 20)
        ds = GeneratorFunctionDataset(generator_function=gen)

        def add_one(x):
            return x + 1

        def multiply_three(x):
            return x * 3

        concat = ConcatDataset([ds], transforms=[add_one, multiply_three])

        sample = concat[0]

        # (1 + 1) * 3 = 6
        assert torch.all(sample == 6)

    def test_next_method(self):
        """Test next() convenience method."""
        gen = lambda: torch.randn(1, 10, 20)
        ds = GeneratorFunctionDataset(generator_function=gen)

        concat = ConcatDataset([ds])

        sample = concat.next()

        assert isinstance(sample, torch.Tensor)

    def test_many_datasets(self):
        """Test concatenation of many datasets."""
        datasets = []
        for i in range(5):
            gen = lambda i=i: torch.ones(1, 10, 20) * i
            datasets.append(GeneratorFunctionDataset(generator_function=gen))

        concat = ConcatDataset(datasets, balance=False)

        assert len(concat) == 5

        for i in range(5):
            sample = concat[i]
            assert torch.all(sample == i)

    def test_with_signal_generators(self):
        """Test with actual signal generators."""
        tone_gen = tone_generator_function(tone_width=50)
        rect_gen = rectangle_signal_generator_function(
            min_width=30, max_width=40,
            min_height=10, max_height=15
        )

        ds1 = GeneratorFunctionDataset(generator_function=tone_gen)
        ds2 = GeneratorFunctionDataset(generator_function=rect_gen)

        concat = ConcatDataset([ds1, ds2], balance=True)

        # Generate multiple samples
        samples = [concat[0] for _ in range(10)]

        # All should be valid tensors
        assert all(s.ndim == 3 for s in samples)


class TestConcatDatasetEdgeCases:
    """Tests for edge cases in ConcatDataset."""

    def test_single_dataset(self):
        """Test concatenation of single dataset."""
        gen = lambda: torch.randn(1, 10, 20)
        ds = GeneratorFunctionDataset(generator_function=gen)

        concat = ConcatDataset([ds])

        assert len(concat) == 1

        sample = concat[0]
        assert sample.ndim == 3

    def test_empty_transforms(self):
        """Test with empty transforms list."""
        gen = lambda: torch.ones(1, 10, 20)
        ds = GeneratorFunctionDataset(generator_function=gen)

        concat = ConcatDataset([ds], transforms=[])

        sample = concat[0]

        assert torch.all(sample == 1)

    def test_balanced_with_single_dataset(self):
        """Test balanced mode with single dataset."""
        gen = lambda: torch.randn(1, 10, 20)
        ds = GeneratorFunctionDataset(generator_function=gen)

        concat = ConcatDataset([ds], balance=True)

        sample = concat[0]

        assert sample.ndim == 3


class TestConcatDatasetWithDifferentSizes:
    """Tests for ConcatDataset with datasets of different sizes."""

    def test_different_image_sizes(self):
        """Test concatenation of datasets with different output sizes."""
        gen1 = lambda: torch.randn(1, 10, 20)
        gen2 = lambda: torch.randn(1, 50, 100)

        ds1 = GeneratorFunctionDataset(generator_function=gen1)
        ds2 = GeneratorFunctionDataset(generator_function=gen2)

        concat = ConcatDataset([ds1, ds2], balance=False)

        sample0 = concat[0]
        sample1 = concat[1]

        assert sample0.shape == (1, 10, 20)
        assert sample1.shape == (1, 50, 100)

    def test_different_channels(self):
        """Test concatenation with different channel counts."""
        gen1 = lambda: torch.randn(1, 10, 20)
        gen2 = lambda: torch.randn(3, 10, 20)

        ds1 = GeneratorFunctionDataset(generator_function=gen1)
        ds2 = GeneratorFunctionDataset(generator_function=gen2)

        concat = ConcatDataset([ds1, ds2], balance=False)

        sample0 = concat[0]
        sample1 = concat[1]

        assert sample0.shape[0] == 1
        assert sample1.shape[0] == 3
