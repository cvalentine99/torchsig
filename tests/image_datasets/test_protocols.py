"""Tests for protocol datasets (FrequencyHopping, CFG)."""

import pytest
import torch
import numpy as np

from torchsig.image_datasets.datasets.protocols import (
    FrequencyHoppingDataset,
    YOLOFrequencyHoppingDataset,
    CFGSignalProtocolDataset,
    VerticalCFGSignalProtocolDataset,
    YOLOCFGSignalProtocolDataset,
    random_hopping,
)
from torchsig.image_datasets.datasets.yolo_datasets import YOLODatum
from torchsig.image_datasets.datasets.synthetic_signals import (
    GeneratorFunctionDataset,
    rectangle_signal_generator_function,
)


class TestRandomHopping:
    """Tests for random_hopping function."""

    def test_basic_hopping(self):
        """Test basic hopping functionality."""
        channel = random_hopping(n_channels=5, channel_order=[])

        assert 0 <= channel < 5

    def test_avoids_last_channel(self):
        """Test that hopping avoids the last used channel."""
        for _ in range(20):
            channel = random_hopping(n_channels=5, channel_order=[2])

            assert channel != 2

    def test_single_channel(self):
        """Test hopping with single channel (edge case)."""
        # With only one channel, it must return 0 even if last was 0
        channel = random_hopping(n_channels=1, channel_order=[0])

        assert channel == 0

    def test_two_channels(self):
        """Test hopping alternates with two channels."""
        channels = []
        for _ in range(10):
            channel = random_hopping(n_channels=2, channel_order=channels)
            channels.append(channel)

        # Should alternate between 0 and 1
        for i in range(1, len(channels)):
            assert channels[i] != channels[i - 1]


class TestFrequencyHoppingDataset:
    """Tests for FrequencyHoppingDataset class."""

    def test_basic_creation(self):
        """Test basic dataset creation."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=5,
            signal_length=20,
            num_signals=3
        )

        assert len(dataset) == 1

    def test_getitem_returns_tensor(self):
        """Test that __getitem__ returns a tensor."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=5,
            signal_length=20,
            num_signals=3
        )

        sample = dataset[0]

        assert isinstance(sample, torch.Tensor)

    def test_output_dimensions(self):
        """Test output image dimensions."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=5,
            signal_length=20,
            num_signals=4
        )

        sample = dataset[0]

        # Height should be channel_height * num_channels = 50
        assert sample.shape[1] == 50
        # Width should be signal_length * num_signals = 80
        assert sample.shape[2] == 80

    def test_with_dataset_input(self):
        """Test using a dataset as signal source."""
        gen_fn = rectangle_signal_generator_function(
            min_width=15, max_width=20,
            min_height=8, max_height=10
        )
        signal_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_dataset,
            channel_height=10,
            num_channels=4,
            signal_length=20,
            num_signals=3
        )

        sample = dataset[0]

        assert sample.ndim == 3

    def test_random_parameters(self):
        """Test with random parameter ranges."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=[8, 12],  # Random range
            num_channels=[3, 5],     # Random range
            signal_length=[15, 25],  # Random range
            num_signals=[2, 4]       # Random range
        )

        # Should work without error
        sample = dataset[0]
        assert sample.ndim == 3

    def test_custom_hopping_function(self):
        """Test with custom hopping function."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        def always_channel_zero(n_channels, order):
            return 0

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=5,
            signal_length=20,
            num_signals=3,
            hopping_function=always_channel_zero
        )

        sample = dataset[0]

        # All signals should be in channel 0 (top of image)
        # Check that bottom channels are empty
        assert torch.sum(sample[0, 40:, :]) == 0

    def test_with_transforms(self):
        """Test with transforms applied."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        def double_transform(x):
            return x * 2

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=3,
            signal_length=20,
            num_signals=2,
            transforms=[double_transform]
        )

        sample = dataset[0]

        # Check that transform was applied
        assert torch.max(sample) >= 2

    def test_next_method(self):
        """Test the next() convenience method."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        dataset = FrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=3,
            signal_length=20,
            num_signals=2
        )

        sample = dataset.next()

        assert isinstance(sample, torch.Tensor)


class TestYOLOFrequencyHoppingDataset:
    """Tests for YOLOFrequencyHoppingDataset class."""

    def test_returns_yolo_datum(self):
        """Test that dataset returns YOLODatum."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        dataset = YOLOFrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=4,
            signal_length=20,
            num_signals=3
        )

        sample = dataset[0]

        assert isinstance(sample, YOLODatum)

    def test_has_labels(self):
        """Test that output has labels."""
        signal_fn = lambda: torch.ones(1, 10, 20)

        dataset = YOLOFrequencyHoppingDataset(
            signal_fn=signal_fn,
            channel_height=10,
            num_channels=4,
            signal_length=20,
            num_signals=3
        )

        sample = dataset[0]

        assert sample.has_labels()
        assert len(sample.labels) == 3  # One per signal


class TestCFGSignalProtocolDataset:
    """Tests for CFGSignalProtocolDataset class."""

    def test_basic_creation(self):
        """Test basic CFG dataset creation."""
        dataset = CFGSignalProtocolDataset(initial_token="start")

        # Add a simple rule
        dataset.add_rule("start", [lambda: torch.ones(1, 10, 20)])

        sample = dataset[0]

        assert isinstance(sample, torch.Tensor)

    def test_multiple_rules(self):
        """Test CFG with multiple rules."""
        dataset = CFGSignalProtocolDataset(initial_token="start")

        dataset.add_rule("start", ["signal", "signal"])
        dataset.add_rule("signal", [lambda: torch.ones(1, 10, 20)])

        sample = dataset[0]

        # Should be wider than single signal
        assert sample.shape[2] > 20

    def test_probabilistic_rules(self):
        """Test that rules are selected probabilistically."""
        dataset = CFGSignalProtocolDataset(initial_token="start")

        small_signal = lambda: torch.ones(1, 5, 10)
        large_signal = lambda: torch.ones(1, 15, 30)

        dataset.add_rule("start", [small_signal], priority=1.0)
        dataset.add_rule("start", [large_signal], priority=1.0)

        # Generate multiple samples to test randomness
        samples = [dataset[0] for _ in range(10)]

        # Should have variety in sizes
        sizes = set(s.shape[2] for s in samples)
        assert len(sizes) > 1

    def test_recursive_rules(self):
        """Test recursive grammar rules."""
        dataset = CFGSignalProtocolDataset(initial_token="sequence")

        signal = lambda: torch.ones(1, 10, 20)

        dataset.add_rule("sequence", [signal, "sequence"], priority=0.7)
        dataset.add_rule("sequence", [signal], priority=0.3)

        sample = dataset[0]

        # Should produce a signal (possibly repeated)
        assert sample.ndim == 3

    def test_with_datasets(self):
        """Test using datasets in rules."""
        gen_fn = rectangle_signal_generator_function(
            min_width=15, max_width=25,
            min_height=10, max_height=15
        )
        signal_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        dataset = CFGSignalProtocolDataset(initial_token="start")
        dataset.add_rule("start", [signal_dataset])

        sample = dataset[0]

        assert sample.ndim == 3

    def test_none_in_rules(self):
        """Test that None in rules produces gaps."""
        dataset = CFGSignalProtocolDataset(initial_token="start")

        signal = lambda: torch.ones(1, 10, 20)

        dataset.add_rule("start", [signal, None, signal])

        sample = dataset[0]

        # Width should be at least 2 signals
        assert sample.shape[2] >= 40

    def test_set_initial_token(self):
        """Test setting initial token after creation."""
        dataset = CFGSignalProtocolDataset()

        signal = lambda: torch.ones(1, 10, 20)

        dataset.add_rule("begin", [signal])
        dataset.set_initial_token("begin")

        sample = dataset[0]

        assert sample.shape == (1, 10, 20)


class TestVerticalCFGSignalProtocolDataset:
    """Tests for VerticalCFGSignalProtocolDataset class."""

    def test_vertical_stacking(self):
        """Test that signals are stacked vertically."""
        dataset = VerticalCFGSignalProtocolDataset(initial_token="start")

        signal = lambda: torch.ones(1, 10, 50)

        dataset.add_rule("start", [signal, signal])

        sample = dataset[0]

        # Height should be stacked (2 * 10 = 20)
        assert sample.shape[1] == 20
        # Width should be max of signals
        assert sample.shape[2] == 50


class TestYOLOCFGSignalProtocolDataset:
    """Tests for YOLOCFGSignalProtocolDataset class."""

    def test_returns_yolo_datum(self):
        """Test that output is YOLODatum."""
        dataset = YOLOCFGSignalProtocolDataset(initial_token="start")

        signal = lambda: torch.ones(1, 10, 20)

        dataset.add_rule("start", [signal])

        sample = dataset[0]

        assert isinstance(sample, YOLODatum)

    def test_has_labels(self):
        """Test that output has labels."""
        dataset = YOLOCFGSignalProtocolDataset(initial_token="start")

        signal = lambda: torch.ones(1, 10, 20)

        dataset.add_rule("start", [signal, signal])

        sample = dataset[0]

        # Should have labels for each signal
        assert sample.has_labels()


class TestProtocolIntegration:
    """Integration tests for protocol datasets."""

    def test_hopping_with_cfg(self):
        """Test using CFG output as hopping signal."""
        cfg_dataset = CFGSignalProtocolDataset(initial_token="start")
        signal = lambda: torch.ones(1, 10, 30)
        cfg_dataset.add_rule("start", [signal, signal])

        hopping_dataset = FrequencyHoppingDataset(
            signal_fn=cfg_dataset,
            channel_height=15,
            num_channels=3,
            signal_length=60,
            num_signals=2
        )

        sample = hopping_dataset[0]

        assert sample.ndim == 3

    def test_complex_cfg_grammar(self):
        """Test complex grammar with multiple levels."""
        dataset = CFGSignalProtocolDataset(initial_token="message")

        preamble = lambda: torch.ones(1, 5, 10)
        data = lambda: torch.ones(1, 8, 20)
        footer = lambda: torch.ones(1, 5, 10)

        dataset.add_rule("message", ["preamble", "body", "footer"])
        dataset.add_rule("preamble", [preamble])
        dataset.add_rule("body", ["data", "data"])
        dataset.add_rule("body", ["data"])
        dataset.add_rule("data", [data])
        dataset.add_rule("footer", [footer])

        sample = dataset[0]

        assert sample.ndim == 3
        # Should have preamble + body + footer width
        assert sample.shape[2] >= 40
