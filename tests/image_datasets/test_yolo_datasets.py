"""Tests for YOLO dataset classes in image_datasets."""

import pytest
import torch
import numpy as np
import tempfile
import os

from torchsig.image_datasets.datasets.yolo_datasets import (
    YOLODatum,
    YOLODatasetAdapter,
    YOLOImageCompositeDatasetComponent,
    YOLOImageCompositeDataset,
    yolo_to_pixels_on_image,
    yolo_box_on_image,
    extract_yolo_boxes,
)
from torchsig.image_datasets.datasets.synthetic_signals import GeneratorFunctionDataset


class TestYOLODatum:
    """Tests for YOLODatum class."""

    def test_basic_creation(self):
        """Test basic YOLODatum creation."""
        img = torch.randn(1, 64, 128)
        labels = [(0, 0.5, 0.5, 0.2, 0.1)]

        datum = YOLODatum(img=img, labels=labels)

        assert torch.equal(datum.img, img)
        assert datum.labels == labels

    def test_creation_with_class_id_only(self):
        """Test creation with only class ID (default label created)."""
        img = torch.randn(1, 64, 128)

        datum = YOLODatum(img=img, labels=0)

        assert datum.labels == [(0, 0.5, 0.5, 1.0, 1.0)]

    def test_creation_with_single_tuple(self):
        """Test creation with single label tuple."""
        img = torch.randn(1, 64, 128)
        label = (1, 0.3, 0.4, 0.2, 0.2)

        datum = YOLODatum(img=img, labels=label)

        assert datum.labels == [label]

    def test_has_labels(self):
        """Test has_labels method."""
        img = torch.randn(1, 64, 128)

        datum_with = YOLODatum(img=img, labels=[(0, 0.5, 0.5, 0.2, 0.1)])
        datum_without = YOLODatum(img=img, labels=[])

        assert datum_with.has_labels() is True
        assert datum_without.has_labels() is False

    def test_shape_property(self):
        """Test shape property."""
        img = torch.randn(1, 64, 128)
        datum = YOLODatum(img=img)

        assert datum.shape == (1, 64, 128)

    def test_len(self):
        """Test __len__ returns 2."""
        img = torch.randn(1, 64, 128)
        datum = YOLODatum(img=img)

        assert len(datum) == 2

    def test_getitem(self):
        """Test __getitem__ for tuple-like access."""
        img = torch.randn(1, 64, 128)
        labels = [(0, 0.5, 0.5, 0.2, 0.1)]
        datum = YOLODatum(img=img, labels=labels)

        assert torch.equal(datum[0], img)
        assert datum[1] == labels

    def test_setitem(self):
        """Test __setitem__ for tuple-like assignment."""
        img1 = torch.randn(1, 64, 128)
        img2 = torch.randn(1, 32, 64)
        labels1 = [(0, 0.5, 0.5, 0.2, 0.1)]
        labels2 = [(1, 0.3, 0.3, 0.1, 0.1)]

        datum = YOLODatum(img=img1, labels=labels1)

        datum[0] = img2
        datum[1] = labels2

        assert torch.equal(datum.img, img2)
        assert datum.labels == labels2

    def test_append_labels_single(self):
        """Test appending a single label."""
        img = torch.randn(1, 64, 128)
        datum = YOLODatum(img=img, labels=[])

        datum.append_labels((0, 0.5, 0.5, 0.2, 0.1))

        assert len(datum.labels) == 1

    def test_append_labels_multiple(self):
        """Test appending multiple labels."""
        img = torch.randn(1, 64, 128)
        datum = YOLODatum(img=img, labels=[(0, 0.1, 0.1, 0.05, 0.05)])

        datum.append_labels([
            (1, 0.5, 0.5, 0.2, 0.1),
            (2, 0.7, 0.7, 0.1, 0.1)
        ])

        assert len(datum.labels) == 3

    def test_size(self):
        """Test size method."""
        img = torch.randn(1, 64, 128)
        datum = YOLODatum(img=img)

        assert datum.size(0) == 1
        assert datum.size(1) == 64
        assert datum.size(2) == 128

    def test_transpose_yolo_labels(self):
        """Test label transposition."""
        composite = YOLODatum(img=torch.zeros(1, 100, 100))
        sub = YOLODatum(
            img=torch.ones(1, 20, 40),
            labels=[(0, 0.5, 0.5, 1.0, 1.0)]  # Centered in sub-image
        )

        # Place sub-image at (10, 20) in composite
        transposed = composite.transpose_yolo_labels(sub, (10, 20))

        assert len(transposed) == 1
        # Check that label is now relative to composite
        label = transposed[0]
        assert label[0] == 0  # Class ID preserved

    def test_compose_yolo_data_add(self):
        """Test composing two YOLODatums with addition."""
        base = YOLODatum(
            img=torch.zeros(1, 100, 100),
            labels=[]
        )
        overlay = YOLODatum(
            img=torch.ones(1, 20, 20),
            labels=[(0, 0.5, 0.5, 1.0, 1.0)]
        )

        base.compose_yolo_data(overlay, (10, 10), image_composition_mode='add')

        # Check that labels were added
        assert len(base.labels) > 0

        # Check that image was modified
        assert torch.sum(base.img) > 0


class TestYOLODatasetAdapter:
    """Tests for YOLODatasetAdapter class."""

    def test_basic_adaptation(self):
        """Test adapting a simple dataset."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        adapted = YOLODatasetAdapter(dataset=base_dataset, class_id=5)

        sample = adapted[0]

        assert isinstance(sample, YOLODatum)
        assert sample.labels == [(5, 0.5, 0.5, 1.0, 1.0)]

    def test_adaptation_without_class_id(self):
        """Test adaptation without class ID (empty labels)."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        adapted = YOLODatasetAdapter(dataset=base_dataset, class_id=None)

        sample = adapted[0]

        assert isinstance(sample, YOLODatum)
        assert sample.labels == []

    def test_len_preserved(self):
        """Test that length is preserved from base dataset."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        adapted = YOLODatasetAdapter(dataset=base_dataset, class_id=0)

        assert len(adapted) == len(base_dataset)


class TestYOLOImageCompositeDatasetComponent:
    """Tests for YOLOImageCompositeDatasetComponent class."""

    def test_basic_component(self):
        """Test basic component creation."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        component = YOLOImageCompositeDatasetComponent(
            component_dataset=base_dataset,
            min_to_add=1,
            max_to_add=3,
            class_id=0
        )

        sample = component[0]

        assert isinstance(sample, YOLODatum)

    def test_get_components_to_add(self):
        """Test getting variable number of components."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        component = YOLOImageCompositeDatasetComponent(
            component_dataset=base_dataset,
            min_to_add=2,
            max_to_add=5,
            class_id=0
        )

        components = component.get_components_to_add()

        assert 2 <= len(components) <= 5
        assert all(isinstance(c, YOLODatum) for c in components)

    def test_next_method(self):
        """Test next() convenience method."""
        gen_fn = lambda: torch.randn(1, 32, 64)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        component = YOLOImageCompositeDatasetComponent(
            component_dataset=base_dataset,
            class_id=1
        )

        sample = component.next()

        assert isinstance(sample, YOLODatum)


class TestYOLOImageCompositeDataset:
    """Tests for YOLOImageCompositeDataset class."""

    def test_basic_composite(self):
        """Test basic composite dataset creation."""
        dataset = YOLOImageCompositeDataset(
            composite_scale=(1, 128, 256),
            dataset_size=10
        )

        assert len(dataset) == 10

    def test_add_component(self):
        """Test adding components to composite."""
        dataset = YOLOImageCompositeDataset(
            composite_scale=(1, 128, 256),
            dataset_size=5
        )

        gen_fn = lambda: torch.ones(1, 20, 40)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        dataset.add_component(
            component_dataset=base_dataset,
            min_to_add=1,
            max_to_add=3,
            class_id=0
        )

        sample = dataset[0]

        assert isinstance(sample, YOLODatum)
        assert sample.shape == (1, 128, 256)

    def test_composite_has_labels(self):
        """Test that composite dataset generates labels."""
        dataset = YOLOImageCompositeDataset(
            composite_scale=(1, 128, 256),
            dataset_size=5
        )

        gen_fn = lambda: torch.ones(1, 20, 40)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        dataset.add_component(
            component_dataset=base_dataset,
            min_to_add=1,
            max_to_add=1,
            class_id=0
        )

        sample = dataset[0]

        assert len(sample.labels) >= 1

    def test_multiple_components(self):
        """Test composite with multiple component types."""
        dataset = YOLOImageCompositeDataset(
            composite_scale=(1, 128, 256),
            dataset_size=5
        )

        gen_fn1 = lambda: torch.ones(1, 15, 30)
        gen_fn2 = lambda: torch.ones(1, 10, 20) * 0.5

        base_dataset1 = GeneratorFunctionDataset(generator_function=gen_fn1)
        base_dataset2 = GeneratorFunctionDataset(generator_function=gen_fn2)

        dataset.add_component(base_dataset1, min_to_add=1, max_to_add=2, class_id=0)
        dataset.add_component(base_dataset2, min_to_add=1, max_to_add=2, class_id=1)

        sample = dataset[0]

        assert isinstance(sample, YOLODatum)
        assert len(sample.labels) >= 2

    def test_with_transforms(self):
        """Test composite with transforms."""
        def double_transform(datum):
            datum.img = datum.img * 2
            return datum

        dataset = YOLOImageCompositeDataset(
            composite_scale=(1, 64, 64),
            dataset_size=3,
            transforms=[double_transform]
        )

        gen_fn = lambda: torch.ones(1, 10, 10)
        base_dataset = GeneratorFunctionDataset(generator_function=gen_fn)

        dataset.add_component(base_dataset, min_to_add=1, max_to_add=1, class_id=0)

        sample = dataset[0]

        # Transform should have been applied
        assert isinstance(sample, YOLODatum)


class TestYOLOUtilityFunctions:
    """Tests for YOLO utility functions."""

    def test_yolo_to_pixels_on_image(self):
        """Test YOLO to pixel coordinate conversion."""
        img = torch.zeros(1, 100, 200)
        box = (0.5, 0.5, 0.2, 0.1)  # (cx, cy, w, h)

        x_start, y_start, x_end, y_end = yolo_to_pixels_on_image(img, box)

        # Center at 0.5, 0.5 on 200x100 image -> center at (100, 50)
        # Width 0.2 * 200 = 40, Height 0.1 * 100 = 10
        # So x: 80-120, y: 45-55
        assert 70 <= x_start <= 90
        assert 110 <= x_end <= 130
        assert 40 <= y_start <= 50
        assert 50 <= y_end <= 60

    def test_yolo_box_on_image(self):
        """Test extracting box region from image."""
        img = torch.ones(1, 100, 200)
        box = (0.5, 0.5, 0.2, 0.2)

        extracted = yolo_box_on_image(img, box)

        assert extracted.ndim == 3
        assert extracted.shape[0] == 1

    def test_extract_yolo_boxes(self):
        """Test extracting all boxes from a YOLODatum."""
        img = torch.randn(1, 100, 200)
        labels = [
            (0, 0.25, 0.5, 0.2, 0.2),
            (1, 0.75, 0.5, 0.2, 0.2),
        ]
        datum = YOLODatum(img=img, labels=labels)

        extracted = extract_yolo_boxes(datum)

        assert len(extracted) == 2
        assert all(isinstance(e, YOLODatum) for e in extracted)


class TestYOLOEdgeCases:
    """Tests for edge cases in YOLO handling."""

    def test_empty_composite(self):
        """Test composite with no components added."""
        dataset = YOLOImageCompositeDataset(
            composite_scale=(1, 64, 64),
            dataset_size=3
        )

        sample = dataset[0]

        assert isinstance(sample, YOLODatum)
        assert sample.shape == (1, 64, 64)
        assert len(sample.labels) == 0

    def test_box_at_edge(self):
        """Test box placement at image edges."""
        composite = YOLODatum(img=torch.zeros(1, 100, 100))
        sub = YOLODatum(
            img=torch.ones(1, 20, 20),
            labels=[(0, 0.5, 0.5, 1.0, 1.0)]
        )

        # Place at corner (0, 0)
        composite.compose_yolo_data(sub, (0, 0), image_composition_mode='add')

        # Should succeed without error
        assert isinstance(composite, YOLODatum)

    def test_datum_iteration(self):
        """Test that YOLODatum can be iterated (tuple unpacking)."""
        img = torch.randn(1, 64, 128)
        labels = [(0, 0.5, 0.5, 0.2, 0.1)]
        datum = YOLODatum(img=img, labels=labels)

        unpacked_img, unpacked_labels = datum

        assert torch.equal(unpacked_img, img)
        assert unpacked_labels == labels
