from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest  # For pytest.raises and pytest.warns used in ModelTester and ValidationTester
from pydantic import ValidationError

from xrdantic.config import ValidationContext  # Used in ValidationTester
from xrdantic.models import Coordinate, DataArray, Dataset, XrBase  # Used in ModelTester
from xrdantic.types import Dim  # Used in ModelTester


class ModelTester:
    """Helper class for testing xrdantic models."""

    @staticmethod
    def assert_field_validation_error(
        model_class: type[XrBase], field_name: str, invalid_value: Any, error_type: str | None = None
    ) -> None:
        """Assert that a field raises a validation error with invalid input."""
        with pytest.raises(ValidationError) as exc_info:
            kwargs = {field_name: invalid_value}
            model_class(**kwargs)

        if error_type:
            assert any(error.get("type") == error_type for error in exc_info.value.errors())

    @staticmethod
    def assert_field_accepts_value(model_class: type[XrBase], field_name: str, valid_value: Any) -> None:
        """Assert that a field accepts a valid input."""
        kwargs = {field_name: valid_value}
        instance = model_class(**kwargs)
        assert getattr(instance, field_name) is not None

    @staticmethod
    def create_test_coordinate(
        dims: tuple[Dim, ...] = (Dim("x"),), size: int = 10, coord_name: str = "test_coord"
    ) -> type[Coordinate]:
        """Create a test coordinate class for testing."""
        from xrdantic.utils import Data  # Local import to avoid circular dependency issues at module level

        class TestCoordinate(Coordinate):
            data: Data[dims, np.float64]  # type: ignore

        return TestCoordinate

    @staticmethod
    def create_test_dataarray(
        dims: tuple[Dim, ...] = (Dim("x"), Dim("y")), data_name: str = "test_data"
    ) -> type[DataArray]:
        """Create a test DataArray class for testing."""
        from xrdantic.utils import Data  # Local import

        class TestDataArray(DataArray):
            data: Data[dims, np.float64]  # type: ignore

        return TestDataArray

    @staticmethod
    def create_test_dataset() -> type[Dataset]:
        """Create a test Dataset class for testing."""

        class TestDataset(Dataset):
            temperature: Any  # Would be a DataArray type
            pressure: Any  # Would be a DataArray type

        return TestDataset


class DataGenerator:
    """Utilities for generating test data."""

    rng = np.random.default_rng(42)

    @staticmethod
    def random_array(size: tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
        """Generate a random array with the specified shape and dtype."""
        return DataGenerator.rng.random(size).astype(dtype)

    @staticmethod
    def sequential_array(size: int, start: int = 0, dtype: type = np.float64) -> np.ndarray:
        """Generate a sequential array."""
        return np.arange(start, start + size, dtype=dtype)

    @staticmethod
    def constant_array(size: tuple[int, ...], value: Any, dtype: type = np.float64) -> np.ndarray:
        """Generate an array filled with a constant value."""
        return np.full(size, value, dtype=dtype)

    @staticmethod
    def array_with_nans(shape: tuple[int, ...], nan_fraction: float = 0.1) -> np.ndarray:
        """Generate an array with some NaN values."""
        arr = DataGenerator.rng.random(shape)
        n_nans = int(np.prod(shape) * nan_fraction)
        flat_arr = arr.flatten()
        nan_indices = DataGenerator.rng.choice(len(flat_arr), n_nans, replace=False)
        flat_arr[nan_indices] = np.nan
        return flat_arr.reshape(shape)

    @staticmethod
    def array_with_infs(shape: tuple[int, ...], inf_fraction: float = 0.1) -> np.ndarray:
        """Generate an array with some infinite values."""
        arr = DataGenerator.rng.random(shape)
        n_infs = int(np.prod(shape) * inf_fraction)
        flat_arr = arr.flatten()
        inf_indices = DataGenerator.rng.choice(len(flat_arr), n_infs, replace=False)
        flat_arr[inf_indices] = np.inf
        return flat_arr.reshape(shape)


class ValidationTester:
    """Utilities for testing validation behavior."""

    @staticmethod
    @contextmanager
    def strict_validation():
        """Context manager for testing with strict validation enabled."""
        with ValidationContext(strict_validation=True):
            yield

    @staticmethod
    @contextmanager
    def lenient_validation():
        """Context manager for testing with lenient validation."""
        with ValidationContext(allow_nan_values=True, allow_inf_values=True, strict_validation=False):
            yield

    @staticmethod
    def assert_validation_warning(warning_type: type[Warning] = UserWarning):
        """Assert that a specific warning is raised during validation."""
        return pytest.warns(warning_type)
