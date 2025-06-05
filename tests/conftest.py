"""
Testing configuration and utilities for xrdantic.

This module provides testing helpers, fixtures, and utilities for validating
xrdantic models and their behavior.
"""

from __future__ import annotations

import numpy as np
import numpydantic.dtype as dtypes
import pytest
from numpydantic import NDArray

from xrdantic.models import Coordinate, DataArray, Dataset
from xrdantic.types import Dim
from xrdantic.utils import Attr, Data, Name

from .utils import DataGenerator, ModelTester


# Pytest fixtures for common test data
@pytest.fixture
def simple_1d_array() -> NDArray:
    """Fixture providing a simple 1D array."""
    return np.arange(10, dtype=np.float64)


@pytest.fixture
def simple_2d_array() -> NDArray:
    """Fixture providing a simple 2D array."""
    return DataGenerator.random_array((5, 10))


@pytest.fixture
def array_with_nans() -> NDArray:
    """Fixture providing an array with NaN values."""
    return DataGenerator.array_with_nans((10,))


@pytest.fixture
def array_with_infs() -> NDArray:
    """Fixture providing an array with infinite values."""
    return DataGenerator.array_with_infs((10,))


@pytest.fixture
def test_coordinate_class() -> type[Coordinate]:
    """Fixture providing a test coordinate class."""
    return ModelTester.create_test_coordinate()


@pytest.fixture
def test_dataarray_class() -> type[DataArray]:
    """Fixture providing a test DataArray class."""
    return ModelTester.create_test_dataarray()


@pytest.fixture
def test_dataset_class() -> type[Dataset]:
    """Fixture providing a test Dataset class."""
    return ModelTester.create_test_dataset()


# Parametrized test helpers
def parametrize_array_dtypes() -> pytest.MarkDecorator:
    """Parametrize test with common array dtypes."""
    return pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])


def parametrize_array_shapes() -> pytest.MarkDecorator:
    """Parametrize test with common array shapes."""
    return pytest.mark.parametrize("shape", [(10,), (5, 10), (2, 3, 4), (10, 10, 10)])


def parametrize_invalid_shapes() -> pytest.MarkDecorator:
    """Parametrize test with invalid array shapes."""
    return pytest.mark.parametrize("shape", [(), (0,), (-1, 5), (5, -1), (0, 0)])


# ===== Dimension and Class Definitions =====

# Define dimensions at module level
X = Dim("x")
Y = Dim("y")
Z = Dim("z")
Time = Dim("time")


# Define coordinate classes
class XCoord(Coordinate):
    data: Data[X, int]
    name: Name
    units: Attr[str] = "pixels"
    long_name: Attr[str] = "X coordinate"


class YCoord(Coordinate):
    data: Data[Y, int]
    name: Name
    units: Attr[str] = "pixels"
    long_name: Attr[str] = "Y coordinate"


class ZCoord(Coordinate):
    data: Data[Z, float]
    name: Name
    units: Attr[str] = "meters"
    long_name: Attr[str] = "Z coordinate"


# Define DataArray classes for weather data
class Temperature(DataArray):
    data: Data[(Y, X), float]
    x: XCoord
    y: YCoord
    name: Name
    units: Attr[str] = "celsius"


class Pressure(DataArray):
    data: Data[(Y, X), float]
    x: XCoord
    y: YCoord
    name: Name
    units: Attr[str] = "hPa"


class Humidity(DataArray):
    data: Data[(Y, X), float]
    x: XCoord
    y: YCoord
    name: Name
    units: Attr[str] = "percent"


# Define main DataArray class for testing
class Image2D(DataArray):
    data: Data[(Y, X), dtypes.Float]
    x: XCoord
    y: YCoord
    name: Name
    units: Attr[str] = "intensity"
    description: Attr[str] = "2D image data"


# Define Dataset class
class WeatherData(Dataset):
    temperature: Temperature
    pressure: Pressure
    humidity: Humidity
    x: XCoord
    y: YCoord
    source: Attr[str] = "weather_station"


@pytest.fixture
def sample_dims() -> dict[str, Dim]:
    """Sample dimension objects for testing."""
    return {"X": X, "Y": Y, "Z": Z, "Time": Time}


@pytest.fixture
def sample_coordinate_classes() -> dict[str, type[Coordinate]]:
    """Sample coordinate classes for testing."""
    return {"XCoord": XCoord, "YCoord": YCoord, "ZCoord": ZCoord}


@pytest.fixture
def sample_dataarray_classes() -> dict[str, type[DataArray]]:
    """Sample DataArray classes for testing."""
    return {"Temperature": Temperature, "Pressure": Pressure, "Humidity": Humidity, "Image2D": Image2D}


@pytest.fixture
def sample_dataarray_class() -> type[DataArray]:
    """Sample DataArray class for testing."""
    return Image2D


@pytest.fixture
def sample_dataset_class() -> type[WeatherData]:
    """Sample Dataset class for testing."""
    return WeatherData
