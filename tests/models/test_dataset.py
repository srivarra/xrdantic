"""
Comprehensive tests for Dataset class using property-based testing.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from xrdantic.models import Dataset
from xrdantic.utils import Attr

from ..utils import DataGenerator  # noqa: TID252

# ===== Hypothesis Strategies =====


@st.composite
def valid_2d_array_data(draw, min_size=2, max_size=15):
    """Generate valid 2D array data for Dataset testing."""
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    elements = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    shape = draw(array_shapes(min_dims=2, max_dims=2, min_side=min_size, max_side=max_size))
    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


@st.composite
def coordinated_data_arrays(draw, min_size=2, max_size=8):
    """Generate coordinated data arrays with matching dimensions."""
    y_size = draw(st.integers(min_value=min_size, max_value=max_size))
    x_size = draw(st.integers(min_value=min_size, max_value=max_size))

    # Generate data for temperature, pressure, humidity with same shape
    temp_data = draw(
        arrays(
            dtype=np.float64,
            shape=(y_size, x_size),
            elements=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
        )
    )
    pressure_data = draw(
        arrays(
            dtype=np.float64,
            shape=(y_size, x_size),
            elements=st.floats(min_value=900, max_value=1100, allow_nan=False, allow_infinity=False),
        )
    )
    humidity_data = draw(
        arrays(
            dtype=np.float64,
            shape=(y_size, x_size),
            elements=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        )
    )

    return {
        "temp_data": temp_data,
        "pressure_data": pressure_data,
        "humidity_data": humidity_data,
        "x_size": x_size,
        "y_size": y_size,
    }


valid_source_name = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(
        whitelist_categories=["Lu", "Ll", "Nd", "Pc"], min_codepoint=ord("A"), max_codepoint=ord("z")
    ),
)


class TestDataset:
    """Test the Dataset class."""

    def test_dataset_creation(self, sample_dataset_class, sample_coordinate_classes, sample_dataarray_classes):
        """Test creating a Dataset instance."""
        WeatherData = sample_dataset_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]
        Temperature, Pressure, Humidity = (
            sample_dataarray_classes["Temperature"],
            sample_dataarray_classes["Pressure"],
            sample_dataarray_classes["Humidity"],
        )

        # Create coordinates
        x_coord = XCoord(data=np.arange(3), name="longitude")
        y_coord = YCoord(data=np.arange(4), name="latitude")

        # Create DataArray instances
        temp_array = Temperature(data=DataGenerator.random_array((4, 3)), x=x_coord, y=y_coord, name="temperature")
        pressure_array = Pressure(
            data=DataGenerator.random_array((4, 3)) * 10 + 1000, x=x_coord, y=y_coord, name="pressure"
        )
        humidity_array = Humidity(
            data=DataGenerator.random_array((4, 3)) * 20 + 50, x=x_coord, y=y_coord, name="humidity"
        )

        weather = WeatherData(
            temperature=temp_array,
            pressure=pressure_array,
            humidity=humidity_array,
            x=x_coord,
            y=y_coord,
        )

        assert weather.temperature.units == "celsius"
        assert weather.pressure.units == "hPa"
        assert weather.humidity.units == "percent"
        assert weather.source == "weather_station"
        assert weather.temperature.data.shape == (4, 3)
        assert weather.pressure.data.shape == (4, 3)
        assert weather.humidity.data.shape == (4, 3)

    @given(coord_data=coordinated_data_arrays(), source=valid_source_name)
    @settings(max_examples=15, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_dataset_creation_property_based(
        self, coord_data, source, sample_dataset_class, sample_coordinate_classes, sample_dataarray_classes
    ):
        """Property-based test for Dataset creation with various data shapes."""
        WeatherData = sample_dataset_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]
        Temperature, Pressure, Humidity = (
            sample_dataarray_classes["Temperature"],
            sample_dataarray_classes["Pressure"],
            sample_dataarray_classes["Humidity"],
        )

        x_size, y_size = coord_data["x_size"], coord_data["y_size"]

        # Create coordinates
        x_coord = XCoord(data=np.arange(x_size), name="longitude")
        y_coord = YCoord(data=np.arange(y_size), name="latitude")

        # Create DataArray instances with property-based data
        temp_array = Temperature(data=coord_data["temp_data"], x=x_coord, y=y_coord, name="temperature")
        pressure_array = Pressure(data=coord_data["pressure_data"], x=x_coord, y=y_coord, name="pressure")
        humidity_array = Humidity(data=coord_data["humidity_data"], x=x_coord, y=y_coord, name="humidity")

        # Create dataset with custom source name
        weather = WeatherData(
            temperature=temp_array,
            pressure=pressure_array,
            humidity=humidity_array,
            x=x_coord,
            y=y_coord,
            source=source,
        )

        # Verify properties
        assert weather.source == source
        assert weather.temperature.data.shape == (y_size, x_size)
        assert weather.pressure.data.shape == (y_size, x_size)
        assert weather.humidity.data.shape == (y_size, x_size)

        # Verify data integrity
        np.testing.assert_array_equal(weather.temperature.data, coord_data["temp_data"])
        np.testing.assert_array_equal(weather.pressure.data, coord_data["pressure_data"])
        np.testing.assert_array_equal(weather.humidity.data, coord_data["humidity_data"])

    def test_dataset_to_xarray(self, sample_dataset_class, sample_coordinate_classes, sample_dataarray_classes):
        """Test converting Dataset to xarray Dataset."""
        WeatherData = sample_dataset_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]
        Temperature, Pressure, Humidity = (
            sample_dataarray_classes["Temperature"],
            sample_dataarray_classes["Pressure"],
            sample_dataarray_classes["Humidity"],
        )

        temp_data = DataGenerator.random_array((4, 3))
        pressure_data = 1000 + DataGenerator.random_array((4, 3)) * 10
        humidity_data = 50 + DataGenerator.random_array((4, 3)) * 20

        # Create coordinates
        x_coord = XCoord(data=np.arange(3), name="longitude")
        y_coord = YCoord(data=np.arange(4), name="latitude")

        # Create DataArray instances
        temp_array = Temperature(data=temp_data, x=x_coord, y=y_coord, name="temperature")
        pressure_array = Pressure(data=pressure_data, x=x_coord, y=y_coord, name="pressure")
        humidity_array = Humidity(data=humidity_data, x=x_coord, y=y_coord, name="humidity")

        weather = WeatherData(
            temperature=temp_array,
            pressure=pressure_array,
            humidity=humidity_array,
            x=x_coord,
            y=y_coord,
        )

        xr_weather = weather.to_xarray()

        assert isinstance(xr_weather, xr.Dataset)

        # Check data variables
        assert "temperature" in xr_weather.data_vars
        assert "pressure" in xr_weather.data_vars
        assert "humidity" in xr_weather.data_vars

        # Check dimensions
        assert xr_weather["temperature"].dims == ("y", "x")
        assert xr_weather["pressure"].dims == ("y", "x")
        assert xr_weather["humidity"].dims == ("y", "x")

        # Check data values
        np.testing.assert_array_equal(xr_weather["temperature"].values, temp_data)
        np.testing.assert_array_equal(xr_weather["pressure"].values, pressure_data)
        np.testing.assert_array_equal(xr_weather["humidity"].values, humidity_data)

        # Check coordinates
        assert "x" in xr_weather.coords
        assert "y" in xr_weather.coords
        np.testing.assert_array_equal(xr_weather.coords["x"].values, np.arange(3))
        np.testing.assert_array_equal(xr_weather.coords["y"].values, np.arange(4))

        # Check attributes
        assert xr_weather.attrs["source"] == "weather_station"

    @given(coord_data=coordinated_data_arrays(), source=valid_source_name)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_dataset_to_xarray_conversion_property_based(
        self, coord_data, source, sample_dataset_class, sample_coordinate_classes, sample_dataarray_classes
    ):
        """Property-based test for Dataset to xarray conversion with data integrity verification."""
        WeatherData = sample_dataset_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]
        Temperature, Pressure, Humidity = (
            sample_dataarray_classes["Temperature"],
            sample_dataarray_classes["Pressure"],
            sample_dataarray_classes["Humidity"],
        )

        x_size, y_size = coord_data["x_size"], coord_data["y_size"]

        # Create coordinates
        x_coord = XCoord(data=np.arange(x_size), name="longitude")
        y_coord = YCoord(data=np.arange(y_size), name="latitude")

        # Create DataArray instances
        temp_array = Temperature(data=coord_data["temp_data"], x=x_coord, y=y_coord, name="temperature")
        pressure_array = Pressure(data=coord_data["pressure_data"], x=x_coord, y=y_coord, name="pressure")
        humidity_array = Humidity(data=coord_data["humidity_data"], x=x_coord, y=y_coord, name="humidity")

        weather = WeatherData(
            temperature=temp_array,
            pressure=pressure_array,
            humidity=humidity_array,
            x=x_coord,
            y=y_coord,
            source=source,
        )

        xr_weather = weather.to_xarray()

        # Verify structure
        assert isinstance(xr_weather, xr.Dataset)
        assert "temperature" in xr_weather.data_vars
        assert "pressure" in xr_weather.data_vars
        assert "humidity" in xr_weather.data_vars

        # Verify data integrity (round-trip accuracy)
        np.testing.assert_array_equal(xr_weather["temperature"].values, coord_data["temp_data"])
        np.testing.assert_array_equal(xr_weather["pressure"].values, coord_data["pressure_data"])
        np.testing.assert_array_equal(xr_weather["humidity"].values, coord_data["humidity_data"])

        # Verify dimensions match
        assert xr_weather["temperature"].dims == ("y", "x")
        assert xr_weather["pressure"].dims == ("y", "x")
        assert xr_weather["humidity"].dims == ("y", "x")

        # Verify coordinates
        assert "x" in xr_weather.coords
        assert "y" in xr_weather.coords
        np.testing.assert_array_equal(xr_weather.coords["x"].values, np.arange(x_size))
        np.testing.assert_array_equal(xr_weather.coords["y"].values, np.arange(y_size))

        # Verify attributes
        assert xr_weather.attrs["source"] == source

    def test_dataset_validation_at_least_one_data_field(self, sample_coordinate_classes, sample_dataset_class):
        """Test that Dataset validation requires at least one DataArray field."""
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        # This should fail - no DataArray fields
        class BadDataset(Dataset):
            x: XCoord  # type: ignore
            y: YCoord  # type: ignore
            source: Attr[str]

        with pytest.raises(ValueError, match="Dataset must have at least one DataArray field"):
            BadDataset(
                x=XCoord(data=np.arange(3), name="longitude"),
                y=YCoord(data=np.arange(4), name="latitude"),
                source="test",  # type: ignore
            )

    @given(source_name=valid_source_name)
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dataset_validation_error_property_based(self, source_name, sample_coordinate_classes):
        """Property-based test for Dataset validation errors."""
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        # Test that dataset without DataArray fields fails consistently
        class EmptyDataset(Dataset):
            x: XCoord  # type: ignore
            y: YCoord  # type: ignore
            source: Attr[str]

        with pytest.raises(ValueError, match="Dataset must have at least one DataArray field"):
            EmptyDataset(
                x=XCoord(data=np.arange(3), name="longitude"),
                y=YCoord(data=np.arange(4), name="latitude"),
                source=source_name,  # type: ignore
            )

    def test_dataset_factory_methods(self, sample_dataset_class, sample_coordinate_classes):
        """Test Dataset factory methods."""
        WeatherData = sample_dataset_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        # Create coordinate instances
        x_coord = XCoord(data=np.arange(3), name="longitude")
        y_coord = YCoord(data=np.arange(4), name="latitude")

        coords_attrs = {
            "x": x_coord,
            "y": y_coord,
        }

        shapes = {"temperature": (4, 3), "pressure": (4, 3), "humidity": (4, 3)}

        # Test zeros_like_fields
        zeros_weather = WeatherData.zeros_like_fields(shapes, **coords_attrs)
        assert isinstance(zeros_weather, xr.Dataset)
        np.testing.assert_array_equal(zeros_weather["temperature"].values, np.zeros((4, 3)))
        np.testing.assert_array_equal(zeros_weather["pressure"].values, np.zeros((4, 3)))
        np.testing.assert_array_equal(zeros_weather["humidity"].values, np.zeros((4, 3)))

        # Test ones_like_fields
        ones_weather = WeatherData.ones_like_fields(shapes, **coords_attrs)
        assert isinstance(ones_weather, xr.Dataset)
        np.testing.assert_array_equal(ones_weather["temperature"].values, np.ones((4, 3)))
        np.testing.assert_array_equal(ones_weather["pressure"].values, np.ones((4, 3)))
        np.testing.assert_array_equal(ones_weather["humidity"].values, np.ones((4, 3)))

    @given(shape_tuple=st.tuples(st.integers(min_value=2, max_value=6), st.integers(min_value=2, max_value=6)))
    @settings(max_examples=8, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dataset_factory_methods_property_based(self, shape_tuple, sample_dataset_class, sample_coordinate_classes):
        """Property-based test for Dataset factory methods with various shapes."""
        WeatherData = sample_dataset_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        y_size, x_size = shape_tuple

        # Create coordinate instances
        x_coord = XCoord(data=np.arange(x_size), name="longitude")
        y_coord = YCoord(data=np.arange(y_size), name="latitude")

        coords_attrs = {
            "x": x_coord,
            "y": y_coord,
        }

        shapes = {"temperature": (y_size, x_size), "pressure": (y_size, x_size), "humidity": (y_size, x_size)}

        # Test zeros_like_fields
        zeros_weather = WeatherData.zeros_like_fields(shapes, **coords_attrs)
        assert isinstance(zeros_weather, xr.Dataset)
        assert zeros_weather["temperature"].shape == (y_size, x_size)
        assert zeros_weather["pressure"].shape == (y_size, x_size)
        assert zeros_weather["humidity"].shape == (y_size, x_size)
        np.testing.assert_array_equal(zeros_weather["temperature"].values, np.zeros((y_size, x_size)))

        # Test ones_like_fields
        ones_weather = WeatherData.ones_like_fields(shapes, **coords_attrs)
        assert isinstance(ones_weather, xr.Dataset)
        assert ones_weather["temperature"].shape == (y_size, x_size)
        np.testing.assert_array_equal(ones_weather["temperature"].values, np.ones((y_size, x_size)))

    @given(coord_data=coordinated_data_arrays(), source=valid_source_name)
    @settings(max_examples=8, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dataset_field_introspection_property_based(
        self, coord_data, source, sample_dataset_class, sample_coordinate_classes, sample_dataarray_classes
    ):
        """Property-based test for Dataset field introspection methods."""
        WeatherData = sample_dataset_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]
        Temperature, Pressure, Humidity = (
            sample_dataarray_classes["Temperature"],
            sample_dataarray_classes["Pressure"],
            sample_dataarray_classes["Humidity"],
        )

        x_size, y_size = coord_data["x_size"], coord_data["y_size"]

        # Create coordinates
        x_coord = XCoord(data=np.arange(x_size), name="longitude")
        y_coord = YCoord(data=np.arange(y_size), name="latitude")

        # Create DataArray instances
        temp_array = Temperature(data=coord_data["temp_data"], x=x_coord, y=y_coord, name="temperature")
        pressure_array = Pressure(data=coord_data["pressure_data"], x=x_coord, y=y_coord, name="pressure")
        humidity_array = Humidity(data=coord_data["humidity_data"], x=x_coord, y=y_coord, name="humidity")

        weather = WeatherData(
            temperature=temp_array,
            pressure=pressure_array,
            humidity=humidity_array,
            x=x_coord,
            y=y_coord,
            source=source,
        )

        # Test field categorization consistency
        dataarray_fields = weather.get_dataarray_model_fields()
        coord_fields = weather.get_coordinate_model_fields()
        attr_fields = weather.get_attr_fields()

        assert len(dataarray_fields) == 3
        assert "temperature" in dataarray_fields
        assert "pressure" in dataarray_fields
        assert "humidity" in dataarray_fields

        assert len(coord_fields) == 2
        assert "x" in coord_fields and "y" in coord_fields

        assert len(attr_fields) == 1
        assert "source" in attr_fields
