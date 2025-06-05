"""
Comprehensive tests for Coordinate class using property-based testing.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from xrdantic.models import Coordinate
from xrdantic.utils import Attr, Data, Name

from ..utils import DataGenerator  # noqa: TID252

# ===== Hypothesis Strategies =====


@st.composite
def valid_coordinate_data(draw, min_size=1, max_size=50):
    """Generate valid 1D array data for coordinate testing."""
    dtype = draw(st.sampled_from([np.int32, np.int64, np.float32, np.float64]))
    elements = (
        st.integers(min_value=-1000, max_value=1000)
        if np.issubdtype(dtype, np.integer)
        else st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    )
    shape = draw(array_shapes(min_dims=1, max_dims=1, min_side=min_size, max_side=max_size))
    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


valid_coord_name = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"], min_codepoint=ord("a"), max_codepoint=ord("z")),
)


class TestCoordinate:
    """Test the Coordinate class."""

    def test_coordinate_creation(self, sample_coordinate_classes):
        """Test creating a Coordinate instance."""
        XCoord = sample_coordinate_classes["XCoord"]

        coord = XCoord(data=np.arange(5), name="x_axis")

        assert coord.name == "x_axis"
        assert coord.units == "pixels"
        assert coord.long_name == "X coordinate"
        np.testing.assert_array_equal(coord.data, np.arange(5))

    @given(coord_data=valid_coordinate_data(), coord_name=valid_coord_name)
    @settings(max_examples=25, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_coordinate_creation_property_based(self, coord_data, coord_name, sample_coordinate_classes):
        """Property-based test for coordinate creation with various data types."""
        XCoord = sample_coordinate_classes["XCoord"]

        # Convert to int for XCoord which expects int data
        int_data = coord_data.astype(np.int32)
        coord = XCoord(data=int_data, name=coord_name)

        assert coord.name == coord_name
        assert coord.units == "pixels"
        assert coord.long_name == "X coordinate"
        np.testing.assert_array_equal(coord.data, int_data)

        # Verify field categorization
        assert len(coord.get_data_fields()) == 1
        assert "data" in coord.get_data_fields()

    def test_coordinate_to_xarray_coord(self, sample_coordinate_classes):
        """Test converting coordinate to xarray coordinate."""
        XCoord = sample_coordinate_classes["XCoord"]

        coord = XCoord(data=np.arange(3), name="x_axis")

        dim_name, coord_array = coord.to_xarray_coord()

        assert dim_name == "x"
        assert isinstance(coord_array, xr.DataArray)
        assert coord_array.name == "x_axis"
        assert coord_array.dims == ("x",)
        assert coord_array.attrs["units"] == "pixels"
        assert coord_array.attrs["long_name"] == "X coordinate"
        np.testing.assert_array_equal(coord_array.values, np.arange(3))

    @given(coord_data=valid_coordinate_data(), coord_name=valid_coord_name)
    @settings(max_examples=15, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_coordinate_to_xarray_conversion_property_based(self, coord_data, coord_name, sample_coordinate_classes):
        """Property-based test for xarray conversion maintaining data integrity."""
        XCoord = sample_coordinate_classes["XCoord"]

        int_data = coord_data.astype(np.int32)
        coord = XCoord(data=int_data, name=coord_name)
        dim_name, coord_array = coord.to_xarray_coord()

        # Verify structure
        assert dim_name == "x"
        assert isinstance(coord_array, xr.DataArray)
        assert coord_array.name == coord_name
        assert coord_array.dims == ("x",)

        # Verify data integrity (round-trip accuracy)
        np.testing.assert_array_equal(coord_array.values, int_data)

        # Verify metadata
        assert coord_array.attrs["units"] == "pixels"
        assert coord_array.attrs["long_name"] == "X coordinate"

    def test_coordinate_validation_single_data_field(self, sample_dims, sample_coordinate_classes):
        """Test that coordinate validation requires exactly one data field."""
        X = sample_dims["X"]

        # This should fail - no data field
        class BadCoord(Coordinate):
            name: Name
            units: Attr[str] = "pixels"

        with pytest.raises(ValueError, match="exactly one data field"):
            BadCoord(name="test")

        # This should fail - multiple data fields
        class BadCoord2(Coordinate):
            data1: Data[X, int]
            data2: Data[X, float]
            name: Name

        with pytest.raises(ValueError, match="exactly one data field"):
            BadCoord2(
                data1=DataGenerator.random_array((3,), dtype=np.int32),  # type: ignore
                data2=DataGenerator.random_array((3,), dtype=np.float32),  # type: ignore
                name="test",
            )

    def test_coordinate_validation_single_dimension(self, sample_dims, sample_coordinate_classes):
        """Test that coordinate validation requires exactly one dimension."""
        X, Y = sample_dims["X"], sample_dims["Y"]

        # This should fail - multiple dimensions
        class BadCoord(Coordinate):
            data: Data[(X, Y), int]  # Multiple dimensions not allowed for coordinates
            name: Name

        with pytest.raises(ValueError, match="1-dimensional"):
            BadCoord(data=DataGenerator.random_array((2, 3)).astype(np.int32), name="test")  # type: ignore

    @given(coord_data=valid_coordinate_data(min_size=2, max_size=10))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinate_validation_multidimensional_error_property_based(self, coord_data, sample_dims):
        """Property-based test for multidimensional coordinate validation errors."""
        X, Y = sample_dims["X"], sample_dims["Y"]

        class BadCoord(Coordinate):
            data: Data[(X, Y), float]
            name: Name

        # Reshape to 2D to trigger error
        if coord_data.size >= 4:
            reshaped_data = coord_data[:4].reshape(2, 2)
            with pytest.raises(ValueError, match="1-dimensional"):
                BadCoord(data=reshaped_data, name="test")  # type: ignore

    def test_coordinate_get_methods(self, sample_coordinate_classes):
        """Test the get_* methods from XrBase."""
        XCoord = sample_coordinate_classes["XCoord"]

        coord = XCoord(data=np.arange(3), name="test")

        # Test instance methods - for coordinates, data is stored in data fields
        data_fields = coord.get_data_fields()
        assert len(data_fields) == 1
        assert "data" in data_fields

        attr_fields = coord.get_attr_fields()
        assert len(attr_fields) == 2
        assert "units" in attr_fields
        assert "long_name" in attr_fields

        name_field = coord.get_name_field()
        assert name_field == "name"

        name_value = coord.get_name_value()
        assert name_value == "test"

        attrs_dict = coord.get_attrs_dict()
        assert attrs_dict["units"] == "pixels"
        assert attrs_dict["long_name"] == "X coordinate"

        # Test class methods - for coordinates, data is stored in data fields
        model_data_fields = XCoord.get_model_data_fields()
        assert len(model_data_fields) == 1
        assert "data" in model_data_fields

    @given(coord_data=valid_coordinate_data(), coord_name=valid_coord_name)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinate_field_introspection_property_based(self, coord_data, coord_name, sample_coordinate_classes):
        """Property-based test for coordinate field introspection methods."""
        XCoord = sample_coordinate_classes["XCoord"]

        int_data = coord_data.astype(np.int32)
        coord = XCoord(data=int_data, name=coord_name)

        # Test field categorization consistency
        data_fields = coord.get_data_fields()
        attr_fields = coord.get_attr_fields()

        assert len(data_fields) == 1
        assert "data" in data_fields
        assert len(attr_fields) == 2
        assert "units" in attr_fields
        assert "long_name" in attr_fields

        # Test name handling
        assert coord.get_name_field() == "name"
        assert coord.get_name_value() == coord_name

    def test_coordinate_error_scenarios_comprehensive(self, sample_dims):
        """Test comprehensive coordinate error scenarios."""
        X = sample_dims["X"]

        # Test infinite values - empty arrays are currently accepted by the system
        class TestCoord(Coordinate):
            data: Data[X, float]
            name: Name

        with pytest.raises(ValueError, match="infinite"):
            TestCoord(data=np.array([1.0, np.inf, 3.0]), name="test")  # type: ignore

    @given(
        float_list=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=20
        )
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinate_list_to_array_conversion_property_based(self, float_list, sample_coordinate_classes):
        """Property-based test for automatic list to array conversion."""
        XCoord = sample_coordinate_classes["XCoord"]

        # Convert to int for XCoord
        int_list = [int(x) for x in float_list]
        coord = XCoord(data=int_list, name="test")  # type: ignore

        assert isinstance(coord.data, np.ndarray)
        np.testing.assert_array_equal(coord.data, np.array(int_list))
