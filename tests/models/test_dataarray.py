"""
Comprehensive tests for DataArray class using property-based testing.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from xrdantic.utils import Data, Name

from ..utils import DataGenerator  # noqa: TID252

# ===== Hypothesis Strategies =====


@st.composite
def valid_2d_array_data(draw, min_size=2, max_size=20):
    """Generate valid 2D array data for DataArray testing."""
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    elements = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    shape = draw(array_shapes(min_dims=2, max_dims=2, min_side=min_size, max_side=max_size))
    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


@st.composite
def valid_coordinate_data(draw, min_size=1, max_size=50):
    """Generate valid 1D coordinate data."""
    dtype = draw(st.sampled_from([np.int32, np.int64]))
    elements = st.integers(min_value=-100, max_value=100)
    shape = draw(array_shapes(min_dims=1, max_dims=1, min_side=min_size, max_side=max_size))
    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


valid_da_name = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"], min_codepoint=ord("a"), max_codepoint=ord("z")),
)


class TestDataArray:
    """Test the DataArray class."""

    def test_dataarray_creation(self, sample_dataarray_class, sample_coordinate_classes):
        """Test creating a DataArray instance."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        image = Image2D(
            data=DataGenerator.random_array((4, 3)),
            x=XCoord(data=np.arange(3), name="x"),
            y=YCoord(data=np.arange(4), name="y"),
            name="test_image",
        )

        assert image.name == "test_image"
        assert image.units == "intensity"
        assert image.description == "2D image data"
        assert image.data.shape == (4, 3)

    @given(array_2d=valid_2d_array_data(min_size=2, max_size=8), da_name=valid_da_name)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_dataarray_creation_property_based(
        self, array_2d, da_name, sample_dataarray_class, sample_coordinate_classes
    ):
        """Property-based test for DataArray creation with various data shapes."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        y_size, x_size = array_2d.shape

        x_coord = XCoord(data=np.arange(x_size), name="x")
        y_coord = YCoord(data=np.arange(y_size), name="y")

        image = Image2D(
            data=array_2d,
            x=x_coord,
            y=y_coord,
            name=da_name,
        )

        assert image.name == da_name
        assert image.units == "intensity"
        assert image.description == "2D image data"
        assert image.data.shape == (y_size, x_size)
        np.testing.assert_array_equal(image.data, array_2d)

        # Verify field categorization
        assert len(image.get_data_fields()) == 1
        assert "data" in image.get_data_fields()

    def test_dataarray_to_xarray(self, sample_dataarray_class, sample_coordinate_classes):
        """Test converting DataArray to xarray DataArray."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        data = DataGenerator.random_array((4, 3))
        image = Image2D(
            data=data, x=XCoord(data=np.arange(3), name="x"), y=YCoord(data=np.arange(4), name="y"), name="test_image"
        )

        xr_image = image.to_xarray()

        assert isinstance(xr_image, xr.DataArray)
        assert xr_image.name == "test_image"
        assert xr_image.dims == ("y", "x")
        assert xr_image.attrs["units"] == "intensity"
        assert xr_image.attrs["description"] == "2D image data"
        np.testing.assert_array_equal(xr_image.values, data)

        # Check coordinates
        assert "x" in xr_image.coords
        assert "y" in xr_image.coords
        np.testing.assert_array_equal(xr_image.coords["x"].values, np.arange(3))
        np.testing.assert_array_equal(xr_image.coords["y"].values, np.arange(4))

    @given(array_2d=valid_2d_array_data(min_size=2, max_size=8), da_name=valid_da_name)
    @settings(max_examples=15, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_dataarray_to_xarray_conversion_property_based(
        self, array_2d, da_name, sample_dataarray_class, sample_coordinate_classes
    ):
        """Property-based test for xarray conversion maintaining data integrity."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        y_size, x_size = array_2d.shape

        x_coord = XCoord(data=np.arange(x_size), name="x")
        y_coord = YCoord(data=np.arange(y_size), name="y")

        image = Image2D(data=array_2d, x=x_coord, y=y_coord, name=da_name)
        xr_image = image.to_xarray()

        # Verify structure
        assert isinstance(xr_image, xr.DataArray)
        assert xr_image.name == da_name
        assert xr_image.dims == ("y", "x")

        # Verify data integrity (round-trip accuracy)
        np.testing.assert_array_equal(xr_image.values, array_2d)

        # Verify metadata
        assert xr_image.attrs["units"] == "intensity"
        assert xr_image.attrs["description"] == "2D image data"

        # Verify coordinates
        assert "x" in xr_image.coords
        assert "y" in xr_image.coords
        np.testing.assert_array_equal(xr_image.coords["x"].values, np.arange(x_size))
        np.testing.assert_array_equal(xr_image.coords["y"].values, np.arange(y_size))

    def test_dataarray_validation_single_data_field(
        self, sample_dims, sample_coordinate_classes, sample_dataarray_class
    ):
        """Test that DataArray validation requires exactly one data field."""
        X, Y = sample_dims["X"], sample_dims["Y"]
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        # This should fail - no data field
        from xrdantic.models import DataArray

        class BadDataArray(DataArray):
            x: XCoord  # type: ignore
            y: YCoord  # type: ignore
            name: Name

        with pytest.raises(ValueError, match="exactly one data field"):
            BadDataArray(x=XCoord(data=np.arange(3), name="x"), y=YCoord(data=np.arange(4), name="y"), name="test")

        # This should fail - multiple data fields
        class BadDataArray2(DataArray):
            data1: Data[(Y, X), float]
            data2: Data[(Y, X), float]
            x: XCoord  # type: ignore
            y: YCoord  # type: ignore
            name: Name

        with pytest.raises(ValueError, match="exactly one data field"):
            BadDataArray2(
                data1=DataGenerator.random_array((4, 3)),  # type: ignore
                data2=DataGenerator.random_array((4, 3)),  # type: ignore
                x=XCoord(data=np.arange(3), name="x"),
                y=YCoord(data=np.arange(4), name="y"),
                name="test",
            )

    @given(array_2d=valid_2d_array_data(min_size=3, max_size=8))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dataarray_dimension_mismatch_property_based(self, array_2d, sample_coordinate_classes):
        """Property-based test for dimension mismatch validation."""
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        y_size, x_size = array_2d.shape

        # Create coordinates with different sizes to trigger mismatch
        if x_size > 2 and y_size > 2:
            wrong_x_coord = XCoord(data=np.arange(x_size - 1), name="x")  # Wrong size
            correct_y_coord = YCoord(data=np.arange(y_size), name="y")

            from xrdantic.models import DataArray
            from xrdantic.types import Dim

            X, Y = Dim("x"), Dim("y")

            class TestDataArray(DataArray):
                data: Data[(Y, X), float]
                x: XCoord  # type: ignore
                y: YCoord  # type: ignore
                name: Name

            with pytest.raises(ValueError):  # Should catch dimension mismatch
                TestDataArray(data=array_2d, x=wrong_x_coord, y=correct_y_coord, name="test")

    def test_dataarray_factory_methods(self, sample_dataarray_class, sample_coordinate_classes):
        """Test DataArray factory methods."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        coords_attrs = {
            "x": XCoord(data=np.arange(3), name="x"),
            "y": YCoord(data=np.arange(4), name="y"),
            "name": "test_image",
        }

        # Test zeros
        zeros_image = Image2D.zeros((4, 3), **coords_attrs)
        assert isinstance(zeros_image, xr.DataArray)
        np.testing.assert_array_equal(zeros_image.values, np.zeros((4, 3)))

        # Test ones
        ones_image = Image2D.ones((4, 3), **coords_attrs)
        assert isinstance(ones_image, xr.DataArray)
        np.testing.assert_array_equal(ones_image.values, np.ones((4, 3)))

        # Test full
        full_image = Image2D.full((4, 3), 5.0, **coords_attrs)
        assert isinstance(full_image, xr.DataArray)
        np.testing.assert_array_equal(full_image.values, np.full((4, 3), 5.0))

        # Test empty (just check shape)
        empty_image = Image2D.empty((4, 3), **coords_attrs)
        assert isinstance(empty_image, xr.DataArray)
        assert empty_image.shape == (4, 3)

        # Test random (just check shape)
        random_image = Image2D.random((4, 3), **coords_attrs)
        assert isinstance(random_image, xr.DataArray)
        assert random_image.shape == (4, 3)

    @given(shape_tuple=st.tuples(st.integers(min_value=2, max_value=8), st.integers(min_value=2, max_value=8)))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dataarray_factory_methods_property_based(
        self, shape_tuple, sample_dataarray_class, sample_coordinate_classes
    ):
        """Property-based test for DataArray factory methods with various shapes."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        y_size, x_size = shape_tuple

        coords_attrs = {
            "x": XCoord(data=np.arange(x_size), name="x"),
            "y": YCoord(data=np.arange(y_size), name="y"),
            "name": "test_image",
        }

        # Test zeros
        zeros_image = Image2D.zeros((y_size, x_size), **coords_attrs)
        assert isinstance(zeros_image, xr.DataArray)
        assert zeros_image.shape == (y_size, x_size)
        np.testing.assert_array_equal(zeros_image.values, np.zeros((y_size, x_size)))

        # Test ones
        ones_image = Image2D.ones((y_size, x_size), **coords_attrs)
        assert isinstance(ones_image, xr.DataArray)
        assert ones_image.shape == (y_size, x_size)
        np.testing.assert_array_equal(ones_image.values, np.ones((y_size, x_size)))

    def test_dataarray_get_methods(self, sample_dataarray_class, sample_coordinate_classes):
        """Test the get_* methods from XrBase."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        image = Image2D(
            data=DataGenerator.random_array((4, 3)),
            x=XCoord(data=np.arange(3), name="x"),
            y=YCoord(data=np.arange(4), name="y"),
            name="test_image",
        )

        # Test instance methods
        data_fields = image.get_data_fields()
        assert len(data_fields) == 1
        assert "data" in data_fields

        coord_model_fields = image.get_coordinate_model_fields()
        assert len(coord_model_fields) == 2
        assert "x" in coord_model_fields
        assert "y" in coord_model_fields

        attr_fields = image.get_attr_fields()
        assert len(attr_fields) == 2
        assert "units" in attr_fields
        assert "description" in attr_fields

        coords_dict = image.get_coords_dict()
        assert "x" in coords_dict
        assert "y" in coords_dict
        assert isinstance(coords_dict["x"], xr.DataArray)
        assert isinstance(coords_dict["y"], xr.DataArray)

    @given(array_2d=valid_2d_array_data(min_size=2, max_size=8), da_name=valid_da_name)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dataarray_field_introspection_property_based(
        self, array_2d, da_name, sample_dataarray_class, sample_coordinate_classes
    ):
        """Property-based test for DataArray field introspection methods."""
        Image2D = sample_dataarray_class
        XCoord, YCoord = sample_coordinate_classes["XCoord"], sample_coordinate_classes["YCoord"]

        y_size, x_size = array_2d.shape

        image = Image2D(
            data=array_2d,
            x=XCoord(data=np.arange(x_size), name="x"),
            y=YCoord(data=np.arange(y_size), name="y"),
            name=da_name,
        )

        # Test field categorization consistency
        data_fields = image.get_data_fields()
        coord_fields = image.get_coordinate_model_fields()
        attr_fields = image.get_attr_fields()

        assert len(data_fields) == 1
        assert "data" in data_fields
        assert len(coord_fields) == 2
        assert "x" in coord_fields and "y" in coord_fields
        assert len(attr_fields) == 2
        assert "units" in attr_fields and "description" in attr_fields

        # Test name handling
        assert image.get_name_field() == "name"
        assert image.get_name_value() == da_name
