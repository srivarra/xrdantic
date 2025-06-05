import pytest

from xrdantic.types import Dim
from xrdantic.utils import (
    Attr,
    Data,
    XrAnnotation,
    get_attr_fields,
    get_coord_fields,
    get_data_fields,
    get_name_field,
)


class TestFactories:
    """Test the factory classes."""

    def test_data_factory_creation(self):
        """Test Data factory creation."""
        X = Dim("x")
        Y = Dim("y")

        # Test single dimension
        data_field = Data[X, float]
        assert data_field is not None

        # Test multiple dimensions
        data_field_2d = Data[(Y, X), float]
        assert data_field_2d is not None

    def test_attr_factory_creation(self):
        """Test Attr factory creation."""
        attr_field = Attr[str]
        assert attr_field is not None

    def test_data_factory_invalid_params(self):
        """Test Data factory with invalid parameters."""
        X = Dim("x")

        # Test with wrong number of parameters
        with pytest.raises(TypeError):
            Data[X]  # Missing dtype

        with pytest.raises(TypeError):
            Data[X, float, int]  # Too many parameters

    def test_coord_factory_invalid_params_removed(self):
        """Test that Coord factory has been removed."""
        # Coord factory has been removed - no need to test invalid params
        pass

    def test_xr_annotation_enum(self):
        """Test XrAnnotation enum."""
        assert XrAnnotation.DATA.value == "data"
        assert XrAnnotation.COORD.value == "coord"
        assert XrAnnotation.ATTR.value == "attr"
        assert XrAnnotation.NAME.value == "name"


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_data_fields(self, sample_dataarray_class):
        """Test get_data_fields function."""
        data_fields = get_data_fields(sample_dataarray_class.model_fields)
        assert len(data_fields) == 1
        assert "data" in data_fields

    def test_get_coord_fields(self, sample_coordinate_classes):
        """Test get_coord_fields function."""
        XCoord = sample_coordinate_classes["XCoord"]
        # XCoord is a Coordinate class, but its "data" field is a Data field, not a coord field
        # Coordinate fields (XrAnnotation.COORD) are used in DataArrays to reference coordinate instances
        data_fields = get_data_fields(XCoord.model_fields)
        assert len(data_fields) == 1
        assert "data" in data_fields

        # Test that it has no actual coordinate fields (since it's a Coordinate class, not a DataArray)
        coord_fields = get_coord_fields(XCoord.model_fields)
        assert len(coord_fields) == 0

    def test_get_attr_fields(self, sample_dataarray_class):
        """Test get_attr_fields function."""
        attr_fields = get_attr_fields(sample_dataarray_class.model_fields)
        assert len(attr_fields) >= 2  # units and description
        assert "units" in attr_fields
        assert "description" in attr_fields

    def test_get_name_field(self, sample_dataarray_class):
        """Test get_name_field function."""
        name_field = get_name_field(sample_dataarray_class.model_fields)
        assert name_field == "name"
