from __future__ import annotations

from functools import cached_property
from typing import Any, Self

import xarray as xr
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic.fields import FieldInfo

from xrdantic.utils import (
    get_attr_fields,
    get_coord_fields,
    get_coordinate_model_fields,
    get_data_fields,
    get_dataarray_model_fields,
    get_dataset_model_fields,
    get_datatree_model_fields,
    get_name_field,
)

# Explicitly control what gets exported for documentation
__all__ = [
    "XrBase",
    "Coordinate",
    "DataArray",
    "Dataset",
    "DataTree",
]


class XrBase(BaseModel):
    """
    Base class for all xrdantic models.

    Provides common functionality for working with xarray-like data structures
    including field introspection, validation, and conversion utilities.

    This class implements the foundation for type-safe, validated xarray-like
    data structures using Pydantic models.
    """

    model_config: ConfigDict = ConfigDict(
        {
            "validate_assignment": True,  # Validate on field assignment
            "validate_default": True,  # Validate default values
            "extra": "forbid",  # Prevent extra fields
            "frozen": False,  # Allow mutability for scientific data
            "arbitrary_types_allowed": True,  # For numpy arrays
            "use_enum_values": True,  # Use enum values for better performance
            "str_strip_whitespace": True,  # Strip whitespace from strings
            "json_schema_serialization_defaults_required": True,  # Optimize JSON schema
            "defer_build": True,  # Defer model rebuilding for better performance
        }
    )

    @cached_property
    def _field_cache(self) -> dict[str, dict[str, FieldInfo]]:
        """Cache field categorization for performance."""
        return {
            "data": get_data_fields(type(self).model_fields),
            "coord": get_coord_fields(type(self).model_fields),
            "attr": get_attr_fields(type(self).model_fields),
            "coordinate_model": get_coordinate_model_fields(type(self).model_fields),
            "dataarray_model": get_dataarray_model_fields(type(self).model_fields),
            "dataset_model": get_dataset_model_fields(type(self).model_fields),
            "datatree_model": get_datatree_model_fields(type(self).model_fields),
        }

    def get_data_fields(self) -> dict[str, FieldInfo]:
        """Get all data fields from this model."""
        return self._field_cache["data"]

    def get_coord_fields(self) -> dict[str, FieldInfo]:
        """Get all coordinate fields from this model."""
        return self._field_cache["coord"]

    def get_attr_fields(self) -> dict[str, FieldInfo]:
        """Get all attribute fields from this model."""
        return self._field_cache["attr"]

    def get_name_field(self) -> str | None:
        """Get the name field from this model."""
        return get_name_field(type(self).model_fields)

    def get_coordinate_model_fields(self) -> dict[str, FieldInfo]:
        """Get all fields that are Coordinate model instances."""
        return self._field_cache["coordinate_model"]

    def get_dataarray_model_fields(self) -> dict[str, FieldInfo]:
        """Get all fields that are DataArray model instances."""
        return self._field_cache["dataarray_model"]

    def get_dataset_model_fields(self) -> dict[str, FieldInfo]:
        """Get all fields that are Dataset model instances."""
        return self._field_cache["dataset_model"]

    def get_datatree_model_fields(self) -> dict[str, FieldInfo]:
        """Get all fields that are DataTree model instances."""
        return self._field_cache["datatree_model"]

    def get_name_value(self) -> Any | None:
        """Get the actual name value from this model."""
        name_field = self.get_name_field()
        return getattr(self, name_field) if name_field else None

    def get_attrs_dict(self) -> dict[str, Any]:
        """Get all attributes as a dictionary."""
        attr_fields = self.get_attr_fields()
        return {k: getattr(self, k) for k in attr_fields.keys()}

    def get_coords_dict(self) -> dict[str, Any]:
        """Get all coordinates as a dictionary."""
        coords = {}

        # Get coordinates from coordinate model fields
        coord_model_fields = self.get_coordinate_model_fields()
        for coord_name in coord_model_fields.keys():
            coord_instance = getattr(self, coord_name)
            dim_name, coord_array = coord_instance.to_xarray_coord()
            coords[dim_name] = coord_array

        # Get direct coordinate fields
        coord_fields = self.get_coord_fields()
        for coord_name, coord_field in coord_fields.items():
            coord_value = getattr(self, coord_name)
            # Safe access to json_schema_extra
            if coord_field.json_schema_extra is not None:
                coord_dims = coord_field.json_schema_extra.get("dims", ())
                if isinstance(coord_dims, tuple | list) and len(coord_dims) == 1:
                    coords[coord_dims[0]] = coord_value

        return coords

    @classmethod
    def get_model_data_fields(cls) -> dict[str, FieldInfo]:
        """Get all data fields from the model class."""
        return get_data_fields(cls.model_fields)

    @classmethod
    def get_model_coord_fields(cls) -> dict[str, FieldInfo]:
        """Get all coordinate fields from the model class."""
        return get_coord_fields(cls.model_fields)

    @classmethod
    def get_model_attr_fields(cls) -> dict[str, FieldInfo]:
        """Get all attribute fields from the model class."""
        return get_attr_fields(cls.model_fields)

    @classmethod
    def get_model_name_field(cls) -> str | None:
        """Get the name field from the model class."""
        return get_name_field(cls.model_fields)

    @classmethod
    def get_model_dataarray_fields(cls) -> dict[str, FieldInfo]:
        """Get all fields that are DataArray model instances from the model class."""
        return get_dataarray_model_fields(cls.model_fields)

    @classmethod
    def get_model_dataset_fields(cls) -> dict[str, FieldInfo]:
        """Get all fields that are Dataset model instances from the model class."""
        return get_dataset_model_fields(cls.model_fields)

    @classmethod
    def get_model_datatree_fields(cls) -> dict[str, FieldInfo]:
        """Get all fields that are DataTree model instances from the model class."""
        return get_datatree_model_fields(cls.model_fields)


class Coordinate(XrBase):
    """
    Base class for coordinate definitions.

    A Coordinate represents a labeled array that provides coordinate values
    for one dimension. It must have exactly one data field with exactly one
    dimension.

    Examples
    --------
        >>> from xrdantic import Coordinate, Data, Name, Attr, Dim
        >>> import numpy as np

        >>> X = Dim("x")
        >>> class XCoord(Coordinate):
        ...     data: Data[X, int]
        ...     name: Name
        ...     units: Attr[str] = "length"

        >>> x_coord = XCoord(data=np.arange(10), name="x")
        >>> dim_name, coord_array = x_coord.to_xarray_coord()

    Raises
    ------
        ValidationError: If the model doesn't have exactly one data field
        ValidationError: If the data field doesn't have exactly one dimension
    """

    @field_validator("*", mode="before")
    @classmethod
    def validate_and_sanitize_data(cls, v, info):
        """Validate and sanitize input data."""
        field_name = info.field_name
        if field_name and "data" in field_name.lower():
            # Import here to avoid circular imports
            from xrdantic.validation import DataValidator

            return DataValidator.sanitize_array_data(v, field_name, is_coordinate=True)
        return v

    def to_xarray_coord(self) -> tuple[str, xr.DataArray]:
        """Convert this coordinate to an xarray coordinate."""
        # Get the data field (for Coordinate classes, data is stored in Data fields)
        data_fields = self.get_data_fields()
        if len(data_fields) != 1:
            from xrdantic.errors import data_field_count_error

            field_count = len(data_fields)
            field_names = list(data_fields.keys())
            raise data_field_count_error("Coordinate", 1, field_count, field_names)

        data_field_name, data_field_info = next(iter(data_fields.items()))
        data_value = getattr(self, data_field_name)

        # Get dimension name from metadata with safe access
        if data_field_info.json_schema_extra is None:
            from xrdantic.errors import missing_dimension_metadata_error

            raise missing_dimension_metadata_error(data_field_name)

        dims = data_field_info.json_schema_extra.get("dims")
        if not dims or not isinstance(dims, tuple | list) or len(dims) != 1:
            from xrdantic.errors import invalid_dimensions_format_error

            raise invalid_dimensions_format_error(data_field_name, dims)

        dim_name = str(dims[0])

        # Get name and attributes
        name_value = self.get_name_value() or dim_name
        attrs = self.get_attrs_dict()

        # Create DataArray
        coord_array = xr.DataArray(data_value, dims=[dim_name], name=name_value, attrs=attrs)

        return dim_name, coord_array

    @model_validator(mode="after")
    def check_data_field(self) -> Self:
        """Validate that this coordinate has exactly one data field with exactly one dimension."""
        # Import here to avoid circular imports
        from xrdantic.validation import DataValidator

        data_fields = self.get_data_fields()
        DataValidator.validate_data_field_count(data_fields, 1, "Coordinate")

        # Get the single data field
        data_field_name, data_field_info = next(iter(data_fields.items()))

        # Validate dimensions
        dims = DataValidator.validate_dimension_metadata(data_field_info, data_field_name)
        DataValidator.validate_coordinate_dimensions(dims, data_field_name)

        # Note: We skip ModelValidator.validate_model_consistency() for coordinates
        # because coordinates are self-defining - they define their own dimension
        # and don't need external coordinates to be consistent.

        return self


class DataArray(XrBase):
    """
    Base class for DataArray definitions.

    A DataArray represents a labeled, multi-dimensional array with coordinates
    and attributes. It must have exactly one data field and can have multiple
    coordinate and attribute fields.

    Examples
    --------
        >>> from xrdantic import DataArray, Data, Attr, Dim
        >>> import numpy as np

        >>> Time = Dim("time")
        >>> Y = Dim("y")
        >>> X = Dim("x")
        >>> class Temperature(DataArray):
        ...     data: Data[(Time, Y, X), float]
        ...     time: TimeCoord
        ...     y: YCoord
        ...     x: XCoord
        ...     units: Attr[str] = "celsius"

        >>> temp = Temperature(data=np.random.random((10, 5, 3)), time=time_coord, y=y_coord, x=x_coord)
        >>> xr_temp = temp.to_xarray()

    Raises
    ------
        ValidationError: If the model doesn't have exactly one data field
        ValidationError: If coordinate dimensions don't match data dimensions
    """

    @field_validator("*", mode="before")
    @classmethod
    def validate_and_sanitize_data(cls, v, info):
        """Validate and sanitize input data."""
        field_name = info.field_name
        if field_name and "data" in field_name.lower():
            from xrdantic.validation import DataValidator

            return DataValidator.sanitize_array_data(v, field_name, is_coordinate=False)
        return v

    def to_xarray(self) -> xr.DataArray:
        """Convert this model to an xarray DataArray."""
        # Get the main data field
        data_fields = self.get_data_fields()
        if len(data_fields) != 1:
            from xrdantic.errors import data_field_count_error

            field_count = len(data_fields)
            field_names = list(data_fields.keys())
            raise data_field_count_error("DataArray", 1, field_count, field_names)

        data_field_name, data_field_info = next(iter(data_fields.items()))
        data_value = getattr(self, data_field_name)

        # Get dimensions from data field metadata with safe access
        if data_field_info.json_schema_extra is None:
            from xrdantic.errors import missing_dimension_metadata_error

            raise missing_dimension_metadata_error(data_field_name)

        dims = data_field_info.json_schema_extra.get("dims", ())
        if not isinstance(dims, tuple | list):
            from xrdantic.errors import invalid_dimensions_format_error

            raise invalid_dimensions_format_error(data_field_name, dims)

        # Convert dims to list of strings for xarray compatibility
        dim_names = [str(dim) for dim in dims]

        # Get coordinates, attributes, and name
        coords = self.get_coords_dict()
        attrs = self.get_attrs_dict()
        name_value = self.get_name_value()

        # Create DataArray
        return xr.DataArray(data_value, dims=dim_names, coords=coords, attrs=attrs, name=name_value)

    @classmethod
    def new(cls, **field_values: Any) -> xr.DataArray:
        r"""
        Create a new DataArray instance with the given field values.

        Parameters
        ----------
        **field_values
            Field values including data and coordinates

        Returns
        -------
        The converted xarray DataArray

        Raises
        ------
        ValidationError
            If required fields are missing or invalid
        """
        instance = cls(**field_values)
        return instance.to_xarray()

    @classmethod
    def zeros(cls, shape: tuple[int, ...], **coords_and_attrs: Any) -> xr.DataArray:
        """Create a DataArray filled with zeros."""
        from xrdantic.factories import DataArrayFactory

        return DataArrayFactory.zeros(cls, shape, **coords_and_attrs)

    @classmethod
    def ones(cls, shape: tuple[int, ...], **coords_and_attrs: Any) -> xr.DataArray:
        """Create a DataArray filled with ones."""
        from xrdantic.factories import DataArrayFactory

        return DataArrayFactory.ones(cls, shape, **coords_and_attrs)

    @classmethod
    def full(cls, shape: tuple[int, ...], fill_value: Any, **coords_and_attrs: Any) -> xr.DataArray:
        """Create a DataArray filled with a constant value."""
        from xrdantic.factories import DataArrayFactory

        return DataArrayFactory.full(cls, shape, fill_value, **coords_and_attrs)

    @classmethod
    def empty(cls, shape: tuple[int, ...], **coords_and_attrs: Any) -> xr.DataArray:
        """Create an uninitialized DataArray."""
        from xrdantic.factories import DataArrayFactory

        return DataArrayFactory.empty(cls, shape, **coords_and_attrs)

    @classmethod
    def random(cls, size: tuple[int, ...], **coords_and_attrs: Any) -> xr.DataArray:
        """Create a DataArray filled with random values."""
        from xrdantic.factories import DataArrayFactory

        return DataArrayFactory.random(cls, size, **coords_and_attrs)

    @model_validator(mode="after")
    def check_data_field(self) -> Self:
        """Validate that this DataArray has exactly one data field."""
        # Import here to avoid circular imports
        from xrdantic.validation import DataValidator, ModelValidator

        data_fields = self.get_data_fields()
        DataValidator.validate_data_field_count(data_fields, 1, "DataArray")

        # Perform comprehensive model consistency validation
        try:
            ModelValidator.validate_model_consistency(self)
        except Exception as e:
            # Convert validation errors to more specific data array errors if appropriate
            if "dimension" in str(e).lower():
                from xrdantic.errors import DimensionError

                data_field_name = next(iter(data_fields.keys())) if data_fields else "unknown"
                raise DimensionError(f"DataArray validation failed: {e}", field_name=data_field_name) from e
            else:
                raise

        return self


class Dataset(XrBase):
    """
    Base class for Dataset definitions.

    A Dataset represents a collection of DataArrays with shared coordinates.
    It must have at least one DataArray field and can have coordinate and
    attribute fields.

    Examples
    --------
        >>> from xrdantic import Dataset, DataArray, Attr, Dim, Data
        >>> import numpy as np
        >>> Time = Dim("time")
        >>> Y = Dim("y")
        >>> X = Dim("x")
        >>> class Temperature(DataArray):
        ...     data: Data[(Time, Y, X), float]
        ...     time: TimeCoord
        ...     y: YCoord
        ...     x: XCoord
        ...     units: Attr[str] = "celsius"
        >>> class Pressure(DataArray):
        ...     data: Data[(Time, Y, X), float]
        ...     time: TimeCoord
        ...     y: YCoord
        ...     x: XCoord
        ...     units: Attr[str] = "pascal"
        >>> class WeatherData(Dataset):
        ...     temperature: Temperature
        ...     pressure: Pressure
        ...     x: XCoord
        ...     y: YCoord
        ...     station_name: Attr[str] = "Station Alpha"
        >>> weather = WeatherData(temperature=temp_array, pressure=pressure_array, x=x_coord, y=y_coord)
        >>> xr_weather = weather.to_xarray()

    Raises
    ------
        ValidationError: If the model doesn't have at least one DataArray field
    """

    def to_xarray(self) -> xr.Dataset:
        """Convert this model to an xarray Dataset."""
        # Get DataArray fields (composition is the only way to define data variables)
        dataarray_fields = self.get_dataarray_model_fields()

        if len(dataarray_fields) == 0:
            from xrdantic.errors import missing_dataarray_fields_error

            raise missing_dataarray_fields_error("Dataset")

        data_vars = {}

        # Process each DataArray field
        for dataarray_field_name in dataarray_fields.keys():
            dataarray_instance = getattr(self, dataarray_field_name)
            # Convert the DataArray instance to xarray and extract its data and dims
            xr_dataarray = dataarray_instance.to_xarray()
            data_vars[dataarray_field_name] = (xr_dataarray.dims, xr_dataarray.data)

        # Get coordinates and attributes
        coords = self.get_coords_dict()
        attrs = self.get_attrs_dict()

        # Create Dataset
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    @classmethod
    def new(cls, **dataarray_instances: Any) -> xr.Dataset:
        """Create a new Dataset instance with the given DataArray instances."""
        instance = cls(**dataarray_instances)
        return instance.to_xarray()

    @classmethod
    def zeros_like_fields(cls, shapes: dict[str, tuple[int, ...]], **coords_and_attrs: Any) -> xr.Dataset:
        """Create a Dataset with zeros for each DataArray field."""
        from xrdantic.factories import DatasetFactory

        return DatasetFactory.zeros_like_fields(cls, shapes, **coords_and_attrs)

    @classmethod
    def ones_like_fields(cls, shapes: dict[str, tuple[int, ...]], **coords_and_attrs: Any) -> xr.Dataset:
        """Create a Dataset with ones for each DataArray field."""
        from xrdantic.factories import DatasetFactory

        return DatasetFactory.ones_like_fields(cls, shapes, **coords_and_attrs)

    @model_validator(mode="after")
    def check_data_fields(self) -> Self:
        """Validate that Dataset has at least one DataArray field."""
        # Import here to avoid circular imports
        from xrdantic.validation import ModelValidator

        dataarray_fields = self.get_dataarray_model_fields()
        ModelValidator.validate_dataarray_fields(dataarray_fields, "Dataset")

        # Perform comprehensive model consistency validation
        try:
            ModelValidator.validate_model_consistency(self)
        except Exception as e:
            # Convert validation errors to more specific dataset errors if appropriate
            if "dimension" in str(e).lower() or "coordinate" in str(e).lower():
                from xrdantic.errors import DimensionError

                raise DimensionError(
                    f"Dataset validation failed: {e}",
                    expected_dims=tuple(dataarray_fields.keys()) if dataarray_fields else (),
                ) from e
            else:
                raise

        return self


class DataTree(XrBase):
    """
    Base class for DataTree definitions.

    A DataTree represents a hierarchical collection of Datasets and other DataTrees,
    similar to xarray.DataTree. It can contain Dataset fields, other DataTree fields,
    coordinate fields, and attribute fields.

    Examples
    --------
        >>> class SimulationTree(DataTree):
        ...     coarse_grid: WeatherDataset
        ...     fine_grid: WeatherDataset
        ...     metadata: MetadataDataset
        ...     simulation_name: Attr[str] = "default"

        >>> sim_tree = SimulationTree(coarse_grid=coarse_dataset, fine_grid=fine_dataset, metadata=meta_dataset)
        >>> xr_tree = sim_tree.to_xarray()

    Raises
    ------
        ValidationError: If the model doesn't have at least one Dataset or DataTree field
    """

    def to_xarray(self) -> xr.DataTree:
        """Convert this model to an xarray DataTree."""
        # Get Dataset and DataTree fields
        dataset_fields = self.get_dataset_model_fields()
        datatree_fields = self.get_datatree_model_fields()

        if len(dataset_fields) == 0 and len(datatree_fields) == 0:
            from xrdantic.errors import missing_dataset_fields_error

            raise missing_dataset_fields_error("DataTree")

        # Create a dictionary to store datasets/datatrees
        tree_dict = {}

        # Process Dataset fields
        for dataset_field_name in dataset_fields.keys():
            dataset_instance = getattr(self, dataset_field_name)
            # Convert the Dataset instance to xarray
            xr_dataset = dataset_instance.to_xarray()
            tree_dict[dataset_field_name] = xr_dataset

        # Process DataTree fields (nested trees)
        for datatree_field_name in datatree_fields.keys():
            datatree_instance = getattr(self, datatree_field_name)
            # Convert the DataTree instance to xarray recursively
            xr_datatree = datatree_instance.to_xarray()
            tree_dict[datatree_field_name] = xr_datatree

        # Get root-level coordinates and attributes
        coords = self.get_coords_dict()
        attrs = self.get_attrs_dict()
        name_value = self.get_name_value()

        # Create DataTree from dictionary
        if coords or attrs:
            # Create root dataset with coordinates and attributes
            root_dataset = xr.Dataset(coords=coords, attrs=attrs)
            tree_dict["/"] = root_dataset

        # Create DataTree from the dictionary
        return xr.DataTree.from_dict(tree_dict, name=name_value)

    @classmethod
    def new(cls, **dataset_and_datatree_instances: Any) -> xr.DataTree:
        """Create a new DataTree instance with the given Dataset and DataTree instances."""
        instance = cls(**dataset_and_datatree_instances)
        return instance.to_xarray()

    @model_validator(mode="after")
    def check_tree_fields(self) -> Self:
        """Validate that DataTree has at least one Dataset or DataTree field."""
        # Import here to avoid circular imports
        from xrdantic.validation import ModelValidator

        dataset_fields = self.get_dataset_model_fields()
        datatree_fields = self.get_datatree_model_fields()

        # Check that we have at least one Dataset or DataTree field
        if len(dataset_fields) == 0 and len(datatree_fields) == 0:
            from xrdantic.errors import missing_dataset_fields_error

            raise missing_dataset_fields_error("DataTree")

        try:
            ModelValidator.validate_model_consistency(self)
        except Exception as e:
            # Convert validation errors to more specific datatree errors if appropriate
            if "dimension" in str(e).lower() or "coordinate" in str(e).lower():
                from xrdantic.errors import DimensionError

                all_fields = list(dataset_fields.keys()) + list(datatree_fields.keys())
                raise DimensionError(
                    f"DataTree validation failed: {e}",
                    expected_dims=tuple(all_fields) if all_fields else (),
                ) from e
            else:
                raise

        return self
