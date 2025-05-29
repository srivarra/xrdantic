"""
Factory methods for creating xarray data structures.

This module provides factory methods for creating DataArrays and Datasets
with common patterns like zeros, ones, random data, etc. Enhanced with
comprehensive validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numpydantic import NDArray

from xrdantic.validation import DataValidator, ModelValidator


class ArrayFactory:
    """Factory for creating arrays with common patterns and validation."""

    @staticmethod
    def zeros(shape: tuple[int, ...]) -> NDArray:
        """Create an array filled with zeros with validation."""
        DataValidator.validate_shape(shape, "zeros")
        array = np.zeros(shape)
        return DataValidator.sanitize_array_data(array, "zeros_array", is_coordinate=False)

    @staticmethod
    def ones(shape: tuple[int, ...]) -> NDArray:
        """Create an array filled with ones with validation."""
        DataValidator.validate_shape(shape, "ones")
        array = np.ones(shape)
        return DataValidator.sanitize_array_data(array, "ones_array", is_coordinate=False)

    @staticmethod
    def full(shape: tuple[int, ...], fill_value: Any) -> NDArray:
        """Create an array filled with a constant value with validation."""
        DataValidator.validate_shape(shape, "full")
        array = np.full(shape, fill_value)
        return DataValidator.sanitize_array_data(array, "full_array", is_coordinate=False)

    @staticmethod
    def empty(shape: tuple[int, ...]) -> NDArray:
        """Create an uninitialized array with validation."""
        DataValidator.validate_shape(shape, "empty")
        array = np.empty(shape)
        return DataValidator.sanitize_array_data(array, "empty_array", is_coordinate=False)

    @staticmethod
    def random(size: tuple[int, ...]) -> NDArray:
        """Create an array filled with random values with validation."""
        DataValidator.validate_shape(size, "random")
        rng = np.random.default_rng()
        array = rng.random(size)
        return DataValidator.sanitize_array_data(array, "random_array", is_coordinate=False)


class DataArrayFactory:
    """Factory for creating DataArray instances with comprehensive validation."""

    @classmethod
    def create_with_data(
        cls, dataarray_class: type, data: NDArray, validate_model_consistency: bool = True, **coords_and_attrs: Any
    ) -> xr.DataArray:
        """
        Create a DataArray instance with the given data and validation.

        Parameters
        ----------
        dataarray_class : type
            DataArray class to create
        data : NDArray
            Data array
        validate_model_consistency : bool, default True
            Whether to validate model consistency after creation
        **coords_and_attrs
            Coordinates and attributes

        Returns
        -------
        xr.DataArray
            Created and validated DataArray
        """
        # Validate shape matches expected dimensions if available
        data_fields = dataarray_class.get_model_data_fields()
        if data_fields:
            data_field_info = next(iter(data_fields.values()))
            if data_field_info.json_schema_extra is not None:
                expected_dims = data_field_info.json_schema_extra.get("dims", ())
                if isinstance(expected_dims, tuple | list) and len(data.shape) != len(expected_dims):  # type: ignore
                    DataValidator.validate_shape_matches_dimensions(
                        data.shape,  # type: ignore
                        tuple(str(dim) for dim in expected_dims),
                        f"{dataarray_class.__name__} creation",
                    )

        # Validate and sanitize input data
        validated_data = DataValidator.sanitize_array_data(
            data, f"{dataarray_class.__name__}_data", is_coordinate=False
        )

        # Create the instance
        instance = dataarray_class(data=validated_data, **coords_and_attrs)

        # Optionally validate model consistency
        if validate_model_consistency:
            try:
                ModelValidator.validate_model_consistency(instance)
            except Exception as e:
                raise ValueError(f"Created {dataarray_class.__name__} failed consistency validation: {e}") from e

        return instance.to_xarray()

    @classmethod
    def zeros(
        cls,
        dataarray_class: type,
        shape: tuple[int, ...],
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.DataArray:
        """Create a DataArray filled with zeros with validation."""
        data = ArrayFactory.zeros(shape)
        return cls.create_with_data(dataarray_class, data, validate_model_consistency, **coords_and_attrs)

    @classmethod
    def ones(
        cls,
        dataarray_class: type,
        shape: tuple[int, ...],
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.DataArray:
        """Create a DataArray filled with ones with validation."""
        data = ArrayFactory.ones(shape)
        return cls.create_with_data(dataarray_class, data, validate_model_consistency, **coords_and_attrs)

    @classmethod
    def full(
        cls,
        dataarray_class: type,
        shape: tuple[int, ...],
        fill_value: Any,
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.DataArray:
        """Create a DataArray filled with a constant value with validation."""
        data = ArrayFactory.full(shape, fill_value)
        return cls.create_with_data(dataarray_class, data, validate_model_consistency, **coords_and_attrs)

    @classmethod
    def empty(
        cls,
        dataarray_class: type,
        shape: tuple[int, ...],
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.DataArray:
        """Create an uninitialized DataArray with validation."""
        data = ArrayFactory.empty(shape)
        return cls.create_with_data(dataarray_class, data, validate_model_consistency, **coords_and_attrs)

    @classmethod
    def random(
        cls,
        dataarray_class: type,
        size: tuple[int, ...],
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.DataArray:
        """Create a DataArray filled with random values with validation."""
        data = ArrayFactory.random(size)
        return cls.create_with_data(dataarray_class, data, validate_model_consistency, **coords_and_attrs)


class DatasetFactory:
    """Factory for creating Dataset instances with comprehensive validation."""

    @classmethod
    def create_with_shapes(
        cls,
        dataset_class: type,
        shapes: dict[str, tuple[int, ...]],
        factory_method: str,
        fill_value: Any = None,
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.Dataset:
        """
        Create a Dataset using the specified factory method for each DataArray field.

        Parameters
        ----------
        dataset_class : type
            Dataset class to create
        shapes : dict[str, tuple[int, ...]]
            Shapes for each DataArray field
        factory_method : str
            Factory method to use ("zeros", "ones", "full")
        fill_value : Any, optional
            Fill value for "full" method
        validate_model_consistency : bool, default True
            Whether to validate model consistency
        **coords_and_attrs
            Coordinates and attributes

        Returns
        -------
        xr.Dataset
            Created and validated Dataset
        """
        dataarray_fields = dataset_class.get_model_dataarray_fields()
        data_vars = {}
        coords = {}
        attrs = {}

        for field_name, field_info in dataarray_fields.items():
            if field_name not in shapes:
                from xrdantic.errors import missing_shape_for_field_error

                raise missing_shape_for_field_error(field_name)

            dataarray_class = field_info.annotation
            if dataarray_class is None:
                from xrdantic.errors import missing_annotation_error

                raise missing_annotation_error(field_name)

            # Use the appropriate factory method to create xarray DataArrays
            # Note: Pass validate_model_consistency=False here to avoid double validation
            if factory_method == "zeros":
                data_vars[field_name] = DataArrayFactory.zeros(
                    dataarray_class,
                    shapes[field_name],
                    validate_model_consistency=False,
                    name=field_name,
                    **coords_and_attrs,
                )
            elif factory_method == "ones":
                data_vars[field_name] = DataArrayFactory.ones(
                    dataarray_class,
                    shapes[field_name],
                    validate_model_consistency=False,
                    name=field_name,
                    **coords_and_attrs,
                )
            elif factory_method == "full" and fill_value is not None:
                data_vars[field_name] = DataArrayFactory.full(
                    dataarray_class,
                    shapes[field_name],
                    fill_value,
                    validate_model_consistency=False,
                    name=field_name,
                    **coords_and_attrs,
                )
            else:
                from xrdantic.errors import unsupported_factory_method_error

                raise unsupported_factory_method_error(factory_method)

        # Extract coordinates from coords_and_attrs
        for key, value in coords_and_attrs.items():
            if hasattr(value, "to_xarray"):  # It's a Coordinate instance
                coords[key] = value.to_xarray()
            elif key in ["name"]:  # Skip name as it's for individual DataArrays
                continue
            else:  # It's an attribute
                attrs[key] = value

        # Create the dataset
        dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Note: We skip dataset instance validation for now since Dataset doesn't have from_xarray
        # The individual DataArray validations already ensure consistency

        return dataset

    @classmethod
    def zeros_like_fields(
        cls,
        dataset_class: type,
        shapes: dict[str, tuple[int, ...]],
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.Dataset:
        """Create a Dataset with zeros for each DataArray field with validation."""
        return cls.create_with_shapes(
            dataset_class, shapes, "zeros", validate_model_consistency=validate_model_consistency, **coords_and_attrs
        )

    @classmethod
    def ones_like_fields(
        cls,
        dataset_class: type,
        shapes: dict[str, tuple[int, ...]],
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.Dataset:
        """Create a Dataset with ones for each DataArray field with validation."""
        return cls.create_with_shapes(
            dataset_class, shapes, "ones", validate_model_consistency=validate_model_consistency, **coords_and_attrs
        )


class DataTreeFactory:
    """Factory for creating DataTree instances with comprehensive validation."""

    @classmethod
    def create_from_datasets(
        cls,
        datatree_class: type,
        datasets: dict[str, Any],
        validate_model_consistency: bool = True,
        **coords_and_attrs: Any,
    ) -> xr.DataTree:
        """
        Create a DataTree instance from a dictionary of datasets.

        Parameters
        ----------
        datatree_class : type
            DataTree class to create
        datasets : dict[str, Any]
            Dictionary mapping field names to Dataset instances
        validate_model_consistency : bool, default True
            Whether to validate model consistency after creation
        **coords_and_attrs
            Root-level coordinates and attributes

        Returns
        -------
        xr.DataTree
            Created and validated DataTree
        """
        # Validate that provided datasets match expected fields
        dataset_fields = datatree_class.get_model_dataset_fields()
        datatree_fields = datatree_class.get_model_datatree_fields()
        expected_fields = set(dataset_fields.keys()) | set(datatree_fields.keys())

        provided_fields = set(datasets.keys())
        missing_fields = expected_fields - provided_fields
        extra_fields = provided_fields - expected_fields

        if missing_fields:
            raise ValueError(f"Missing required fields for {datatree_class.__name__}: {missing_fields}")
        if extra_fields:
            raise ValueError(f"Unexpected fields for {datatree_class.__name__}: {extra_fields}")

        # Create the instance
        instance = datatree_class(**datasets, **coords_and_attrs)

        # Optionally validate model consistency
        if validate_model_consistency:
            try:
                ModelValidator.validate_model_consistency(instance)
            except Exception as e:
                raise ValueError(f"Created {datatree_class.__name__} failed consistency validation: {e}") from e

        return instance.to_xarray()

    @classmethod
    def new(cls, datatree_class: type, **field_values: Any) -> xr.DataTree:
        """Create a new DataTree instance with the given field values."""
        return cls.create_from_datasets(datatree_class, field_values)
