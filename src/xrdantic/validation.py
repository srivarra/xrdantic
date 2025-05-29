"""
Validation utilities for xrdantic models.

This module contains common validation logic used across different model types
to ensure data integrity and type safety.
"""

from __future__ import annotations

from typing import Any

import natsort as ns
import numpy as np
from pydantic import ValidationError, ValidationInfo
from pydantic.fields import FieldInfo

from xrdantic import utils
from xrdantic.errors import (
    CoordinateError,
    DimensionError,
    ValidationCollector,
    coordinate_dimension_error,
    coordinate_dimensionality_error,
    data_field_count_error,
    infinite_values_error,
    invalid_dimensions_format_error,
    invalid_shape_error,
    missing_dataarray_fields_error,
    missing_dimension_metadata_error,
    shape_dimension_mismatch_error,
)


class DataValidator:
    """Common validation logic for data fields."""

    @staticmethod
    def validate_data_field_count(data_fields: dict[str, FieldInfo], expected_count: int, model_type: str) -> None:
        """Validate that the model has the expected number of data fields."""
        field_count = len(data_fields)
        field_names = list(data_fields.keys())

        if (expected_count == 1 and field_count != 1) or (expected_count > 1 and field_count < expected_count):
            raise data_field_count_error(model_type, expected_count, field_count, field_names)

    @staticmethod
    def validate_dimension_metadata(field_info: FieldInfo, field_name: str) -> tuple[str, ...]:
        """Validate and extract dimension metadata from a field."""
        if field_info.json_schema_extra is None:
            raise missing_dimension_metadata_error(field_name)

        dims = field_info.json_schema_extra.get("dims", ())
        if not isinstance(dims, tuple | list):
            raise invalid_dimensions_format_error(field_name, dims)

        return tuple(str(dim) for dim in dims)

    @staticmethod
    def validate_coordinate_dimensions(dims: tuple[str, ...], field_name: str) -> None:
        """Validate that coordinate has exactly one dimension."""
        if len(dims) != 1:
            raise coordinate_dimension_error(field_name, dims)

    @staticmethod
    def sanitize_array_data(value: Any, field_name: str, is_coordinate: bool = False) -> Any:
        """Sanitize and validate array data with enhanced validation."""
        if isinstance(value, list | tuple):
            value = np.array(value)

        # Enhanced validation for different data types
        if hasattr(value, "dtype"):
            # Check for floating point issues
            if np.issubdtype(value.dtype, np.floating):
                if np.any(np.isinf(value)):
                    raise infinite_values_error(field_name)
                # Optional: warn about NaN values
                if np.any(np.isnan(value)):
                    import warnings

                    warnings.warn(f"Field '{field_name}' contains NaN values", UserWarning, stacklevel=2)

            # Check for complex numbers with invalid parts
            if np.issubdtype(value.dtype, np.complexfloating):
                if np.any(np.isinf(value.real)) or np.any(np.isinf(value.imag)):
                    raise infinite_values_error(field_name)

        # Validate 1D for coordinates
        if is_coordinate and hasattr(value, "ndim") and value.ndim != 1:
            raise coordinate_dimensionality_error(field_name, value.ndim)

        return value

    @staticmethod
    def validate_shape(shape: tuple[int, ...], operation_name: str) -> None:
        """Validate shape for array creation operations."""
        if not shape or any(dim <= 0 for dim in shape):
            raise invalid_shape_error(operation_name, shape)

    @staticmethod
    def validate_shape_matches_dimensions(
        shape: tuple[int, ...], expected_dims: tuple[str, ...], operation_name: str
    ) -> None:
        """Validate that shape matches expected dimensions."""
        if len(shape) != len(expected_dims):
            raise shape_dimension_mismatch_error(shape, expected_dims, operation_name)

    @staticmethod
    def validate_with_context(value: Any, info: ValidationInfo) -> Any:
        """Enhanced validation with context information."""
        if info.context:
            # Use context for more sophisticated validation
            strict_mode = info.context.get("strict_validation", False)
            if strict_mode:
                # Perform additional validations in strict mode
                pass
        return value


class ModelValidator:
    """Validation logic for model structure."""

    @staticmethod
    def validate_dataarray_fields(dataarray_fields: dict[str, FieldInfo], model_type: str) -> None:
        """Validate DataArray fields for Dataset models."""
        if len(dataarray_fields) == 0:
            raise missing_dataarray_fields_error(model_type)

    @staticmethod
    def _process_data_field_for_consistency(
        model_instance: Any,
        data_field_name: str,
        data_field_info: FieldInfo,
        collector: ValidationCollector,
        data_dimensions: set[str],
        data_dimension_sizes: dict[str, int],
    ) -> None:
        try:
            dims = DataValidator.validate_dimension_metadata(data_field_info, data_field_name)
            data_dimensions.update(dims)

            data_value = getattr(model_instance, data_field_name)
            if hasattr(data_value, "shape"):
                if len(data_value.shape) == len(dims):
                    for i, dim_name in enumerate(dims):
                        size = data_value.shape[i]
                        if dim_name in data_dimension_sizes:
                            if data_dimension_sizes[dim_name] != size:
                                error = DimensionError(
                                    f"Dimension '{dim_name}' has inconsistent sizes across data fields: "
                                    f"expected {data_dimension_sizes[dim_name]}, got {size} in field '{data_field_name}'",
                                    field_name=data_field_name,
                                    expected_dims=(dim_name,),
                                    shape=data_value.shape,
                                )
                                collector.add_error(error)
                        else:
                            data_dimension_sizes[dim_name] = size
                else:
                    error = DimensionError(
                        f"Data field '{data_field_name}' has shape {data_value.shape} but {len(dims)} dimensions: {dims}",
                        field_name=data_field_name,
                        expected_dims=dims,
                        shape=data_value.shape,
                    )
                    collector.add_error(error)
        except (ValidationError, ValueError, TypeError, KeyError) as e:
            if not isinstance(e, DimensionError | CoordinateError):
                error = DimensionError(
                    f"Failed to validate dimensions for data field '{data_field_name}': {e}",
                    field_name=data_field_name,
                )
                collector.add_error(error)
            else:
                collector.add_error(e)

    @staticmethod
    def _process_coord_field_for_consistency(
        model_instance: Any,
        coord_field_name: str,
        coord_field_info: FieldInfo,
        collector: ValidationCollector,
        available_coord_dims: set[str],
        coord_dimension_sizes: dict[str, int],
    ) -> None:
        try:
            dims = DataValidator.validate_dimension_metadata(coord_field_info, coord_field_name)
            if len(dims) == 1:
                dim_name = dims[0]
                available_coord_dims.add(dim_name)
                coord_value = getattr(model_instance, coord_field_name)
                if hasattr(coord_value, "shape"):
                    if len(coord_value.shape) == 1:
                        coord_dimension_sizes[dim_name] = coord_value.shape[0]
                    else:
                        error = CoordinateError(
                            f"Coordinate field '{coord_field_name}' must be 1-dimensional, got shape {coord_value.shape}",
                            field_name=coord_field_name,
                            coordinate_name=dim_name,
                            ndim=len(coord_value.shape),
                        )
                        collector.add_error(error)
            else:
                error = CoordinateError(
                    f"Coordinate field '{coord_field_name}' must have exactly one dimension, got: {dims}",
                    field_name=coord_field_name,
                    dimensions=dims,
                )
                collector.add_error(error)
        except (ValidationError, ValueError, TypeError, KeyError) as e:
            if not isinstance(e, DimensionError | CoordinateError):
                error = CoordinateError(
                    f"Failed to validate coordinate field '{coord_field_name}': {e}",
                    field_name=coord_field_name,
                )
                collector.add_error(error)
            else:
                collector.add_error(e)

    @staticmethod
    def _process_coord_model_data_field_for_consistency(
        coord_model_instance: Any,
        coord_model_name_context: str,
        coord_data_field_name: str,
        coord_data_field_info: FieldInfo,
        collector: ValidationCollector,
        available_coord_dims: set[str],
        coord_dimension_sizes: dict[str, int],
    ) -> None:
        full_field_name = f"{coord_model_name_context}.{coord_data_field_name}"
        try:
            dims = DataValidator.validate_dimension_metadata(coord_data_field_info, full_field_name)
            if len(dims) == 1:
                dim_name = dims[0]
                available_coord_dims.add(dim_name)
                coord_value = getattr(coord_model_instance, coord_data_field_name)
                if hasattr(coord_value, "shape"):
                    if len(coord_value.shape) == 1:
                        coord_dimension_sizes[dim_name] = coord_value.shape[0]
                    else:
                        error = CoordinateError(
                            f"Coordinate model '{full_field_name}' must be 1-dimensional, got shape {coord_value.shape}",
                            field_name=full_field_name,
                            coordinate_name=dim_name,
                            ndim=len(coord_value.shape),
                        )
                        collector.add_error(error)
            else:
                error = CoordinateError(
                    f"Coordinate model '{full_field_name}' must have exactly one dimension, got: {dims}",
                    field_name=full_field_name,
                    dimensions=dims,
                )
                collector.add_error(error)
        except (ValidationError, ValueError, TypeError, KeyError) as e:
            if not isinstance(e, DimensionError | CoordinateError):
                error = CoordinateError(
                    f"Failed to validate coordinate model field '{full_field_name}': {e}",
                    field_name=full_field_name,
                )
                collector.add_error(error)
            else:
                collector.add_error(e)

    @staticmethod
    def _process_coord_model_group_for_consistency(
        model_instance: Any,
        coord_model_name: str,
        collector: ValidationCollector,
        available_coord_dims: set[str],
        coord_dimension_sizes: dict[str, int],
    ) -> None:
        try:
            coord_model_instance = getattr(model_instance, coord_model_name)
            coord_data_fields = coord_model_instance.get_data_fields()
        except (AttributeError, ValidationError, ValueError, TypeError, KeyError) as e:
            if not isinstance(e, DimensionError | CoordinateError):
                error = CoordinateError(
                    f"Failed to access or get fields from coordinate model '{coord_model_name}': {e}",
                    field_name=coord_model_name,
                )
                collector.add_error(error)
            else:  # Should be rare for getattr/get_data_fields to raise XrdanticError directly
                collector.add_error(e)
            return

        for coord_data_field_name, coord_data_field_info in coord_data_fields.items():
            ModelValidator._process_coord_model_data_field_for_consistency(
                coord_model_instance,
                coord_model_name,
                coord_data_field_name,
                coord_data_field_info,
                collector,
                available_coord_dims,
                coord_dimension_sizes,
            )

    @staticmethod
    def validate_model_consistency(model_instance: Any) -> None:
        """
        Validate cross-field consistency for the entire model.

        This performs comprehensive validation of dimensional consistency across
        all fields in the model, ensuring that:
        - Coordinate dimensions match those used in data fields
        - Coordinate array sizes match the corresponding dimension sizes in data
        - All referenced dimensions have corresponding coordinates
        - Consistent dimension sizes across multiple data fields

        Parameters
        ----------
        model_instance : Any
            The model instance to validate

        Raises
        ------
        XrdanticValidationGroup
            If multiple validation errors are found
        DimensionError
            If dimensions are inconsistent across fields
        CoordinateError
            If coordinate definitions don't match data dimensions
        """
        collector = ValidationCollector(continue_on_error=True)

        # Get all field types by calling utils directly
        model_fields = model_instance.model_fields
        data_fields = utils.get_data_fields(model_fields)
        coord_fields = utils.get_coord_fields(model_fields)
        coord_model_fields = utils.get_coordinate_model_fields(model_fields)

        # Collect all dimensions used in data fields
        data_dimensions: set[str] = set()
        data_dimension_sizes: dict[str, int] = {}

        # Validate data fields and collect dimension information
        for data_field_name, data_field_info in data_fields.items():
            ModelValidator._process_data_field_for_consistency(
                model_instance,
                data_field_name,
                data_field_info,
                collector,
                data_dimensions,
                data_dimension_sizes,
            )

        # Collect all available coordinate dimensions
        available_coord_dims: set[str] = set()
        coord_dimension_sizes: dict[str, int] = {}

        # Validate direct coordinate fields
        for coord_field_name, coord_field_info in coord_fields.items():
            ModelValidator._process_coord_field_for_consistency(
                model_instance,
                coord_field_name,
                coord_field_info,
                collector,
                available_coord_dims,
                coord_dimension_sizes,
            )

        # Validate coordinate model fields
        for coord_model_name in coord_model_fields.keys():
            ModelValidator._process_coord_model_group_for_consistency(
                model_instance,
                coord_model_name,
                collector,
                available_coord_dims,
                coord_dimension_sizes,
            )

        # Validate that all data dimensions have corresponding coordinates
        missing_coords = data_dimensions - available_coord_dims
        if missing_coords:
            error = DimensionError(
                f"Data fields reference dimensions {sorted(missing_coords)} but no corresponding coordinates are defined. "
                f"Available coordinate dimensions: {sorted(available_coord_dims)}",
                expected_dims=tuple(sorted(data_dimensions)),
                actual_dims=tuple(sorted(available_coord_dims)),
            )
            collector.add_error(error)

        # Validate that coordinate sizes match data dimension sizes
        for dim_name in data_dimensions & available_coord_dims:
            if dim_name in data_dimension_sizes and dim_name in coord_dimension_sizes:
                data_size = data_dimension_sizes[dim_name]
                coord_size = coord_dimension_sizes[dim_name]
                if data_size != coord_size:
                    error = CoordinateError(
                        f"Coordinate '{dim_name}' has size {coord_size} but data fields expect size {data_size}. "
                        f"Coordinate array length must match the corresponding dimension size in data arrays.",
                        coordinate_name=dim_name,
                        dimensions=(dim_name,),
                    )
                    collector.add_error(error)

        # Check for unused coordinates (warning, not error)
        unused_coords = available_coord_dims - data_dimensions
        if unused_coords:
            import warnings

            warnings.warn(
                f"Coordinate dimensions {ns.natsorted(unused_coords)} are defined but not used by any data fields. "
                f"This may indicate a modeling issue or could be intentional for future extensions.",
                UserWarning,
                stacklevel=3,
            )

        # Raise all collected errors as a group (or single error if only one)
        collector.raise_if_errors("Model consistency validation failed")
