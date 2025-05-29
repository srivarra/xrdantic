"""
Custom Pydantic validation errors for xrdantic.

This module provides specialized error classes for different types of validation
failures that can occur when working with xarray-like data structures.
"""

from __future__ import annotations

import logging

# Exception groups (Python 3.11+)
from builtins import ExceptionGroup
from collections.abc import Callable
from typing import Any

# Configure logger for xrdantic errors
logger = logging.getLogger(__name__)


class XrdanticError(ValueError):
    """
    Base class for all xrdantic validation errors.

    Provides common functionality and consistent error formatting for
    scientific data validation failures.
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "xrdantic_error",
        field_name: str | None = None,
        context: dict[str, Any] | None = None,
        log_error: bool = True,
    ):
        """
        Initialize a custom xrdantic error.

        Parameters
        ----------
        message
            Human-readable error message
        error_type
            Specific error type identifier
        field_name, optional
            Name of the field that caused the error
        context, optional
            Additional context information for debugging
        log_error, default True
            Whether to log the error for debugging
        """
        self.field_name = field_name
        self.context = context or {}
        self.error_type = error_type

        # Log error for debugging if enabled
        if log_error:
            log_msg = f"{error_type}: {message}"
            if field_name:
                log_msg += f" (field: {field_name})"
            if self.context:
                log_msg += f" (context: {self.context})"
            logger.debug(log_msg)

        super().__init__(message)

    def get_detailed_message(self) -> str:
        """Get a detailed error message including context."""
        message = str(self)
        if self.context:
            message += f"\nContext: {self.context}"
        return message

    def to_dict(self) -> dict[str, Any]:
        """Convert error to a dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "field_name": self.field_name,
            "message": str(self),
            "context": self.context,
        }


class DataFieldError(XrdanticError):
    """Raised when there are issues with data field definitions or validation."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        expected_count: int | None = None,
        actual_count: int | None = None,
        field_names: list[str] | None = None,
    ):
        self.expected_count = expected_count
        self.actual_count = actual_count
        self.field_names = field_names
        context = {"expected_count": expected_count, "actual_count": actual_count, "field_names": field_names}
        super().__init__(message, error_type="data_field_error", field_name=field_name, context=context)


class DimensionError(XrdanticError):
    """Raised when there are issues with dimension definitions or validation."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        expected_dims: tuple[str, ...] | None = None,
        actual_dims: tuple[str, ...] | None = None,
        shape: tuple[int, ...] | None = None,
    ):
        context = {"expected_dims": expected_dims, "actual_dims": actual_dims, "shape": shape}
        super().__init__(message, error_type="dimension_error", field_name=field_name, context=context)


class CoordinateError(XrdanticError):
    """Raised when there are issues with coordinate definitions or validation."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        coordinate_name: str | None = None,
        dimensions: tuple[str, ...] | None = None,
        ndim: int | None = None,
    ):
        context = {"coordinate_name": coordinate_name, "dimensions": dimensions, "ndim": ndim}
        super().__init__(message, error_type="coordinate_error", field_name=field_name, context=context)


class ShapeError(XrdanticError):
    """Raised when there are issues with array shape validation."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        actual_shape: tuple[int, ...] | None = None,
        operation_name: str | None = None,
    ):
        context = {"expected_shape": expected_shape, "actual_shape": actual_shape, "operation_name": operation_name}
        super().__init__(message, error_type="shape_error", field_name=field_name, context=context)


class ArrayDataError(XrdanticError):
    """Raised when there are issues with array data content validation."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        data_type: str | None = None,
        invalid_values: str | None = None,
        is_coordinate: bool = False,
    ):
        context = {"data_type": data_type, "invalid_values": invalid_values, "is_coordinate": is_coordinate}
        super().__init__(message, error_type="array_data_error", field_name=field_name, context=context)


class ModelStructureError(XrdanticError):
    """Raised when there are issues with the overall model structure."""

    def __init__(
        self,
        message: str,
        *,
        model_type: str | None = None,
        required_fields: list[str] | None = None,
        missing_fields: list[str] | None = None,
        extra_fields: list[str] | None = None,
    ):
        context = {
            "model_type": model_type,
            "required_fields": required_fields,
            "missing_fields": missing_fields,
            "extra_fields": extra_fields,
        }
        super().__init__(message, error_type="model_structure_error", context=context)


class DataArrayFieldError(XrdanticError):
    """Raised when there are issues with DataArray field definitions."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        dataarray_class: str | None = None,
        missing_annotation: bool = False,
    ):
        context = {"dataarray_class": dataarray_class, "missing_annotation": missing_annotation}
        super().__init__(message, error_type="dataarray_field_error", field_name=field_name, context=context)


class MetadataError(XrdanticError):
    """Raised when there are issues with field metadata."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        metadata_type: str | None = None,
        missing_keys: list[str] | None = None,
    ):
        context = {"metadata_type": metadata_type, "missing_keys": missing_keys}
        super().__init__(message, error_type="metadata_error", field_name=field_name, context=context)


class FactoryError(XrdanticError):
    """Raised when there are issues with factory method operations."""

    def __init__(
        self,
        message: str,
        *,
        factory_method: str | None = None,
        shape: tuple[int, ...] | None = None,
        field_name: str | None = None,
    ):
        context = {"factory_method": factory_method, "shape": shape}
        super().__init__(message, error_type="factory_error", field_name=field_name, context=context)


# Convenience functions for creating common errors


def data_field_count_error(
    model_type: str, expected_count: int, actual_count: int, field_names: list[str]
) -> DataFieldError:
    """Create an error for incorrect number of data fields."""
    if expected_count == 1 and actual_count != 1:
        message = (
            f"{model_type} must have exactly one data field, "
            f"but found {actual_count}: {field_names}. "
            f"Please define exactly one field with Data[...] annotation."
        )
    elif expected_count > 1 and actual_count < expected_count:
        message = (
            f"{model_type} must have at least {expected_count} data field(s), but found {actual_count}: {field_names}."
        )
    else:
        message = (
            f"{model_type} has incorrect number of data fields. "
            f"Expected {expected_count}, got {actual_count}: {field_names}."
        )

    return DataFieldError(
        message, field_name=None, expected_count=expected_count, actual_count=actual_count, field_names=field_names
    )


def missing_dimension_metadata_error(field_name: str) -> MetadataError:
    """Create an error for missing dimension metadata."""
    return MetadataError(
        f"Data field '{field_name}' missing dimension metadata",
        field_name=field_name,
        metadata_type="dimension",
        missing_keys=["dims"],
    )


def invalid_dimensions_format_error(field_name: str, dims: Any) -> DimensionError:
    """Create an error for invalid dimensions format."""
    return DimensionError(
        f"Invalid dimensions format for field '{field_name}': {dims}",
        field_name=field_name,
        actual_dims=tuple(str(d) for d in dims) if isinstance(dims, tuple | list) else None,
    )


def coordinate_dimension_error(field_name: str, dims: tuple[str, ...]) -> CoordinateError:
    """Create an error for coordinate having wrong number of dimensions."""
    return CoordinateError(
        f"Coordinate data field '{field_name}' must have exactly one dimension, got: {dims}",
        field_name=field_name,
        dimensions=dims,
    )


def infinite_values_error(field_name: str) -> ArrayDataError:
    """Create an error for infinite values in data."""
    return ArrayDataError(
        f"Data field '{field_name}' contains infinite values",
        field_name=field_name,
        invalid_values="infinite",
        data_type="floating",
    )


def coordinate_dimensionality_error(field_name: str, ndim: int) -> CoordinateError:
    """Create an error for coordinate having wrong dimensionality."""
    return CoordinateError(
        f"Coordinate data field '{field_name}' must be 1-dimensional, got {ndim}D", field_name=field_name, ndim=ndim
    )


def invalid_shape_error(operation_name: str, shape: tuple[int, ...]) -> ShapeError:
    """Create an error for invalid array shape."""
    return ShapeError(
        f"Invalid shape for {operation_name}: {shape}. All dimensions must be positive.",
        expected_shape=tuple(dim for dim in shape if dim > 0) or None,
        actual_shape=shape,
        operation_name=operation_name,
    )


def shape_dimension_mismatch_error(
    shape: tuple[int, ...], expected_dims: tuple[str, ...], operation_name: str
) -> ShapeError:
    """Create an error for shape not matching expected dimensions."""
    return ShapeError(
        f"Shape {shape} doesn't match expected dimensions {expected_dims} for {operation_name}. "
        f"Expected {len(expected_dims)} dimensions, got {len(shape)}.",
        expected_shape=tuple(len(expected_dims) * [0]) if expected_dims else None,
        actual_shape=shape,
        operation_name=operation_name,
    )


def missing_dataarray_fields_error(model_type: str) -> ModelStructureError:
    """Create an error for models missing required DataArray fields."""
    return ModelStructureError(
        f"{model_type} must have at least one DataArray field. "
        "Please define fields with DataArray subclass annotations.",
        model_type=model_type,
        required_fields=["DataArray fields"],
    )


def missing_dataset_fields_error(model_type: str) -> ModelStructureError:
    """Create an error for models missing required Dataset or DataTree fields."""
    return ModelStructureError(
        f"{model_type} must have at least one Dataset or DataTree field. "
        "Please define fields with Dataset or DataTree subclass annotations.",
        model_type=model_type,
        required_fields=["Dataset or DataTree fields"],
    )


def missing_shape_for_field_error(field_name: str) -> FactoryError:
    """Create an error for missing shape in factory operations."""
    return FactoryError(
        f"Shape not provided for DataArray field '{field_name}'", field_name=field_name, factory_method="create"
    )


def missing_annotation_error(field_name: str) -> DataArrayFieldError:
    """Create an error for missing field annotation."""
    return DataArrayFieldError(
        f"DataArray field '{field_name}' has no annotation", field_name=field_name, missing_annotation=True
    )


def unsupported_factory_method_error(factory_method: str) -> FactoryError:
    """Create an error for unsupported factory methods."""
    return FactoryError(f"Unsupported factory method: {factory_method}", factory_method=factory_method)


def no_dimensions_error(variable_type: str) -> DimensionError:
    """Create an error for missing dimensions in variable definitions."""
    return DimensionError(f"No dimensions provided for {variable_type} variable.", expected_dims=(), actual_dims=None)


class XrdanticValidationGroup(ExceptionGroup[XrdanticError]):
    """
    Exception group specifically for xrdantic validation errors.

    Provides enhanced functionality for grouping and handling multiple
    validation errors that can occur during scientific data validation.
    """

    def __init__(
        self,
        message: str,
        exceptions: list[XrdanticError],
        *,
        group_type: str = "validation_group",
        context: dict[str, Any] | None = None,
    ):
        self.group_type = group_type
        self.context = context or {}
        super().__init__(message, exceptions)

    def filter_by_error_type(self, error_type: str) -> list[XrdanticError]:
        """Filter exceptions by their error type."""
        return [e for e in self.exceptions if isinstance(e, XrdanticError) and e.error_type == error_type]

    def filter_by_field(self, field_name: str) -> list[XrdanticError]:
        """Filter exceptions by field name."""
        return [e for e in self.exceptions if isinstance(e, XrdanticError) and e.field_name == field_name]

    def group_by_error_type(self) -> dict[str, list[XrdanticError]]:
        """Group exceptions by their error type."""
        groups = {}
        for error in self.exceptions:
            if isinstance(error, XrdanticError):
                error_type = error.error_type
                if error_type not in groups:
                    groups[error_type] = []
                groups[error_type].append(error)
        return groups

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all errors in the group."""
        summary = {
            "total_errors": len(self.exceptions),
            "error_types": {},
            "affected_fields": set(),
        }

        for error in self.exceptions:
            if isinstance(error, XrdanticError):
                error_type = error.error_type
                summary["error_types"][error_type] = summary["error_types"].get(error_type, 0) + 1
                if error.field_name:
                    summary["affected_fields"].add(error.field_name)

        summary["affected_fields"] = list(summary["affected_fields"])
        return summary


class ValidationCollector:
    """
    Utility class for collecting validation errors and creating exception groups.

    Useful for comprehensive validation that continues after the first error.
    """

    def __init__(self, continue_on_error: bool = True):
        self.continue_on_error = continue_on_error
        self.errors: list[XrdanticError] = []

    def add_error(self, error: XrdanticError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)
        if not self.continue_on_error:
            raise error

    def validate_and_collect(self, validation_func, *args, **kwargs) -> Any:
        """Run a validation function and collect any errors."""
        try:
            return validation_func(*args, **kwargs)
        except XrdanticError as e:
            self.add_error(e)
            return None

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0

    def raise_if_errors(self, message: str = "Validation failed") -> None:
        """Raise an exception group if any errors were collected."""
        if self.errors:
            if len(self.errors) == 1:
                # Single error - raise it directly
                raise self.errors[0]
            else:
                # Multiple errors - raise as group
                raise XrdanticValidationGroup(message, self.errors)

    def clear(self) -> list[XrdanticError]:
        """Clear collected errors and return them."""
        errors = self.errors.copy()
        self.errors.clear()
        return errors


def collect_field_validation_errors(
    model_instance: Any,
    field_validations: dict[str, Callable[[Any, str], None]],
) -> list[XrdanticError]:
    """
    Run multiple field validations and collect all errors.

    Parameters
    ----------
    model_instance
        The model instance to validate
    field_validations
        Dictionary mapping field names to validation functions

    Returns
    -------
    list[XrdanticError]
        List of all validation errors found
    """
    collector = ValidationCollector(continue_on_error=True)

    for field_name, validation_func in field_validations.items():
        field_value = getattr(model_instance, field_name, None)
        collector.validate_and_collect(validation_func, field_value, field_name)

    return collector.errors


def validate_model_comprehensive(
    model_instance: Any,
    *,
    validate_dimensions: bool = True,
    validate_coordinates: bool = True,
    validate_data: bool = True,
    raise_on_error: bool = True,
) -> list[XrdanticError] | None:
    """
    Perform comprehensive validation of a model, collecting all errors.

    Parameters
    ----------
    model_instance
        The model instance to validate
    validate_dimensions
        Whether to validate dimensions
    validate_coordinates
        Whether to validate coordinates
    validate_data
        Whether to validate data content
    raise_on_error
        Whether to raise an exception group if errors are found

    Returns
    -------
    list[XrdanticError] or None
        List of errors if raise_on_error=False, otherwise None

    Raises
    ------
    XrdanticValidationGroup
        If errors are found and raise_on_error=True
    """
    collector = ValidationCollector(continue_on_error=True)

    if validate_dimensions:
        # Add dimension validation logic
        pass

    if validate_coordinates:
        # Add coordinate validation logic
        pass

    if validate_data:
        # Add data validation logic
        pass

    if raise_on_error:
        collector.raise_if_errors("Comprehensive model validation failed")
        return None
    else:
        return collector.errors
