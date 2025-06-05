from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

from numpydantic import Shape
from pydantic import Field
from pydantic.fields import FieldInfo

from xrdantic.types import Dim, TDims, TDType, TNDArray

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class XrAnnotation(Enum):
    """Enum for the different types of annotations."""

    DATA = "data"
    COORD = "coord"
    ATTR = "attr"
    NAME = "name"


def _dims_to_shape_string(dims: tuple[str, ...]) -> str:
    """Converts a tuple of dimension names to the shape string for NDArray."""
    return ", ".join(f"* {dim}" for dim in dims)


def xr_meta(
    *,
    kind: XrAnnotation,
    dims: Dim | tuple[Dim, ...] | None = None,
    dtype: Any = None,
) -> dict[str, Any]:
    """
    Processes dimensions and creates metadata dictionary with validation.

    Parameters
    ----------
    kind : XrAnnotation
        The kind of annotation (DATA, COORD, ATTR, NAME)
    dims : Dim or tuple[Dim, ...], optional
        Dimension specification
    dtype : Any, optional
        Data type specification

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with validated dimensions

    Raises
    ------
    TypeError
        If dims argument is invalid
    """
    from xrdantic.validation import DataValidator

    metadata: dict[str, Any] = {"kind": kind}
    processed_dims: tuple[str, ...] | None = None

    if isinstance(dims, Dim):
        processed_dims = (str(dims),)
    elif isinstance(dims, tuple):
        _dims_list = []
        for dim in dims:
            if isinstance(dim, Dim):
                _dims_list.append(str(dim))
            else:
                raise TypeError(f"Invalid item in dims tuple: {dim!r}. Expected only Dim instances.")
        if not _dims_list:
            raise TypeError("Empty tuple provided for dims argument.")
        processed_dims = tuple(_dims_list)
    elif dims is not None:
        raise TypeError(
            f"Invalid dims argument type: {dims!r} ({type(dims)}). Expected a Dim instance or a tuple of Dim instances."
        )

    if processed_dims is not None:
        metadata["dims"] = processed_dims

        # Validate coordinate dimensions if this is a coordinate field
        if kind == XrAnnotation.COORD:
            try:
                DataValidator.validate_coordinate_dimensions(processed_dims, "coordinate_field")
            except Exception as e:
                raise TypeError(f"Invalid coordinate dimensions: {e}") from e

    if dtype is not None:
        metadata["dtype"] = dtype
    return metadata


def get_data_fields(model_fields: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """Get all data fields from model fields with validation."""
    data_fields = {}
    for k, v in model_fields.items():
        if vjson := v.json_schema_extra:
            if vjson.get("kind") == XrAnnotation.DATA:
                data_fields[k] = v
    return data_fields


def get_coord_fields(model_fields: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """Get all coordinate fields from model fields with validation."""
    coord_fields = {}
    for k, v in model_fields.items():
        if vjson := v.json_schema_extra:
            if vjson.get("kind") == XrAnnotation.COORD:
                coord_fields[k] = v
    return coord_fields


def get_attr_fields(model_fields: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """Get all attribute fields from model fields."""
    attr_fields = {}
    for k, v in model_fields.items():
        if vjson := v.json_schema_extra:
            if vjson.get("kind") == XrAnnotation.ATTR:
                attr_fields[k] = v
    return attr_fields


def get_name_field(model_fields: dict[str, FieldInfo]) -> str | None:
    """Get the name field value from model fields."""
    for k, v in model_fields.items():
        if vjson := v.json_schema_extra:
            if vjson.get("kind") == XrAnnotation.NAME:
                return k
    return None


def get_coordinate_model_fields(model_fields: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """Get all fields that are Coordinate model instances."""
    from xrdantic.models import Coordinate  # Import here to avoid circular imports

    coordinate_fields = {}
    for k, v in model_fields.items():
        # Check if the annotation is a class and is a subclass of Coordinate
        annotation = v.annotation
        if isinstance(annotation, type) and issubclass(annotation, Coordinate):
            coordinate_fields[k] = v
    return coordinate_fields


def get_dataarray_model_fields(model_fields: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """Get all fields that are DataArray model instances."""
    from xrdantic.models import DataArray  # Import here to avoid circular imports

    dataarray_fields = {}
    for k, v in model_fields.items():
        # Check if the annotation is a class and is a subclass of DataArray
        annotation = v.annotation
        if isinstance(annotation, type) and issubclass(annotation, DataArray):
            dataarray_fields[k] = v
    return dataarray_fields


def get_dataset_model_fields(model_fields: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """Get all fields that are Dataset model instances."""
    from xrdantic.models import Dataset  # Import here to avoid circular imports

    dataset_fields = {}
    for k, v in model_fields.items():
        # Check if the annotation is a class and is a subclass of Dataset
        annotation = v.annotation
        if isinstance(annotation, type) and issubclass(annotation, Dataset):
            dataset_fields[k] = v
    return dataset_fields


def get_datatree_model_fields(model_fields: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """Get all fields that are DataTree model instances."""
    from xrdantic.models import DataTree  # Import here to avoid circular imports

    datatree_fields = {}
    for k, v in model_fields.items():
        # Check if the annotation is a class and is a subclass of DataTree
        annotation = v.annotation
        if isinstance(annotation, type) and issubclass(annotation, DataTree):
            datatree_fields[k] = v
    return datatree_fields


def validate_field_metadata(field_name: str, field_info: FieldInfo, expected_kind: XrAnnotation) -> bool:
    """
    Validate field metadata.

    Parameters
    ----------
    field_name : str
        Name of the field
    field_info : FieldInfo
        Pydantic field information
    expected_kind : XrAnnotation
        Expected kind of annotation

    Returns
    -------
    bool
        True if validation passes

    Raises
    ------
    ValueError
        If field metadata is invalid or missing
    """
    from xrdantic.validation import DataValidator

    if field_info.json_schema_extra is None:
        raise ValueError(f"Field '{field_name}' is missing required metadata")

    if callable(field_info.json_schema_extra):
        raise ValueError(f"Field '{field_name}' has callable json_schema_extra, expected dict")

    # Simple validation without complex type checking
    try:
        if expected_kind == XrAnnotation.DATA:
            DataValidator.validate_dimension_metadata(field_info, field_name)
        elif expected_kind == XrAnnotation.COORD:
            DataValidator.validate_dimension_metadata(field_info, field_name)
            dims = DataValidator.validate_dimension_metadata(field_info, field_name)
            DataValidator.validate_coordinate_dimensions(dims, field_name)
    except Exception as e:
        raise ValueError(f"Field '{field_name}' validation failed: {e}") from e

    return True


class Data:
    """
    Factory for Data Variable field annotations with enhanced validation.

    Usage: Data[Dims, DType]
    Example: data: Data[tuple[X, Y], np.float64]
    """

    @classmethod
    def __class_getitem__(cls, params: tuple[TDims, TDType]) -> Annotated[TNDArray, Field]:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(f"{cls.__name__}[...] requires two arguments: Dims and DType.")

        dims_arg, dtype_arg = params
        metadata = xr_meta(kind=XrAnnotation.DATA, dims=dims_arg, dtype=dtype_arg)
        processed_dims = metadata.get("dims")

        if processed_dims:
            shape_string = _dims_to_shape_string(processed_dims)

            return Annotated[TNDArray[Shape[shape_string], dtype_arg], Field(json_schema_extra=metadata)]  # type: ignore
        else:
            from xrdantic.errors import no_dimensions_error

            raise no_dimensions_error("Data")


class Attr:
    """
    Factory for Attribute field annotations.

    Usage: Attr[Type]
    Example: units: Attr[str] = "meters"
    """

    @classmethod
    def __class_getitem__(cls, attr_type: type[T]) -> Any:
        return Annotated[attr_type | Literal[None], Field(json_schema_extra=xr_meta(kind=XrAnnotation.ATTR))]


Name = Annotated[str | Literal[None], Field(json_schema_extra=xr_meta(kind=XrAnnotation.NAME))]
"""Factory for Name field annotations. The default name field type is str."""
