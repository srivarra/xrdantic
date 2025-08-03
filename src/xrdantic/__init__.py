from importlib.metadata import version

# Import only essential components for public API
from .config import (
    ValidationContext,
    XrdanticSettings,
    get_settings,
    reset_settings,
    update_settings,
)
from .models import Coordinate, DataArray, Dataset, DataTree
from .types import Dim
from .utils import Attr, Data, Name

__all__ = [
    # Core model classes for building xarray data structures
    "Coordinate",
    "DataArray",
    "Dataset",
    "DataTree",
    # Type annotation helpers
    "Attr",
    "Data",
    "Dim",
    "Name",
    # Configuration and settings
    "ValidationContext",
    "get_settings",
    "reset_settings",
    "update_settings",
    "XrdanticSettings",
]

__version__ = version("xrdantic")
