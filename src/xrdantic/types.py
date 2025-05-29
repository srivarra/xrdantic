from __future__ import annotations

from collections.abc import Hashable
from typing import TypeAlias, TypeVar

from numpydantic import NDArray


class Dim:
    """A class for representing a dimension of an Xarray data structure."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """The name of the dimension."""
        return self._name

    def __repr__(self):
        return f"Dim('{self.name}')"

    def __str__(self) -> str:
        return self.name


TDims = TypeVar("TDims", bound=tuple[Dim, ...], covariant=True)
TDType = TypeVar("TDType", covariant=True)
THashable = TypeVar("THashable", bound=Hashable)

TNDArray: TypeAlias = NDArray
