"""
Configuration management for xrdantic.

This module provides simple configuration for validation behavior.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class XrdanticSettings(BaseSettings):
    """
    Global settings for xrdantic behavior.

    Settings can be overridden by environment variables (e.g., XRDANTIC_STRICT_VALIDATION=true).
    """

    # Core validation settings
    strict_validation: bool = Field(default=False, description="Enable strict validation mode")
    allow_nan_values: bool = Field(default=True, description="Allow NaN values in arrays")
    allow_inf_values: bool = Field(default=False, description="Allow infinite values in arrays")

    # Development settings
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    model_config = SettingsConfigDict(
        env_prefix="XRDANTIC_",
        validate_assignment=True,
        extra="ignore",
    )


# Global settings instance
_settings: XrdanticSettings | None = None


def get_settings() -> XrdanticSettings:
    """Get the global xrdantic settings instance."""
    global _settings
    if _settings is None:
        _settings = XrdanticSettings()
    return _settings


def update_settings(**kwargs: Any) -> None:
    """
    Update global settings.

    Parameters
    ----------
    **kwargs
        Settings to update. Only valid XrdanticSettings fields will be considered.

    Raises
    ------
    ValueError
        If an invalid setting key is provided.
    """
    settings = get_settings()
    valid_keys = XrdanticSettings.model_fields.keys()

    for key, value in kwargs.items():
        if key in valid_keys:
            setattr(settings, key, value)
        else:
            raise ValueError(f"Invalid setting: '{key}'. Valid settings are: {', '.join(valid_keys)}")


def reset_settings() -> None:
    """Reset settings to default values."""
    global _settings
    _settings = None


class ValidationContext:
    """Context manager for temporary validation settings."""

    def __init__(self, **temporary_settings: Any):
        self.temporary_settings = temporary_settings
        self.original_settings_values: dict[str, Any] = {}

        valid_keys = XrdanticSettings.model_fields.keys()
        for key in self.temporary_settings:
            if key not in valid_keys:
                raise ValueError(f"Invalid temporary setting: '{key}'. Valid settings are: {', '.join(valid_keys)}")

    def __enter__(self) -> ValidationContext:
        settings = get_settings()
        for key, temp_value in self.temporary_settings.items():
            self.original_settings_values[key] = getattr(settings, key)
            setattr(settings, key, temp_value)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        settings = get_settings()
        for key, original_value in self.original_settings_values.items():
            setattr(settings, key, original_value)
