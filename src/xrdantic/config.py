"""
Configuration management for xrdantic.

This module provides centralized configuration for managing application-wide
settings and validation behavior.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class XrdanticSettings(BaseModel):
    """
    Global settings for xrdantic behavior.

    Centralized configuration for validation and performance settings.
    """

    # Validation settings
    strict_validation: bool = Field(default=False, description="Enable strict validation mode with enhanced checks")

    validate_coordinates: bool = Field(default=True, description="Enable coordinate validation")

    validate_dimensions: bool = Field(default=True, description="Enable dimension consistency validation")

    # Performance settings
    use_validation_cache: bool = Field(default=True, description="Enable caching of validation results")

    max_cache_size: int = Field(default=128, ge=0, description="Maximum size of validation cache")

    # Array handling settings
    allow_nan_values: bool = Field(default=True, description="Allow NaN values in floating point arrays")

    allow_inf_values: bool = Field(default=False, description="Allow infinite values in arrays")

    auto_convert_lists: bool = Field(default=True, description="Automatically convert lists to numpy arrays")

    # Error handling settings
    log_validation_errors: bool = Field(default=True, description="Log validation errors for debugging")

    detailed_error_messages: bool = Field(default=True, description="Include detailed context in error messages")

    # Memory management
    enable_memory_optimization: bool = Field(default=False, description="Enable memory optimization features")

    # Development settings
    debug_mode: bool = Field(default=False, description="Enable debug mode with additional logging")

    model_config = {
        "validate_assignment": True,
        "extra": "ignore",
        "frozen": False,
    }


@lru_cache
def get_settings() -> XrdanticSettings:
    """
    Get the global xrdantic settings instance.

    This function is cached to ensure a single settings instance
    is used throughout the application.

    Returns
    -------
    XrdanticSettings
        The global settings instance
    """
    return XrdanticSettings()


def update_settings(**kwargs: Any) -> None:
    """
    Update global settings.

    Parameters
    ----------
    **kwargs
        Settings to update

    Raises
    ------
    ValidationError
        If the updated settings are invalid.
    """
    current_settings = get_settings()
    new_settings_data = current_settings.model_dump()

    # Update with new values, only considering valid fields
    for key, value in kwargs.items():
        if key in XrdanticSettings.model_fields:
            new_settings_data[key] = value
        else:
            raise ValueError(f"Invalid setting: {key}")

    try:
        XrdanticSettings(**new_settings_data)
    except ValidationError:
        raise

    get_settings.cache_clear()

    settings_to_update = get_settings()

    # Apply the validated new settings
    for key, value in new_settings_data.items():
        if hasattr(settings_to_update, key):  # Redundant check, but safe
            setattr(settings_to_update, key, value)


class ValidationContext:
    """
    Context manager for temporary validation settings.

    Allows temporary override of validation settings for specific operations.
    """

    def __init__(self, **temporary_settings: Any):
        self.temporary_settings = temporary_settings
        self.original_settings: dict[str, Any] = {}

    def __enter__(self) -> ValidationContext:
        settings = get_settings()
        # Store original settings
        for key in self.temporary_settings:
            if hasattr(settings, key):
                self.original_settings[key] = getattr(settings, key)
                setattr(settings, key, self.temporary_settings[key])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        settings = get_settings()
        # Restore original settings
        for key, value in self.original_settings.items():
            setattr(settings, key, value)


# Convenience functions for common configuration patterns
def enable_strict_mode() -> None:
    """Enable strict validation mode."""
    update_settings(strict_validation=True, detailed_error_messages=True)


def enable_performance_mode() -> None:
    """Enable performance optimization mode."""
    update_settings(
        use_validation_cache=True,
        enable_memory_optimization=True,
        log_validation_errors=False,
    )


def enable_debug_mode() -> None:
    """Enable debug mode with extensive logging."""
    update_settings(
        debug_mode=True,
        log_validation_errors=True,
        detailed_error_messages=True,
    )
