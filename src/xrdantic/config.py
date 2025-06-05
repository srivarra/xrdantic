"""
Configuration management for xrdantic.

This module provides centralized configuration for managing application-wide
settings and validation behavior.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class XrdanticSettings(BaseSettings):
    """
    Global settings for xrdantic behavior.

    Centralized configuration for validation and performance settings.
    Settings can be overridden by environment variables (e.g., XRDANTIC_STRICT_VALIDATION=true).
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

    model_config = SettingsConfigDict(
        env_prefix="XRDANTIC_",
        validate_assignment=True,
        extra="ignore",
        frozen=False,
    )


@lru_cache
def get_settings() -> XrdanticSettings:
    """
    Get the global xrdantic settings instance.

    This function is cached to ensure a single settings instance
    is used throughout the application.

    Returns
    -------
    The global settings instance
    """
    return XrdanticSettings()


def update_settings(**kwargs: Any) -> None:
    """
    Update global settings.

    Parameters
    ----------
    **kwargs
        Settings to update. Only valid XrdanticSettings fields will be considered.

    Raises
    ------
    ValidationError
        If the updated settings are invalid.
    ValueError
        If an invalid setting key is provided.
    """
    current_settings_obj = get_settings()
    new_settings_data = current_settings_obj.model_dump()

    valid_keys = XrdanticSettings.model_fields.keys()
    for key, value in kwargs.items():
        if key in valid_keys:
            new_settings_data[key] = value
        else:
            raise ValueError(f"Invalid setting: '{key}'. Valid settings are: {', '.join(valid_keys)}")

    try:
        validated_settings_data = XrdanticSettings(**new_settings_data).model_dump()
    except ValidationError:
        raise

    get_settings.cache_clear()

    global_settings_instance_to_update = get_settings()

    # Apply the validated new values to this global instance.
    for key, value in validated_settings_data.items():
        setattr(global_settings_instance_to_update, key, value)


class ValidationContext:
    """
    Context manager for temporary validation settings.

    Allows temporary override of validation settings for specific operations.
    """

    def __init__(self, **temporary_settings: Any):
        self.temporary_settings = temporary_settings
        self.original_settings_values: dict[str, Any] = {}

        valid_keys = XrdanticSettings.model_fields.keys()
        for key in self.temporary_settings:
            if key not in valid_keys:
                raise ValueError(f"Invalid temporary setting: '{key}'. Valid settings are: {', '.join(valid_keys)}")

    def __enter__(self) -> ValidationContext:
        current_settings_obj = get_settings()
        for key, temp_value in self.temporary_settings.items():
            self.original_settings_values[key] = getattr(current_settings_obj, key)
            setattr(current_settings_obj, key, temp_value)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        current_settings_obj = get_settings()
        for key, original_value in self.original_settings_values.items():
            setattr(current_settings_obj, key, original_value)


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


def reset_to_default_settings() -> None:
    """Resets all xrdantic settings to their default values or environment-defined values."""
    get_settings.cache_clear()
    get_settings()
