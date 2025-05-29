"""Tests for configuration management."""

import pytest

from xrdantic.config import (
    ValidationContext,
    XrdanticSettings,
    enable_debug_mode,
    enable_performance_mode,
    enable_strict_mode,
    get_settings,
    update_settings,
)


class TestXrdanticSettings:
    """Test XrdanticSettings model."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = XrdanticSettings()
        assert settings.strict_validation is False
        assert settings.validate_coordinates is True
        assert settings.validate_dimensions is True
        assert settings.use_validation_cache is True
        assert settings.max_cache_size == 128
        assert settings.allow_nan_values is True
        assert settings.allow_inf_values is False
        assert settings.auto_convert_lists is True
        assert settings.log_validation_errors is True
        assert settings.detailed_error_messages is True
        assert settings.enable_memory_optimization is False
        assert settings.debug_mode is False

    def test_settings_validation(self):
        """Test settings validation."""
        # Test valid max_cache_size
        settings = XrdanticSettings(max_cache_size=256)
        assert settings.max_cache_size == 256

        # Test invalid max_cache_size
        with pytest.raises(ValueError):
            XrdanticSettings(max_cache_size=-1)

    def test_settings_assignment_validation(self):
        """Test validate_assignment config."""
        settings = XrdanticSettings()

        # Should validate on assignment
        settings.max_cache_size = 512
        assert settings.max_cache_size == 512

        with pytest.raises(ValueError):
            settings.max_cache_size = -5

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        # Should not raise error due to extra="ignore"
        settings = XrdanticSettings(unknown_field=True)
        assert not hasattr(settings, "unknown_field")


class TestSettingsFunctions:
    """Test settings management functions."""

    def test_get_settings(self):
        """Test get_settings function."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should return the same cached instance
        assert settings1 is settings2
        assert isinstance(settings1, XrdanticSettings)

    def test_update_settings(self):
        """Test update_settings function."""
        # Store original values
        original_settings = get_settings()
        original_strict = original_settings.strict_validation
        original_debug = original_settings.debug_mode

        try:
            # Update settings
            update_settings(strict_validation=True, debug_mode=True)

            updated_settings = get_settings()
            assert updated_settings.strict_validation is True
            assert updated_settings.debug_mode is True

            # Test updating non-existent attribute
            update_settings(non_existent_field=True)
            # Should not raise error, just ignore

        finally:
            # Restore original values
            update_settings(strict_validation=original_strict, debug_mode=original_debug)

    def test_enable_strict_mode(self):
        """Test enable_strict_mode convenience function."""
        # Store original values
        original_settings = get_settings()
        original_strict = original_settings.strict_validation
        original_detailed = original_settings.detailed_error_messages

        try:
            enable_strict_mode()

            settings = get_settings()
            assert settings.strict_validation is True
            assert settings.detailed_error_messages is True

        finally:
            # Restore original values
            update_settings(strict_validation=original_strict, detailed_error_messages=original_detailed)

    def test_enable_performance_mode(self):
        """Test enable_performance_mode convenience function."""
        # Store original values
        original_settings = get_settings()
        original_cache = original_settings.use_validation_cache
        original_memory = original_settings.enable_memory_optimization
        original_log = original_settings.log_validation_errors

        try:
            enable_performance_mode()

            settings = get_settings()
            assert settings.use_validation_cache is True
            assert settings.enable_memory_optimization is True
            assert settings.log_validation_errors is False

        finally:
            # Restore original values
            update_settings(
                use_validation_cache=original_cache,
                enable_memory_optimization=original_memory,
                log_validation_errors=original_log,
            )

    def test_enable_debug_mode(self):
        """Test enable_debug_mode convenience function."""
        # Store original values
        original_settings = get_settings()
        original_debug = original_settings.debug_mode
        original_log = original_settings.log_validation_errors
        original_detailed = original_settings.detailed_error_messages

        try:
            enable_debug_mode()

            settings = get_settings()
            assert settings.debug_mode is True
            assert settings.log_validation_errors is True
            assert settings.detailed_error_messages is True

        finally:
            # Restore original values
            update_settings(
                debug_mode=original_debug, log_validation_errors=original_log, detailed_error_messages=original_detailed
            )


class TestValidationContext:
    """Test ValidationContext context manager."""

    def test_validation_context_basic(self):
        """Test basic ValidationContext functionality."""
        original_settings = get_settings()
        original_strict = original_settings.strict_validation

        with ValidationContext(strict_validation=True):
            settings = get_settings()
            assert settings.strict_validation is True

        # Should restore original value
        settings = get_settings()
        assert settings.strict_validation == original_strict

    def test_validation_context_multiple_settings(self):
        """Test ValidationContext with multiple settings."""
        original_settings = get_settings()
        original_strict = original_settings.strict_validation
        original_debug = original_settings.debug_mode

        with ValidationContext(strict_validation=True, debug_mode=True):
            settings = get_settings()
            assert settings.strict_validation is True
            assert settings.debug_mode is True

        # Should restore original values
        settings = get_settings()
        assert settings.strict_validation == original_strict
        assert settings.debug_mode == original_debug

    def test_validation_context_invalid_setting(self):
        """Test ValidationContext with invalid setting."""
        original_settings = get_settings()
        original_strict = original_settings.strict_validation

        # Should not raise error, just ignore invalid settings
        with ValidationContext(invalid_setting=True, strict_validation=True):
            settings = get_settings()
            assert settings.strict_validation is True
            assert not hasattr(settings, "invalid_setting")

        # Should restore original value
        settings = get_settings()
        assert settings.strict_validation == original_strict

    def test_validation_context_exception_handling(self):
        """Test ValidationContext restores settings even when exception occurs."""
        original_settings = get_settings()
        original_strict = original_settings.strict_validation

        try:
            with ValidationContext(strict_validation=True):
                settings = get_settings()
                assert settings.strict_validation is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should restore original value even after exception
        settings = get_settings()
        assert settings.strict_validation == original_strict

    def test_validation_context_nested(self):
        """Test nested ValidationContext."""
        original_settings = get_settings()
        original_strict = original_settings.strict_validation
        original_debug = original_settings.debug_mode

        with ValidationContext(strict_validation=True):
            settings = get_settings()
            assert settings.strict_validation is True
            assert settings.debug_mode == original_debug

            with ValidationContext(debug_mode=True):
                inner_settings = get_settings()
                assert inner_settings.strict_validation is True
                assert inner_settings.debug_mode is True

            # Should restore debug_mode but keep strict_validation
            after_inner = get_settings()
            assert after_inner.strict_validation is True
            assert after_inner.debug_mode == original_debug

        # Should restore all original values
        final_settings = get_settings()
        assert final_settings.strict_validation == original_strict
        assert final_settings.debug_mode == original_debug
