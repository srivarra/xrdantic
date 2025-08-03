"""Tests for configuration management."""

import pytest

from xrdantic.config import (
    ValidationContext,
    XrdanticSettings,
    get_settings,
    reset_settings,
    update_settings,
)


@pytest.fixture
def managed_xrdantic_settings():
    """Fixture to provide xrdantic settings and ensure they are reset after the test."""
    settings = get_settings()
    yield settings
    reset_settings()


class TestXrdanticSettings:
    """Test XrdanticSettings model."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = XrdanticSettings()
        assert settings.strict_validation is False
        assert settings.allow_nan_values is True
        assert settings.allow_inf_values is False
        assert settings.debug_mode is False

    def test_settings_assignment_validation(self):
        """Test validate_assignment config."""
        settings = XrdanticSettings()
        settings.strict_validation = True
        assert settings.strict_validation is True

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        settings = XrdanticSettings(**{"unknown_field": True})  # type: ignore
        assert not hasattr(settings, "unknown_field")


class TestSettingsFunctions:
    """Test settings management functions."""

    def test_get_settings(self, managed_xrdantic_settings):
        """Test get_settings function."""
        settings1 = managed_xrdantic_settings
        settings2 = get_settings()
        assert settings1 is settings2
        assert isinstance(settings1, XrdanticSettings)

    def test_update_settings(self, managed_xrdantic_settings):
        """Test update_settings function."""
        update_settings(strict_validation=True, debug_mode=True)

        updated_settings = get_settings()
        assert updated_settings.strict_validation is True
        assert updated_settings.debug_mode is True

        with pytest.raises(ValueError, match=r"Invalid setting: 'non_existent_field'\. Valid settings are: .*"):
            update_settings(non_existent_field=True)


class TestValidationContext:
    """Test ValidationContext context manager."""

    def test_validation_context_basic(self, managed_xrdantic_settings):
        """Test basic ValidationContext functionality."""
        original_strict = managed_xrdantic_settings.strict_validation

        with ValidationContext(strict_validation=True):
            settings = get_settings()
            assert settings.strict_validation is True

        restored_settings = get_settings()
        assert restored_settings.strict_validation == original_strict
        assert restored_settings is managed_xrdantic_settings

    def test_validation_context_multiple_settings(self, managed_xrdantic_settings):
        """Test ValidationContext with multiple settings."""
        original_strict = managed_xrdantic_settings.strict_validation
        original_debug = managed_xrdantic_settings.debug_mode

        with ValidationContext(strict_validation=True, debug_mode=True):
            settings = get_settings()
            assert settings.strict_validation is True
            assert settings.debug_mode is True

        restored_settings = get_settings()
        assert restored_settings.strict_validation == original_strict
        assert restored_settings.debug_mode == original_debug
        assert restored_settings is managed_xrdantic_settings

    def test_validation_context_invalid_setting(self, managed_xrdantic_settings):
        """Test ValidationContext with invalid setting."""
        with pytest.raises(ValueError, match=r"Invalid temporary setting: 'invalid_setting'\. Valid settings are: .*"):
            with ValidationContext(invalid_setting=True, strict_validation=True):
                pass

    def test_validation_context_exception_handling(self, managed_xrdantic_settings):
        """Test ValidationContext restores settings even when exception occurs."""
        original_strict = managed_xrdantic_settings.strict_validation

        try:
            with ValidationContext(strict_validation=True):
                settings = get_settings()
                assert settings.strict_validation is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        restored_settings = get_settings()
        assert restored_settings.strict_validation == original_strict
        assert restored_settings is managed_xrdantic_settings

    def test_validation_context_nested(self, managed_xrdantic_settings):
        """Test nested ValidationContext."""
        original_strict = managed_xrdantic_settings.strict_validation
        original_debug = managed_xrdantic_settings.debug_mode

        with ValidationContext(strict_validation=True):
            settings = get_settings()
            assert settings.strict_validation is True
            assert settings.debug_mode == original_debug

            with ValidationContext(debug_mode=True):
                inner_settings = get_settings()
                assert inner_settings.strict_validation is True
                assert inner_settings.debug_mode is True

            after_inner_settings = get_settings()
            assert after_inner_settings.strict_validation is True
            assert after_inner_settings.debug_mode == original_debug

        final_settings = get_settings()
        assert final_settings.strict_validation == original_strict
        assert final_settings.debug_mode == original_debug
        assert final_settings is managed_xrdantic_settings
