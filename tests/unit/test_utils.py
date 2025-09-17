"""Tests for utility functions."""

import pytest

from agentic_lab.core.exceptions import ValidationError
from agentic_lab.utils.logging import setup_logger
from agentic_lab.utils.validators import (
    validate_config,
    validate_positive_number,
    validate_string_not_empty,
)


def test_setup_logger():
    """Test logger setup."""
    logger = setup_logger("test.logger", "DEBUG")
    assert logger.name == "test.logger"
    assert logger.level == 10  # DEBUG level


def test_setup_logger_with_custom_format():
    """Test logger setup with custom format."""
    custom_format = "%(name)s - %(message)s"
    logger = setup_logger("test.custom", format_string=custom_format)
    assert logger.name == "test.custom"


def test_validate_config_success():
    """Test successful config validation."""
    config = {"name": "test", "value": 42}
    required_keys = ["name", "value"]

    # Should not raise any exception
    validate_config(config, required_keys)


def test_validate_config_missing_keys():
    """Test config validation with missing keys."""
    config = {"name": "test"}
    required_keys = ["name", "value", "other"]

    with pytest.raises(ValidationError) as exc_info:
        validate_config(config, required_keys)

    assert "Missing required configuration keys: value, other" in str(exc_info.value)


def test_validate_positive_number_valid():
    """Test positive number validation with valid values."""
    validate_positive_number(10, "test_value")
    validate_positive_number(3.14, "test_value")
    validate_positive_number(0.1, "test_value")


def test_validate_positive_number_invalid():
    """Test positive number validation with invalid values."""
    with pytest.raises(ValidationError, match="test_value must be a positive number"):
        validate_positive_number(-1, "test_value")

    with pytest.raises(ValidationError, match="test_value must be a positive number"):
        validate_positive_number(0, "test_value")

    with pytest.raises(ValidationError, match="test_value must be a positive number"):
        validate_positive_number("not_a_number", "test_value")


def test_validate_string_not_empty_valid():
    """Test string validation with valid values."""
    validate_string_not_empty("valid string", "test_string")
    validate_string_not_empty("  valid with spaces  ", "test_string")


def test_validate_string_not_empty_invalid():
    """Test string validation with invalid values."""
    with pytest.raises(ValidationError, match="test_string must be a non-empty string"):
        validate_string_not_empty("", "test_string")

    with pytest.raises(ValidationError, match="test_string must be a non-empty string"):
        validate_string_not_empty("   ", "test_string")

    with pytest.raises(ValidationError, match="test_string must be a non-empty string"):
        validate_string_not_empty(123, "test_string")
