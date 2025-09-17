"""Validation utilities for agentic lab."""

from typing import Any, Dict, List

from ..core.exceptions import ValidationError


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that a configuration dictionary contains required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key names

    Raises:
        ValidationError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValidationError(
            f"Missing required configuration keys: {', '.join(missing_keys)}"
        )


def validate_positive_number(value: Any, name: str) -> None:
    """
    Validate that a value is a positive number.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is not a positive number
    """
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValidationError(f"{name} must be a positive number, got: {value}")


def validate_string_not_empty(value: Any, name: str) -> None:
    """
    Validate that a value is a non-empty string.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is not a non-empty string
    """
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string, got: {value}")
