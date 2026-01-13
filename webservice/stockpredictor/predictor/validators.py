# webservice/stockpredictor/predictor/validators.py
"""
Input validation utilities for the stock predictor application.
Provides security-focused validation for user inputs.
"""
import re
import logging
from typing import Tuple, Optional
from django.conf import settings
from django.core.exceptions import ValidationError as DjangoValidationError

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Custom validation error with user-friendly messages.

    Named differently from Django's ValidationError to avoid shadowing.
    """

    def __init__(self, message: str, code: str = 'invalid'):
        self.message = message
        self.code = code
        super().__init__(self.message)


def validate_ticker(ticker: str) -> Tuple[bool, Optional[str], str]:
    """
    Validate a stock ticker symbol.

    Args:
        ticker: The ticker symbol to validate

    Returns:
        Tuple of (is_valid, error_message, sanitized_ticker)

    Security considerations:
        - Prevents injection attacks via ticker parameter
        - Limits length to prevent buffer/cache issues
        - Restricts to alphanumeric characters plus common separators
    """
    if not ticker:
        return False, "Ticker symbol is required", ""

    # Strip whitespace
    ticker = ticker.strip()

    # Check length
    max_length = getattr(settings, 'MAX_TICKER_LENGTH', 15)
    if len(ticker) > max_length:
        logger.warning(f"Ticker validation failed: too long ({len(ticker)} chars)")
        return False, f"Ticker symbol must be {max_length} characters or less", ""

    # Check for valid characters using pattern from settings
    pattern = getattr(settings, 'TICKER_PATTERN', r'^[A-Za-z0-9\-\.]+$')
    if not re.match(pattern, ticker):
        logger.warning(f"Ticker validation failed: invalid characters in '{ticker}'")
        return False, "Ticker symbol contains invalid characters", ""

    # Normalize to uppercase (standard ticker format)
    sanitized = ticker.upper()

    return True, None, sanitized


def validate_search_query(query: str) -> Tuple[bool, Optional[str], str]:
    """
    Validate a search query string.

    Args:
        query: The search query to validate

    Returns:
        Tuple of (is_valid, error_message, sanitized_query)

    Security considerations:
        - Prevents SQL injection via search parameter
        - Limits length to prevent DoS
        - Removes potentially dangerous characters
    """
    if not query:
        return True, None, ""  # Empty query is valid (returns no results)

    # Strip whitespace
    query = query.strip()

    # Check length
    max_length = getattr(settings, 'MAX_SEARCH_QUERY_LENGTH', 100)
    if len(query) > max_length:
        logger.warning(f"Search query validation failed: too long ({len(query)} chars)")
        return False, f"Search query must be {max_length} characters or less", ""

    # Remove potentially dangerous characters while keeping alphanumeric, spaces, and common punctuation
    # This prevents SQL injection and XSS attempts
    sanitized = re.sub(r'[^\w\s\-\.\,]', '', query)

    if sanitized != query:
        logger.info(f"Search query sanitized: removed special characters")

    return True, None, sanitized


def validate_positive_integer(value: str, field_name: str = 'value',
                              min_val: int = 1, max_val: int = 1000) -> Tuple[bool, Optional[str], int]:
    """
    Validate and parse a positive integer from string input.

    Args:
        value: The string value to validate
        field_name: Name of the field for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Tuple of (is_valid, error_message, parsed_integer)
    """
    if not value:
        return False, f"{field_name} is required", 0

    try:
        parsed = int(value)
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid integer", 0

    if parsed < min_val:
        return False, f"{field_name} must be at least {min_val}", 0

    if parsed > max_val:
        return False, f"{field_name} must be at most {max_val}", 0

    return True, None, parsed


def sanitize_cache_key(key: str) -> str:
    """
    Sanitize a string for use as a cache key.

    Args:
        key: The key to sanitize

    Returns:
        Sanitized cache key safe for use with Django cache

    Security considerations:
        - Prevents cache key injection
        - Ensures consistent key format
    """
    # Remove any characters that could cause cache issues
    # Keep only alphanumeric, underscore, and hyphen
    sanitized = re.sub(r'[^\w\-]', '_', key)

    # Limit length to prevent issues with some cache backends
    max_key_length = 250
    if len(sanitized) > max_key_length:
        # Use a hash suffix if truncated to maintain uniqueness
        import hashlib
        hash_suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        sanitized = sanitized[:max_key_length - 17] + '_' + hash_suffix

    return sanitized


def get_safe_error_message(exception: Exception, include_details: bool = False) -> str:
    """
    Get a safe error message that doesn't expose sensitive information.

    Args:
        exception: The exception to get a message for
        include_details: Whether to include exception details (only in DEBUG mode)

    Returns:
        A safe error message string
    """
    from django.conf import settings

    # Generic messages for different exception types
    error_messages = {
        'DoesNotExist': 'The requested resource was not found',
        'ValidationError': 'Invalid input provided',
        'InputValidationError': 'Invalid input provided',
        'PermissionDenied': 'Permission denied',
        'TimeoutError': 'The request timed out',
        'ConnectionError': 'Service temporarily unavailable',
    }

    exception_type = type(exception).__name__

    # Use generic message in production
    if not settings.DEBUG or not include_details:
        return error_messages.get(exception_type, 'An error occurred while processing your request')

    # In debug mode with details requested, include more info
    return f"{exception_type}: {str(exception)}"
