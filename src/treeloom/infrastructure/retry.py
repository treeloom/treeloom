"""Retry with exponential backoff for transient errors."""
import asyncio
import logging

logger = logging.getLogger(__name__)

DEFAULT_RETRY_EXCEPTIONS = (OSError, UnicodeDecodeError, ConnectionError, TimeoutError)


async def retry_with_backoff(
    fn,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = DEFAULT_RETRY_EXCEPTIONS,
    skip_on: tuple = (UnicodeDecodeError,),
):
    """Call async fn, retry on transient exceptions with exponential backoff.

    Args:
        fn: Async callable to retry.
        max_attempts: Total attempts (including first).
        base_delay: Initial delay in seconds, doubles each attempt.
        exceptions: Exception types to retry on.
        skip_on: Exception types to skip silently instead of failing.

    Returns:
        fn() result on success.

    Raises:
        Last exception if all attempts exhausted (not in skip_on).
    """
    for attempt in range(max_attempts):
        try:
            return await fn()
        except skip_on as e:
            logger.warning("Skipping (non-retryable): %s", e)
            raise SkipError(str(e)) from e
        except exceptions as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt + 1, max_attempts, e, delay,
            )
            await asyncio.sleep(delay)


class SkipError(Exception):
    """Raised when an error should be skipped, not retried."""
