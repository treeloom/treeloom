"""Shared source-file reading with loud (not silent) encoding fallback."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_source(file_path: str) -> str:
    """Read a source file as UTF-8, falling back to replacement chars LOUDLY.

    Files that aren't valid UTF-8 are still indexed (legacy encodings are
    common and usually harmless), but the fallback is visible: a warning
    log line and the `treeloom_encoding_fallbacks_total` counter. We do
    NOT record a job_file_errors row — that would inflate job error
    counts for benign legacy-encoded files.
    """
    raw = Path(file_path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        from treeloom.infrastructure import metrics

        metrics.encoding_fallbacks.inc()
        logger.warning(
            "%s is not valid UTF-8 (%s) — indexing with replacement "
            "characters; affected snippets may contain U+FFFD",
            file_path, exc,
        )
        return raw.decode("utf-8", errors="replace")
