"""Backward-compatible re-export of the indexer_service module.

`application/indexer_service.py` has been split (indexer_state,
indexer_runners, indexer_authz), so the names below no longer all live in one
module. They are re-exported from wherever they now are, keeping
`from treeloom.indexer_service import X` working for existing callers.

This is an import-compatibility shim ONLY. Do not patch through it: a
re-exported name has its own binding here, so replacing it leaves the module
that actually runs the code reading its original. Patch the defining module —
indexer_authz, indexer_runners or indexer_state.
"""

from treeloom.application.indexer_service import (  # noqa: F401
    IndexDirectoryRequest,
    IndexFileRequest,
    IndexRepoRequest,
    JobAck,
    SKIP_FILENAMES,
    USE_SUMMARY_VECTOR,
    LOG_INTERVAL,
    app,
    delete_job,
    get_job,
    get_sources,
    handle_build_community,
    handle_index_directory,
    handle_index_file,
    handle_index_repo,
    health,
    indexer,
    list_jobs,
    remove_source,
    status,
)
from treeloom.application.indexer_authz import _public_job  # noqa: F401
from treeloom.application.indexer_runners import (  # noqa: F401
    LOG_FILE_INTERVAL,
    _collect_files,
    _fail_job,
    _finalize_job,
    _graph_index_file,
    _log_progress,
    _new_job,
    _process_file,
    _reset_source,
    _run_incremental_index,
    _run_index_directory_job,
    _run_index_file_job,
    _run_index_repo_job,
    _summaries_for_chunks,
    _walk_and_index,
)
from treeloom.application.routes_webhook import (  # noqa: F401
    WebhookRequest,
    handle_webhook,
)
from treeloom.application.lifecycle import shutdown  # noqa: F401
