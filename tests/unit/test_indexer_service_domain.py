"""Domain tests for indexer_service — request model validation."""
import pytest
from treeloom.indexer_service import IndexFileRequest, IndexDirectoryRequest, IndexRepoRequest


def test_index_file_request_valid():
    req = IndexFileRequest(file_path="/home/user/source/repo/src/main.py")
    assert req.file_path == "/home/user/source/repo/src/main.py"


def test_index_directory_request_valid():
    req = IndexDirectoryRequest(directory="/home/user/source/repo/src/")
    assert req.directory == "/home/user/source/repo/src/"
    assert req.pattern == "**/*"


def test_index_repo_request_valid():
    req = IndexRepoRequest(path="/home/user/source/repo")
    assert req.path == "/home/user/source/repo"


def test_index_file_request_missing_path():
    """Missing required field should raise validation error."""
    with pytest.raises(Exception):
        IndexFileRequest()


def test_index_directory_request_missing_path():
    with pytest.raises(Exception):
        IndexDirectoryRequest()


def test_index_repo_request_defaults():
    """IndexRepoRequest has all optional fields — should work with defaults."""
    req = IndexRepoRequest()
    assert req.path is None
    assert req.url is None
    assert req.branch is None
