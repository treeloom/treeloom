"""finding 20 CWE-835 — the preflight scanner would read a character device forever.

os.walk lists a symlink among filenames, os.path.getsize follows it and a
character device reports 0 bytes, so the size cap never fired. The reader
then used `for line in f`, which is bounded by the next newline rather than
by any byte count — and /dev/zero never produces one, so the first "line"
grew until the process died. Reproduced against the pre-fix code: MemoryError
with a 2GB rlimit, unbounded without one.

Reachability is the part worth stating: /preflight's `url` mode shallow-clones
a REMOTE repository and scans it, so a repo containing `evil.py -> /dev/zero`
takes the indexer down from the other side of the network.

Two independent fixes, tested separately — walk_repo refuses non-regular
files, and the reader is block-based so it cannot outrun its cap regardless
of what it is pointed at.
"""

import os
import stat

import pytest

from treeloom.preflight import scanner


@pytest.fixture
def devzero_link(tmp_path):
    link = tmp_path / "evil.py"
    try:
        link.symlink_to("/dev/zero")
    except OSError:  # pragma: no cover
        pytest.skip("cannot create symlink here")
    if not os.path.exists("/dev/zero"):  # pragma: no cover
        pytest.skip("/dev/zero unavailable")
    return link


class TestReaderIsBounded:
    def test_character_device_terminates(self, devzero_link):
        """The decisive one. Pre-fix this never returned."""
        loc, max_line = scanner._count_loc_and_max_line(str(devzero_link), 0)
        assert max_line <= scanner._LOC_READ_CAP_BYTES
        assert loc >= 0

    def test_it_does_not_read_past_the_cap(self, devzero_link):
        _, max_line = scanner._count_loc_and_max_line(str(devzero_link), 0)
        assert max_line == scanner._LOC_READ_CAP_BYTES, (
            "a stream with no newline should stop exactly at the cap"
        )

    def test_does_not_use_line_iteration(self):
        """`for line in f` is bounded by the next newline, not by bytes — the
        cap cannot be enforced from inside it.

        Checks the CODE, not the docstring: the docstring names the pattern it
        avoids, and a naive substring search over the source matches that and
        passes for the wrong reason.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(scanner._count_loc_and_max_line))
        )
        fn = tree.body[0]
        if ast.get_docstring(fn):
            fn.body = fn.body[1:]
        code = ast.dump(ast.Module(body=fn.body, type_ignores=[]))
        assert "'line'" not in code, "still iterating the handle line by line"
        assert "'read'" in code, "should read in fixed-size blocks"


class TestReaderStillCountsCorrectly:
    """The hardening must not change the numbers preflight calibrates on."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            (b"a\nbb\nccc", (3, 3)),        # no trailing newline
            (b"a\nbb\nccc\n", (3, 4)),      # trailing newline counts in the line
            (b"", (0, 0)),
            (b"\n", (1, 1)),
            (b"no-newline-at-all", (1, 17)),
            (b"\n\n\n", (3, 1)),
        ],
    )
    def test_counts(self, tmp_path, content, expected):
        f = tmp_path / "f.py"
        f.write_bytes(content)
        assert scanner._count_loc_and_max_line(str(f), len(content)) == expected

    def test_a_line_spanning_block_boundaries(self, tmp_path):
        """The block reader tracks the run length across reads; an off-by-one
        here would silently mis-measure long minified files."""
        long_line = b"x" * (scanner._BLOCK_SIZE * 2 + 7)
        f = tmp_path / "f.py"
        f.write_bytes(long_line + b"\n" + b"short\n")
        loc, max_line = scanner._count_loc_and_max_line(
            str(f), len(long_line) + 7
        )
        assert loc == 2
        assert max_line == len(long_line) + 1

    def test_large_files_are_still_estimated_not_read(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_bytes(b"x")
        size = scanner._LOC_READ_CAP_BYTES + 1
        assert scanner._count_loc_and_max_line(str(f), size) == (size // 40, 0)


class TestWalkRefusesNonRegularFiles:
    def test_character_device_is_skipped_entirely(self, tmp_path, devzero_link):
        (tmp_path / "real.py").write_text("print('hi')\n")
        stats, _ = _walk(tmp_path)
        names = {s.rel_path for s in stats}
        assert "real.py" in names
        assert "evil.py" not in names, "a device link must not be scanned at all"

    def test_fifo_is_skipped(self, tmp_path):
        fifo = tmp_path / "pipe.py"
        try:
            os.mkfifo(fifo)
        except (AttributeError, OSError):  # pragma: no cover
            pytest.skip("mkfifo unavailable")
        (tmp_path / "real.py").write_text("x = 1\n")
        stats, _ = _walk(tmp_path)
        assert {s.rel_path for s in stats} == {"real.py"}

    def test_a_symlink_to_a_regular_file_is_still_scanned(self, tmp_path):
        """Refusing every symlink would break ordinary repo layouts; the check
        is on the TARGET's type, not on being a link."""
        target = tmp_path / "real.py"
        target.write_text("a\nb\n")
        (tmp_path / "alias.py").symlink_to(target)
        stats, _ = _walk(tmp_path)
        assert {s.rel_path for s in stats} == {"real.py", "alias.py"}


def _walk(root):
    """walk_repo with gitignore filtering off, so tmp dirs behave."""
    import inspect

    sig = inspect.signature(scanner.walk_repo)
    kwargs = {}
    if "use_gitignore" in sig.parameters:
        kwargs["use_gitignore"] = False
    result = scanner.walk_repo(str(root), **kwargs)
    if isinstance(result, tuple):
        return result[0], result[1:]
    return result, ()
