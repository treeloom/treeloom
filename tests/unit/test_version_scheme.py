"""Treeloom versions are CalVer: the release date, ``YYYY.M.D``.

Rules (docs/docker-images.md, "Versioning"):

- no zero padding (``2026.9.2``, never ``2026.09.02``) so the PEP 440
  normalised version and the git tag are the same string;
- a same-day re-release appends a counter: ``2026.9.22.1``;
- the git tag is the version with a ``v`` prefix and must equal
  ``pyproject.toml`` — the publish workflow refuses a mismatch.
"""
import datetime as _dt
import pathlib
import re
import tomllib

CALVER = re.compile(r"^(?P<y>\d{4})\.(?P<m>[1-9]|1[0-2])\.(?P<d>[1-9]|[12]\d|3[01])(?:\.(?P<n>[1-9]\d*))?$")

_PYPROJECT = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"


def _project_version() -> str:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_pyproject_version_is_calver():
    version = _project_version()
    assert CALVER.match(version), f"{version!r} is not YYYY.M.D[.N] CalVer"


def test_pyproject_version_is_a_real_date():
    m = CALVER.match(_project_version())
    assert m is not None
    _dt.date(int(m["y"]), int(m["m"]), int(m["d"]))  # raises on e.g. 2026.2.30


def test_calver_shape_examples():
    assert CALVER.match("2026.9.22")
    assert CALVER.match("2026.9.22.1")
    assert CALVER.match("2026.10.1")
    assert not CALVER.match("2026.09.22"), "zero padding is not allowed"
    assert not CALVER.match("0.4.0")
    assert not CALVER.match("v2026.9.22"), "the v prefix belongs to the git tag only"
    assert not CALVER.match("2026.9.22.0"), "the same-day counter starts at 1"
