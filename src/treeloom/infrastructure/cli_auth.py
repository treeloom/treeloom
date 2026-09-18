"""Bearer-token resolution for the operator CLIs.

`--key <token>` puts the credential in the process argument list, where any
local user can read it out of `ps` or /proc/<pid>/cmdline for as long as the
command runs, and where it lands in the shell history file afterwards
(CWE-214). Neither is visible to the person typing it.

The flag still works — breaking every documented invocation to fix a local
disclosure would be a poor trade — but it warns, and there are now two ways
to pass a token that do not expose it:

* ``--key-file PATH`` reads the token from a file (permissions are the
  operator's to set).
* ``--key -`` reads it from stdin, so it can be piped from a secret manager
  without ever touching disk: ``vault read -field=token ... | cmd --key -``

Precedence: --key-file, then --key, then $TREELOOM_KEY, then
$TREELOOM_MCP_API_KEY. Environment first would silently ignore an explicit
flag.
"""

from __future__ import annotations

import os
import sys


_ARGV_WARNING = (
    "warning: --key puts the token in this process's argument list, where any "
    "local user can read it (ps / /proc), and in your shell history. "
    "Prefer --key-file PATH, or `--key -` to read it from stdin."
)


def add_key_arguments(parser) -> None:
    """Register the --key / --key-file pair on an argparse parser."""
    parser.add_argument(
        "--key",
        default=None,
        help=(
            "Bearer API key. Use '-' to read it from stdin. Passing the token "
            "literally exposes it in ps/shell history — prefer --key-file. "
            "(default: $TREELOOM_KEY or $TREELOOM_MCP_API_KEY)"
        ),
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="Read the bearer API key from this file instead of the command line.",
    )


def resolve_key(key: str | None, key_file: str | None = None) -> str | None:
    """Resolve the bearer token from flags then environment.

    Returns None when no token is configured; callers treat that as
    "unauthenticated", which is correct against an indexer with auth off.
    """
    if key_file:
        with open(key_file, "r", encoding="utf-8") as fh:
            token = fh.read().strip()
        return token or None

    if key == "-":
        token = sys.stdin.read().strip()
        return token or None

    if key:
        print(_ARGV_WARNING, file=sys.stderr)
        return key

    return (
        os.environ.get("TREELOOM_KEY")
        or os.environ.get("TREELOOM_MCP_API_KEY")
        or None
    )
