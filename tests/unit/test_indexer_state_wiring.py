"""The shared-state module must be read through the module, not imported by name.

`indexer_service.py` is being split, and the pieces need somewhere common to
find the stores and the job table. The hazard in doing that is subtle and
silent: a module that does

    from .indexer_state import _source_repo

binds whatever object existed at import time. Rebinding
`indexer_state._source_repo` afterwards — which is what startup does for the
job stores, and what every test does for the adapters — then has no effect on
that module, while the tests covering it still pass. The service and its
extracted parts would disagree about which store is live, on the auth path,
with nothing failing.

These tests assert the wiring itself rather than any behaviour built on it,
so they keep holding as more code moves out of indexer_service.
"""

import ast
import pathlib

import pytest

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as state


class TestRebindingIsObserved:
    """The property that makes the split safe."""

    def test_service_sees_a_rebound_store(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(state, "_source_repo", sentinel)
        # Read the way production code does.
        assert svc._state._source_repo is sentinel

    def test_service_sees_a_rebound_job_store(self, monkeypatch):
        """_job_store is None until startup rebinds it — the case that would
        break hardest under a by-name import."""
        sentinel = object()
        monkeypatch.setattr(state, "_job_store", sentinel)
        assert svc._state._job_store is sentinel

    def test_the_job_table_is_one_shared_object(self):
        """_jobs is mutated, never rebound, so identity is what matters —
        and it is why patch.dict works across modules."""
        assert svc._state._jobs is state._jobs


class TestNoModuleImportsStateByName:
    """A `from ... import <name>` of any state name is the bug this guards."""

    STATE_NAMES = {
        "_source_repo", "_user_store", "_group_store", "_grant_store",
        "_search_audit", "_token_store", "_api_key_store", "_session_store",
        "_jobs", "_last_job_id", "_job_store", "_job_group_store", "_job_queue",
        "_job_file_error_store", "_login_attempt_store", "_community_build_state",
        "_community_build_task", "_freshness_sampler_task", "_login_purge_task",
        "_fleet_refresh_task",
    }

    def test_no_source_file_imports_a_state_name(self):
        root = pathlib.Path("src/treeloom")
        offenders = []
        for f in root.rglob("*.py"):
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if not node.module.endswith("indexer_state"):
                        continue
                    for alias in node.names:
                        if alias.name in self.STATE_NAMES:
                            offenders.append(f"{f}:{node.lineno} imports {alias.name}")
        assert offenders == [], (
            "import the MODULE and read attributes off it:\n  " + "\n  ".join(offenders)
        )

    def test_state_module_has_no_application_imports(self):
        """It must stay a leaf, or it can re-enter an import cycle."""
        tree = ast.parse(
            pathlib.Path("src/treeloom/application/indexer_state.py").read_text()
        )
        bad = [
            n.module for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.module
            and n.module.startswith("treeloom.application")
        ]
        assert bad == [], f"indexer_state must not import from application: {bad}"


class TestServiceKeepsNoLocalCopies:
    def test_indexer_service_defines_no_state_names(self):
        """A module-level `_jobs = {}` left behind in indexer_service would
        shadow the shared one for every reader in that file."""
        tree = ast.parse(
            pathlib.Path("src/treeloom/application/indexer_service.py").read_text()
        )
        defined = set()
        for node in tree.body:
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            for t in targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
        clash = defined & TestNoModuleImportsStateByName.STATE_NAMES
        assert clash == set(), f"shadowing shared state: {sorted(clash)}"


class TestSplitModulesStayAcyclic:
    """indexer_service imports the extracted modules; they must not import it
    back. A cycle here would not fail loudly — Python would serve a
    half-initialised module and the symptom would surface later, somewhere
    else, as a missing attribute."""

    LEAVES = ["indexer_state.py", "indexer_runners.py", "indexer_authz.py",
              "routes_auth.py", "routes_webhook.py"]

    @pytest.mark.parametrize("leaf", LEAVES)
    def test_leaf_does_not_import_the_service(self, leaf):
        tree = ast.parse(
            (pathlib.Path("src/treeloom/application") / leaf).read_text()
        )
        bad = []
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and "indexer_service" in n.module:
                bad.append(f"line {n.lineno}")
            if isinstance(n, ast.Import):
                bad += [f"line {n.lineno}" for a in n.names if "indexer_service" in a.name]
        assert bad == [], f"{leaf} imports indexer_service ({', '.join(bad)}) — cycle"

    def test_the_service_does_import_them(self):
        """The direction that SHOULD exist — if this breaks, the re-exports
        are gone and callers are silently getting something else."""
        src = pathlib.Path("src/treeloom/application/indexer_service.py").read_text()
        assert "from treeloom.application import indexer_state" in src
        assert "from treeloom.application import indexer_runners as runners" in src
        assert "from treeloom.application import indexer_authz as authz" in src


class TestSplitModulesImportEachOtherByModuleOnly:
    """Cross-module access within the split must be module-qualified.

    `from .indexer_runners import compute_source_staleness` binds the object
    that existed at import time. Patching `indexer_runners.compute_source_staleness`
    afterwards then reaches the defining module and NOT the importer, so one
    caller sees the patch and another does not — which one depends on where
    the call happens to live.

    This exact mistake was made five separate times while splitting this
    module (GIT_ALLOWED_HOSTS, the login constants, MAX_QUEUE_DEPTH,
    DATABASE_URL, compute_source_staleness), each time caught by a test
    failing rather than by review. Enforcing it is cheaper than remembering
    it.

    `indexer_state` is exempt as an IMPORT TARGET in one direction only —
    nothing may import names from it either; that is covered above.
    """

    SPLIT = ["indexer_service", "indexer_state", "indexer_runners",
             "indexer_authz", "routes_auth", "routes_webhook", "lifecycle"]

    @pytest.mark.parametrize("module", SPLIT)
    def test_no_name_imported_from_a_sibling(self, module):
        path = pathlib.Path("src/treeloom/application") / f"{module}.py"
        tree = ast.parse(path.read_text())
        offenders = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.ImportFrom) or not n.module:
                continue
            target = n.module.rsplit(".", 1)[-1]
            if target in self.SPLIT and target != module:
                names = ", ".join(a.name for a in n.names)
                offenders.append(f"line {n.lineno}: from {target} import {names}")
        assert offenders == [], (
            f"{module} imports names from a sibling; import the module and "
            f"qualify instead:\n  " + "\n  ".join(offenders)
        )

    def test_every_split_module_is_importable(self):
        """Cheap canary: a cycle or a missing symbol shows up here first."""
        import importlib

        for m in self.SPLIT:
            importlib.import_module(f"treeloom.application.{m}")
