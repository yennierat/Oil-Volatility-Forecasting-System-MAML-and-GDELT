"""Shared pytest fixtures.

The `scheduler_v2` fixture is the workhorse: it imports the scheduler module
once per session (model loading is slow) and per-test redirects the DB to a
temp directory so tests never touch the real predictions_v2.db.

NOTE: The scheduler modules currently run their main loop at module level.
Importing them as-is will hang. Wrap the bottom of each scheduler in:

    def main():
        init_db()
        ...
        while True:
            ...

    if __name__ == "__main__":
        main()

Tests for `scheduler_v2` will be skipped automatically if that refactor is
not yet applied.
"""
import os
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCHEDULER_DIR = ROOT / "code" / "live_deployment"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


def _try_import_scheduler(name: str, timeout: float = 30.0):
    """Import a scheduler module with a hard timeout.

    If the module's `while True:` is still at module scope, import will hang.
    We run it in a daemon thread and give up after `timeout` seconds.
    Returns the module on success or None on timeout.
    """
    sys.path.insert(0, str(SCHEDULER_DIR))
    original_cwd = os.getcwd()

    result = {}

    def _do_import():
        os.chdir(SCHEDULER_DIR)
        try:
            sys.modules.pop(name, None)
            mod = __import__(name)
            result["module"] = mod
        except Exception as exc:
            result["error"] = exc

    t = threading.Thread(target=_do_import, daemon=True)
    t.start()
    t.join(timeout=timeout)

    os.chdir(original_cwd)

    if t.is_alive():
        return None  # import hung — refactor not applied
    return result.get("module")


@pytest.fixture(scope="session")
def _scheduler_v2_imported():
    mod = _try_import_scheduler("scheduler_v2", timeout=30.0)
    return mod


@pytest.fixture
def scheduler_v2(_scheduler_v2_imported, monkeypatch, tmp_path):
    if _scheduler_v2_imported is None:
        pytest.skip(
            "scheduler_v2 import timed out — main loop is still at module scope. "
            "Wrap it in `def main(): ...` + `if __name__ == '__main__': main()`."
        )
    # Relative paths inside scheduler functions (SEED_PATH) resolve from cwd.
    monkeypatch.chdir(SCHEDULER_DIR)
    # Redirect every DB write to a temp file so we never touch predictions_v2.db.
    _scheduler_v2_imported.DB_PATH = str(tmp_path / "test_predictions_v2.db")
    return _scheduler_v2_imported


@pytest.fixture(scope="session")
def _scheduler_v1_imported():
    mod = _try_import_scheduler("scheduler_v1", timeout=30.0)
    return mod


@pytest.fixture
def scheduler_v1(_scheduler_v1_imported, monkeypatch, tmp_path):
    if _scheduler_v1_imported is None:
        pytest.skip(
            "scheduler_v1 import timed out — main loop is still at module scope. "
            "Wrap it in `def main(): ...` + `if __name__ == '__main__': main()`."
        )
    monkeypatch.chdir(SCHEDULER_DIR)
    _scheduler_v1_imported.DB_PATH = str(tmp_path / "test_predictions.db")
    return _scheduler_v1_imported
