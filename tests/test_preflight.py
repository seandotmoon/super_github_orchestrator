"""Unit tests for preflight git hygiene."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from super_orchestrator.preflight import ORCHESTRATOR_EXCLUDE_PATTERN, ensure_orchestrator_excluded

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )
    return completed.stdout


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    return path


def test_exclude_added_once_and_hides_orchestrator_dir(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    exclude = ensure_orchestrator_excluded(repo)
    ensure_orchestrator_excluded(repo)
    content = exclude.read_text(encoding="utf-8")
    assert content.count(ORCHESTRATOR_EXCLUDE_PATTERN) == 1

    (repo / ".orchestrator" / "logs").mkdir(parents=True)
    (repo / ".orchestrator" / "logs" / "1-implement-1.log").write_text("x", encoding="utf-8")
    (repo / "real.txt").write_text("y", encoding="utf-8")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    assert "real.txt" in status
    assert ".orchestrator" not in status


def test_exclude_works_in_linked_worktree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    worktree = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(worktree))
    assert (worktree / ".git").is_file()

    ensure_orchestrator_excluded(worktree)
    (worktree / ".orchestrator").mkdir()
    (worktree / ".orchestrator" / "state.sqlite").write_text("x", encoding="utf-8")
    status = _git(worktree, "status", "--porcelain", "--untracked-files=all")
    assert ".orchestrator" not in status
