"""Preflight checks before the orchestrator loop starts."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from super_orchestrator.github import GitHubClient, GitHubError
from super_orchestrator.opencode import OpenCodeClient, OpenCodeError
from super_orchestrator.state import HaltPolicy


class PreflightError(RuntimeError):
    """Raised when a required precondition for orchestration is not met."""


REQUIRED_AGENTS: tuple[str, ...] = ("issue-worker", "pr-reviewer")

# Pattern appended to the target repo's local exclude file so agents running
# `git add -A` never commit orchestrator state or logs.
ORCHESTRATOR_EXCLUDE_PATTERN = "/.orchestrator/"


@dataclass(frozen=True)
class PreflightResult:
    """Successful preflight summary."""

    repo: str
    workdir: Path
    agent_ids: frozenset[str]


def ensure_orchestrator_excluded(workdir: Path) -> Path:
    """
    Add `.orchestrator/` to the target checkout's `.git/info/exclude`.

    Uses `git rev-parse --git-path` so linked worktrees and submodules (where
    `.git` is a file) resolve to the right exclude file. The exclude file is
    local-only, so the target repo's tracked `.gitignore` is not modified.

    Returns:
        Path of the exclude file.

    Raises:
        PreflightError: When git is unavailable or the file cannot be written.
    """
    if shutil.which("git") is None:
        raise PreflightError("git was not found on PATH")
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--git-path", "info/exclude"],
            cwd=str(workdir),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError(f"failed to run git rev-parse in {workdir}: {exc}") from exc
    if completed.returncode != 0:
        raise PreflightError(
            f"git rev-parse failed in {workdir}: {completed.stderr.strip()}"
        )
    raw_path = completed.stdout.strip()
    if raw_path == "":
        raise PreflightError("git rev-parse returned an empty exclude path")
    exclude_path = Path(raw_path)
    if not exclude_path.is_absolute():
        exclude_path = workdir / exclude_path

    try:
        existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
        entries = {line.strip() for line in existing.splitlines()}
        if not entries.intersection({ORCHESTRATOR_EXCLUDE_PATTERN, ".orchestrator/", ".orchestrator"}):
            exclude_path.parent.mkdir(parents=True, exist_ok=True)
            separator = "" if existing == "" or existing.endswith("\n") else "\n"
            addition = "\n".join(
                ["# super-orchestrator state and logs", ORCHESTRATOR_EXCLUDE_PATTERN, ""]
            )
            with exclude_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{separator}{addition}")
    except OSError as exc:
        raise PreflightError(f"failed to update {exclude_path}: {exc}") from exc
    return exclude_path


def run_preflight(
    *,
    repo: str,
    workdir: Path,
    github: GitHubClient,
    opencode: OpenCodeClient,
    require_agents: bool = True,
    on_halt: HaltPolicy = "stop",
) -> PreflightResult:
    """
    Verify binaries, workdir, repo match, required agents, and local git hygiene.

    Args:
        repo: Expected owner/name slug.
        workdir: Target repository checkout.
        github: GitHub client bound to the same workdir/repo.
        opencode: OpenCode client bound to the same workdir.
        require_agents: When True, fail if issue-worker / pr-reviewer are missing.
        on_halt: Halt policy; `skip` requires the skip label to exist.

    Returns:
        PreflightResult on success.

    Raises:
        PreflightError: When any check fails.
    """
    if not isinstance(repo, str) or "/" not in repo:
        raise PreflightError("repo must look like 'owner/name'")
    if not isinstance(workdir, Path):
        raise PreflightError("workdir must be a Path")

    if shutil.which("opencode") is None:
        raise PreflightError("opencode was not found on PATH (expected inside the container)")
    if shutil.which("gh") is None:
        raise PreflightError("gh was not found on PATH")

    if not workdir.exists() or not workdir.is_dir():
        raise PreflightError(f"workdir does not exist or is not a directory: {workdir}")
    # `.git` is a directory in normal clones and a file in worktrees/submodules.
    if not (workdir / ".git").exists():
        raise PreflightError(f"workdir is not a git checkout (missing .git): {workdir}")

    try:
        remote_slug = github.resolve_remote_slug()
    except GitHubError as exc:
        raise PreflightError(f"failed to resolve gh repo for workdir: {exc}") from exc
    if remote_slug.lower() != repo.lower():
        raise PreflightError(
            f"workdir remote {remote_slug!r} does not match --repo {repo!r}"
        )

    try:
        github.viewer_login()
    except GitHubError as exc:
        raise PreflightError(f"failed to resolve the authenticated gh user: {exc}") from exc

    ensure_orchestrator_excluded(workdir)

    if on_halt == "skip":
        try:
            github.ensure_label(
                github.skip_label,
                description="super-orchestrator will not process this issue",
            )
        except GitHubError as exc:
            raise PreflightError(
                f"failed to create skip label {github.skip_label!r}: {exc}"
            ) from exc

    agent_ids: frozenset[str] = frozenset()
    if require_agents:
        try:
            agent_ids = frozenset(opencode.list_agent_ids())
        except OpenCodeError as exc:
            raise PreflightError(f"failed to list OpenCode agents: {exc}") from exc
        missing = [name for name in REQUIRED_AGENTS if name not in agent_ids]
        if len(missing) > 0:
            raise PreflightError(
                " ".join(
                    [
                        f"required OpenCode primary agents missing: {', '.join(missing)}.",
                        "Copy templates/target/.opencode into the target repository.",
                    ]
                )
            )

    return PreflightResult(repo=repo, workdir=workdir, agent_ids=agent_ids)
