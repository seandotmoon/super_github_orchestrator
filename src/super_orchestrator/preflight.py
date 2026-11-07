"""Preflight checks before the orchestrator loop starts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from super_orchestrator.github import GitHubClient, GitHubError
from super_orchestrator.opencode import OpenCodeClient, OpenCodeError


class PreflightError(RuntimeError):
    """Raised when a required precondition for orchestration is not met."""


REQUIRED_AGENTS: tuple[str, ...] = ("issue-worker", "pr-reviewer")


@dataclass(frozen=True)
class PreflightResult:
    """Successful preflight summary."""

    repo: str
    workdir: Path
    agents_output: str


def run_preflight(
    *,
    repo: str,
    workdir: Path,
    github: GitHubClient,
    opencode: OpenCodeClient,
    require_agents: bool = True,
) -> PreflightResult:
    """
    Verify binaries, workdir, repo match, and required primary agents.

    Args:
        repo: Expected owner/name slug.
        workdir: Target repository checkout.
        github: GitHub client bound to the same workdir/repo.
        opencode: OpenCode client bound to the same workdir.
        require_agents: When True, fail if issue-worker / pr-reviewer are missing.

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
    git_dir = workdir / ".git"
    if not git_dir.exists():
        raise PreflightError(f"workdir is not a git checkout (missing .git): {workdir}")

    try:
        remote_slug = github.resolve_remote_slug()
    except GitHubError as exc:
        raise PreflightError(f"failed to resolve gh repo for workdir: {exc}") from exc
    if remote_slug.lower() != repo.lower():
        raise PreflightError(
            f"workdir remote {remote_slug!r} does not match --repo {repo!r}"
        )

    agents_output = ""
    if require_agents:
        try:
            agents_output = opencode.list_agents()
        except OpenCodeError as exc:
            raise PreflightError(f"failed to list OpenCode agents: {exc}") from exc
        missing = [name for name in REQUIRED_AGENTS if name not in agents_output]
        if len(missing) > 0:
            raise PreflightError(
                "required OpenCode primary agents missing: "
                + ", ".join(missing)
                + ". Copy templates/target/.opencode into the target repository."
            )

    return PreflightResult(repo=repo, workdir=workdir, agents_output=agents_output)
