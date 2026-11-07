"""Allow `python -m super_orchestrator`."""

from __future__ import annotations

from super_orchestrator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
