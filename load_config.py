"""Load YAML configs with ${VAR:-default} environment substitution."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _substitute(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _substitute(raw or {})


def pilot_example_counts(data: dict[str, Any]) -> tuple[int, int]:
    """Resolve (nq, hotpot) example counts from config ``data`` section.

    If either ``pilot_nq`` or ``pilot_hotpot`` is present, those keys are used
    (defaulting the sibling to 100). Otherwise ``main_nq`` / ``main_hotpot``
    apply (for ``configs/main_experiment.yaml``). If none are present,
    returns ``(100, 100)``.
    """
    if "pilot_nq" in data or "pilot_hotpot" in data:
        return int(data.get("pilot_nq", 100)), int(data.get("pilot_hotpot", 100))
    if "main_nq" in data or "main_hotpot" in data:
        return int(data.get("main_nq", 100)), int(data.get("main_hotpot", 100))
    return 100, 100
