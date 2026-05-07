from __future__ import annotations

from pathlib import Path
from typing import Iterable


def print_header(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def ensure_exists(paths: Iterable[str]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files or directories:\n" + "\n".join(missing)
        )
