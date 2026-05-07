from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    run([sys.executable, "src/bootstrap_demo_data.py"])
    run([sys.executable, "src/check_project_state.py"])
    run([sys.executable, "src/smoke_test_step2.py"])


if __name__ == "__main__":
    main()
