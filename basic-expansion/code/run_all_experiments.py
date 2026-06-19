"""Run all basic-expansion experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SCRIPTS = [
    ROOT / "2.1-structure-anova" / "code" / "run_anova.py",
    ROOT / "2.2-dynamic-panel" / "code" / "run_dynamic_panel.py",
    ROOT / "2.3-clustering" / "code" / "run_kmeans.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nAll basic-expansion experiments finished.", flush=True)


if __name__ == "__main__":
    main()
