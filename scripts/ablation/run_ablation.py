from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tuning.run_grid_search import run_spec


def run_ablation(spec_path: Path, dry_run: bool = False, max_runs: int | None = None) -> Path:
    return run_spec(
        spec_path,
        dry_run=dry_run,
        max_runs=max_runs,
        name_key="ablation_name",
        candidate_key="variants",
        default_output_root="result/ablation",
        run_label="ablation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReproFlow ablation specs.")
    parser.add_argument("spec", type=Path, help="Path to configs/ablation/*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write summary without running training.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit variant-seed runs for smoke tests.")
    args = parser.parse_args()
    run_ablation(args.spec, dry_run=args.dry_run, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
