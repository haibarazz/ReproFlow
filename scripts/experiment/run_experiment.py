from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tuning.run_grid_search import _format_override, _load_best_metric, _slugify, _write_seed_aggregate


def _seed_values(spec: dict[str, Any]) -> list[Any | None]:
    seeds = spec.get("seeds")
    if seeds:
        return list(seeds)
    if "seed" in spec:
        return [spec.get("seed")]
    return [None]


def _seed_overrides(spec: dict[str, Any], seed: Any | None) -> dict[str, Any]:
    if seed is None:
        return {}
    overrides = {str(key): seed for key in spec.get("seed_overrides", ["random.seed"])}
    if bool(spec.get("seed_controls_data_split", False)):
        overrides.setdefault("data.split.random_state", seed)
    return overrides


def _build_deep_command(
    spec: dict[str, Any],
    method: dict[str, Any],
    seed: Any | None,
    run_dir: Path,
    run_name: str,
) -> list[str]:
    experiment_name = _slugify(str(spec.get("experiment_name", "experiment")))
    data = method.get("data", spec.get("data"))
    model = method.get("model")
    trainer = method.get("trainer", spec.get("trainer"))
    training_loop = method.get("training_loop", spec.get("training_loop", "default"))
    metrics = method.get("metrics", spec.get("metrics"))
    if not model:
        raise ValueError(f"Deep learning method {method.get('name')} must define model.")

    command = [sys.executable, str(spec.get("main_script", "main.py"))]
    for key, value in (
        ("data", data),
        ("model", model),
        ("trainer", trainer),
        ("training_loop", training_loop),
        ("metrics", metrics),
    ):
        if value is not None:
            command.append(_format_override(key, value))

    overrides = {
        **dict(spec.get("base_overrides", {})),
        **dict(method.get("overrides", {})),
        **_seed_overrides(spec, seed),
    }
    command.extend(_format_override(key, value) for key, value in overrides.items())
    command.extend(
        [
            _format_override("experiment.name", experiment_name),
            _format_override("tracking.experiment_id", experiment_name),
            _format_override("tracking.run_id", run_name),
            _format_override("log_path", str(run_dir / "logs")),
            _format_override("best_model_path", str(run_dir / "checkpoint.pth")),
            _format_override("report.output_dir", str(run_dir / "reports")),
            _format_override("hydra.run.dir", str(run_dir / "hydra")),
        ]
    )
    return command


def _build_ml_command(
    spec: dict[str, Any],
    method: dict[str, Any],
    seed: Any | None,
    run_dir: Path,
) -> list[str]:
    data = method.get("data", spec.get("data"))
    save_dir = run_dir / "ml_benchmark"
    primary_metric = method.get("primary_metric", spec.get("benchmark_metric", "roc_auc"))
    command = [
        sys.executable,
        str(spec.get("ml_script", "run_ml_benchmark.py")),
        _format_override("data", data),
        _format_override("output.save_dir", str(save_dir)),
        _format_override("output.primary_metric", primary_metric),
    ]
    if seed is not None and bool(spec.get("seed_controls_data_split", False)):
        command.append(_format_override("data.split.random_state", seed))
    for key, value in dict(method.get("overrides", {})).items():
        command.append(_format_override(key, value))
    return command


def _append_ml_rows(
    rows: list[dict[str, Any]],
    method_name: str,
    seed: Any | None,
    run_name: str,
    run_dir: Path,
    status: str,
    returncode: int | None,
    metric_name: str,
) -> None:
    result_path = run_dir / "ml_benchmark" / "benchmark_latest.csv"
    if status != "ok" or not result_path.exists():
        rows.append(
            {
                "run_name": run_name,
                "candidate_name": method_name,
                "method_type": "ml_benchmark",
                "seed": seed,
                "status": status,
                "returncode": returncode,
                "monitor_metric": metric_name,
                "monitor_mode": "max",
                "best_metric": None,
                "best_epoch": None,
                "run_dir": str(run_dir),
            }
        )
        return

    frame = pd.read_csv(result_path)
    for _, result in frame.iterrows():
        model_name = str(result.get("model_name", "ml_model"))
        metric_value = result.get(metric_name)
        rows.append(
            {
                "run_name": f"{run_name}_{_slugify(model_name)}",
                "candidate_name": f"{method_name}::{model_name}",
                "method_type": "ml_benchmark",
                "seed": seed,
                "status": result.get("status", status),
                "returncode": returncode,
                "monitor_metric": metric_name,
                "monitor_mode": "max",
                "best_metric": metric_value,
                "best_epoch": None,
                "run_dir": str(run_dir),
            }
        )


def run_experiment(spec_path: Path, dry_run: bool = False, max_runs: int | None = None) -> Path:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    experiment_name = _slugify(str(spec.get("experiment_name", spec_path.stem)))
    output_root = Path(str(spec.get("output_root", "result/experiments")))
    experiment_root = output_root / experiment_name
    runs_root = experiment_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = experiment_root / f"summary_{started_at}.csv"
    latest_summary_path = experiment_root / "summary_latest.csv"
    aggregate_path = experiment_root / f"summary_seed_aggregate_{started_at}.csv"
    latest_aggregate_path = experiment_root / "summary_seed_aggregate_latest.csv"
    snapshot_path = experiment_root / f"experiment_snapshot_{started_at}.yaml"
    snapshot_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    seeds = _seed_values(spec)
    methods = list(spec.get("methods", []))
    planned_runs = [
        {"method_index": method_idx, "method": method, "seed": seed}
        for method_idx, method in enumerate(methods, 1)
        for seed in seeds
    ]
    if max_runs is not None:
        planned_runs = planned_runs[:max_runs]

    rows: list[dict[str, Any]] = []
    for idx, planned in enumerate(planned_runs, 1):
        method = planned["method"]
        method_name = _slugify(str(method.get("name", f"method_{planned['method_index']:03d}")))
        method_type = str(method.get("type", "deep_learning"))
        seed = planned["seed"]
        seed_suffix = f"_seed{_slugify(seed)}" if seed is not None else ""
        run_name = f"{idx:03d}_{method_name}{seed_suffix}"
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if method_type == "ml_benchmark":
            command = _build_ml_command(spec, method, seed, run_dir)
        else:
            command = _build_deep_command(spec, method, seed, run_dir, run_name)

        print("\n" + "=" * 80)
        print(f"Experiment run {idx}/{len(planned_runs)}: {run_name}")
        print(" ".join(command))

        status = "dry_run" if dry_run else "pending"
        returncode = None
        if not dry_run:
            completed = subprocess.run(command, cwd=Path.cwd(), text=True)
            returncode = completed.returncode
            status = "ok" if completed.returncode == 0 else "failed"

        if method_type == "ml_benchmark":
            _append_ml_rows(
                rows,
                method_name,
                seed,
                run_name,
                run_dir,
                status,
                returncode,
                str(method.get("primary_metric", spec.get("benchmark_metric", "roc_auc"))),
            )
        else:
            best_metric = None
            best_epoch = None
            if status == "ok":
                best_metric, best_epoch = _load_best_metric(
                    run_dir / "logs",
                    str(spec.get("monitor_metric", "val_loss")),
                    str(spec.get("monitor_mode", "min")).lower(),
                )
            row = {
                "run_name": run_name,
                "candidate_name": method_name,
                "method_type": method_type,
                "seed": seed,
                "status": status,
                "returncode": returncode,
                "monitor_metric": str(spec.get("monitor_metric", "val_loss")),
                "monitor_mode": str(spec.get("monitor_mode", "min")).lower(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "run_dir": str(run_dir),
            }
            for key, value in dict(method.get("overrides", {})).items():
                row[f"param::{key}"] = value
            rows.append(row)

        pd.DataFrame(rows).to_csv(summary_path, index=False)
        pd.DataFrame(rows).to_csv(latest_summary_path, index=False)
        _write_seed_aggregate(rows, aggregate_path, latest_aggregate_path)

    print(f"\nSummary: {summary_path}")
    print(f"Latest: {latest_summary_path}")
    print(f"Seed aggregate: {latest_aggregate_path}")
    return latest_summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a first-class ReproFlow experiment manifest.")
    parser.add_argument("spec", type=Path, help="Path to configs/experiment/*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write summary without running.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit method-seed runs for smoke tests.")
    args = parser.parse_args()
    run_experiment(args.spec, dry_run=args.dry_run, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
