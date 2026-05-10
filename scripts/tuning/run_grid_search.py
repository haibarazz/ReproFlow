from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


AGGREGATE_COLUMNS = [
    "candidate_name",
    "n_runs",
    "monitor_metric",
    "monitor_mode",
    "mean_best_metric",
    "std_best_metric",
    "min_best_metric",
    "max_best_metric",
    "best_seed",
    "best_metric",
    "best_epoch",
    "best_run_name",
    "best_run_dir",
]


def _slugify(value: str) -> str:
    cleaned = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "run"


def _format_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        value_text = "true" if value else "false"
    elif value is None:
        value_text = "null"
    else:
        value_text = str(value)
    return f"{key}={value_text}"


def _expand_candidates(spec: dict[str, Any], candidate_key: str = "candidates") -> list[dict[str, Any]]:
    explicit_candidates = spec.get(candidate_key)
    if explicit_candidates is None and candidate_key != "candidates":
        explicit_candidates = spec.get("candidates")

    if explicit_candidates is not None:
        return [
            {
                "name": str(candidate.get("name", f"candidate_{idx:03d}")),
                "overrides": dict(candidate.get("overrides", {})),
            }
            for idx, candidate in enumerate(explicit_candidates, 1)
        ]

    grid = dict(spec.get("grid", {}))
    if not grid:
        return [{"name": "base", "overrides": {}}]

    keys = list(grid)
    candidates = []
    for idx, values in enumerate(itertools.product(*(grid[key] for key in keys)), 1):
        overrides = dict(zip(keys, values))
        name_bits = [f"{_slugify(key)}-{_slugify(value)}" for key, value in overrides.items()]
        candidates.append({"name": f"{idx:03d}_" + "__".join(name_bits), "overrides": overrides})
    return candidates


def _seed_values(spec: dict[str, Any]) -> list[Any | None]:
    if "seeds" in spec:
        seeds = spec.get("seeds") or []
        return list(seeds) if seeds else [None]
    if "seed" in spec:
        return [spec.get("seed")]
    return [None]


def _seed_overrides(spec: dict[str, Any], seed: Any | None) -> dict[str, Any]:
    if seed is None:
        return {}

    override_keys = spec.get("seed_overrides", ["random.seed"])
    overrides = {str(key): seed for key in override_keys}
    if bool(spec.get("seed_controls_data_split", False)):
        overrides.setdefault("data.split.random_state", seed)
    return overrides


def _load_best_metric(history_dir: Path, monitor_metric: str, monitor_mode: str) -> tuple[float | None, int | None]:
    history_files = sorted(history_dir.glob("history_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not history_files:
        return None, None
    history = pd.read_csv(history_files[0])
    if history.empty:
        return None, None

    metric_col = monitor_metric
    if metric_col not in history.columns and metric_col.startswith("val_"):
        metric_col = metric_col
    if metric_col not in history.columns:
        metric_col = "val_loss"
    ascending = monitor_mode == "min"
    sorted_history = history.sort_values(metric_col, ascending=ascending)
    best = sorted_history.iloc[0]
    return float(best[metric_col]), int(best["epoch"])


def _build_command(
    spec: dict[str, Any],
    run_dir: Path,
    overrides: dict[str, Any],
    experiment_id: str,
    run_name: str,
) -> list[str]:
    choices = dict(spec.get("choices", {}))
    base_overrides = dict(spec.get("base_overrides", {}))
    merged_overrides = {**base_overrides, **overrides}

    command = [sys.executable, str(spec.get("main_script", "main.py"))]
    for key in ("data", "model", "trainer", "training_loop", "metrics"):
        if key in choices and choices[key] is not None:
            command.append(_format_override(key, choices[key]))

    command.extend(_format_override(key, value) for key, value in merged_overrides.items())
    command.extend(
        [
            _format_override("experiment.name", experiment_id),
            _format_override("tracking.experiment_id", experiment_id),
            _format_override("tracking.run_id", run_name),
            _format_override("log_path", str(run_dir / "logs")),
            _format_override("best_model_path", str(run_dir / "checkpoint.pth")),
            _format_override("report.output_dir", str(run_dir / "reports")),
            _format_override("hydra.run.dir", str(run_dir / "hydra")),
        ]
    )
    return command


def _write_seed_aggregate(
    rows: list[dict[str, Any]],
    aggregate_path: Path,
    latest_aggregate_path: Path,
) -> None:
    empty = pd.DataFrame(columns=AGGREGATE_COLUMNS)
    if not rows:
        empty.to_csv(aggregate_path, index=False)
        empty.to_csv(latest_aggregate_path, index=False)
        return

    summary = pd.DataFrame(rows)
    if "best_metric" not in summary.columns:
        empty.to_csv(aggregate_path, index=False)
        empty.to_csv(latest_aggregate_path, index=False)
        return

    valid = summary[summary["status"].eq("ok")].copy()
    valid["best_metric"] = pd.to_numeric(valid["best_metric"], errors="coerce")
    valid = valid.dropna(subset=["best_metric"])
    if valid.empty:
        empty.to_csv(aggregate_path, index=False)
        empty.to_csv(latest_aggregate_path, index=False)
        return

    monitor_metric = valid["monitor_metric"].iloc[0]
    monitor_mode = str(valid["monitor_mode"].iloc[0]).lower()
    ascending = monitor_mode == "min"

    aggregate_rows = []
    for candidate_name, group in valid.groupby("candidate_name", sort=False):
        ranked = group.sort_values("best_metric", ascending=ascending)
        best = ranked.iloc[0]
        aggregate_rows.append(
            {
                "candidate_name": candidate_name,
                "n_runs": int(group.shape[0]),
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
                "mean_best_metric": float(group["best_metric"].mean()),
                "std_best_metric": float(group["best_metric"].std(ddof=0)),
                "min_best_metric": float(group["best_metric"].min()),
                "max_best_metric": float(group["best_metric"].max()),
                "best_seed": best.get("seed"),
                "best_metric": float(best["best_metric"]),
                "best_epoch": best.get("best_epoch"),
                "best_run_name": best.get("run_name"),
                "best_run_dir": best.get("run_dir"),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows).sort_values("mean_best_metric", ascending=ascending)
    aggregate.to_csv(aggregate_path, index=False)
    aggregate.to_csv(latest_aggregate_path, index=False)


def run_spec(
    spec_path: Path,
    dry_run: bool = False,
    max_runs: int | None = None,
    *,
    name_key: str = "tuning_name",
    candidate_key: str = "candidates",
    default_output_root: str = "result/tuning",
    run_label: str = "tuning",
) -> Path:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec_name = str(spec.get(name_key, spec.get("tuning_name", spec_path.stem)))
    experiment_id = _slugify(str(spec.get("tracking_experiment_id", spec_name)))
    output_root = Path(str(spec.get("output_root", default_output_root)))
    spec_root = output_root / spec_name
    runs_root = spec_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = spec_root / f"summary_{started_at}.csv"
    latest_summary_path = spec_root / "summary_latest.csv"
    aggregate_path = spec_root / f"summary_seed_aggregate_{started_at}.csv"
    latest_aggregate_path = spec_root / "summary_seed_aggregate_latest.csv"
    snapshot_path = spec_root / f"spec_snapshot_{started_at}.yaml"
    snapshot_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    monitor_metric = str(spec.get("monitor_metric", "val_loss"))
    monitor_mode = str(spec.get("monitor_mode", "min")).lower()
    candidates = _expand_candidates(spec, candidate_key=candidate_key)
    seeds = _seed_values(spec)
    planned_runs = [
        {
            "candidate_index": candidate_idx,
            "candidate": candidate,
            "seed": seed,
        }
        for candidate_idx, candidate in enumerate(candidates, 1)
        for seed in seeds
    ]
    if max_runs is not None:
        planned_runs = planned_runs[:max_runs]

    rows = []
    for idx, planned in enumerate(planned_runs, 1):
        candidate = planned["candidate"]
        candidate_name = _slugify(candidate["name"])
        seed = planned["seed"]
        seed_suffix = f"_seed{_slugify(seed)}" if seed is not None else ""
        run_name = f"{idx:03d}_{candidate_name}{seed_suffix}"
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_overrides = {**candidate["overrides"], **_seed_overrides(spec, seed)}
        command = _build_command(spec, run_dir, run_overrides, experiment_id, run_name)
        print("\n" + "=" * 80)
        print(f"{run_label.title()} run {idx}/{len(planned_runs)}: {run_name}")
        print(" ".join(command))

        row = {
            "run_name": run_name,
            "candidate_name": candidate_name,
            "candidate_index": planned["candidate_index"],
            "seed": seed,
            "status": "dry_run" if dry_run else "pending",
            "returncode": None,
            "monitor_metric": monitor_metric,
            "monitor_mode": monitor_mode,
            "best_metric": None,
            "best_epoch": None,
            "run_dir": str(run_dir),
        }
        for key, value in run_overrides.items():
            row[f"param::{key}"] = value

        if not dry_run:
            completed = subprocess.run(command, cwd=Path.cwd(), text=True)
            row["returncode"] = completed.returncode
            row["status"] = "ok" if completed.returncode == 0 else "failed"
            if completed.returncode == 0:
                best_metric, best_epoch = _load_best_metric(run_dir / "logs", monitor_metric, monitor_mode)
                row["best_metric"] = best_metric
                row["best_epoch"] = best_epoch

        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        pd.DataFrame(rows).to_csv(latest_summary_path, index=False)
        _write_seed_aggregate(rows, aggregate_path, latest_aggregate_path)

    print(f"\nSummary: {summary_path}")
    print(f"Latest: {latest_summary_path}")
    print(f"Seed aggregate: {latest_aggregate_path}")
    return latest_summary_path


def run_tuning(spec_path: Path, dry_run: bool = False, max_runs: int | None = None) -> Path:
    return run_spec(
        spec_path,
        dry_run=dry_run,
        max_runs=max_runs,
        name_key="tuning_name",
        candidate_key="candidates",
        default_output_root="result/tuning",
        run_label="tuning",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReproFlow grid/candidate tuning specs.")
    parser.add_argument("spec", type=Path, help="Path to configs/tuning/*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write summary without running training.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of candidates for smoke tests.")
    args = parser.parse_args()
    run_tuning(args.spec, dry_run=args.dry_run, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
