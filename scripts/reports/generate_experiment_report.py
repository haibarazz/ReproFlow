from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import EmptyDataError


def _discover_summaries() -> list[Path]:
    patterns = [
        "result/tuning/*/summary_latest.csv",
        "result/tuning/*/summary_seed_aggregate_latest.csv",
        "result/ablation/*/summary_latest.csv",
        "result/ablation/*/summary_seed_aggregate_latest.csv",
        "result/experiments/*/summary_latest.csv",
        "result/experiments/*/summary_seed_aggregate_latest.csv",
        "ml_results/*/benchmark_latest.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path.cwd().glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"

    frame = frame.head(max_rows).copy()
    frame = frame.fillna("")
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _select_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aggregate_cols = [
        "candidate_name",
        "n_runs",
        "monitor_metric",
        "monitor_mode",
        "mean_best_metric",
        "std_best_metric",
        "best_seed",
        "best_metric",
        "best_run_name",
    ]
    run_cols = [
        "run_name",
        "candidate_name",
        "seed",
        "status",
        "monitor_metric",
        "monitor_mode",
        "best_metric",
        "best_epoch",
        "run_dir",
    ]
    benchmark_cols = [
        "model_name",
        "model",
        "task_type",
        "accuracy",
        "f1",
        "auc",
        "auc_pr",
        "rmse",
        "mae",
        "r2",
        "status",
    ]

    for preferred in (aggregate_cols, run_cols, benchmark_cols):
        available = [column for column in preferred if column in frame.columns]
        if len(available) >= 2:
            return frame[available]
    return frame.iloc[:, : min(10, len(frame.columns))]


def generate_report(summary_paths: Iterable[Path], output: Path, title: str) -> Path:
    summary_paths = list(summary_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Summary files: {len(summary_paths)}",
        "",
    ]

    for summary_path in summary_paths:
        if not summary_path.exists():
            lines.extend([f"## {summary_path}", "", "_File not found._", ""])
            continue

        try:
            frame = pd.read_csv(summary_path)
        except EmptyDataError:
            frame = pd.DataFrame()
        display = _select_columns(frame)
        lines.extend(
            [
                f"## {summary_path}",
                "",
                f"- Rows: {len(frame)}",
                "",
                _markdown_table(display),
                "",
            ]
        )

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a markdown report from ReproFlow summaries.")
    parser.add_argument("--summary", action="append", type=Path, dest="summaries", help="Summary CSV path. Can be passed more than once.")
    parser.add_argument("--output", type=Path, default=Path("result/reports/aggregate_report.md"))
    parser.add_argument("--title", default="ReproFlow Aggregate Experiment Report")
    args = parser.parse_args()

    summary_paths = args.summaries if args.summaries else _discover_summaries()
    output = generate_report(summary_paths, args.output, args.title)
    print(f"Aggregate report: {output}")


if __name__ == "__main__":
    main()
