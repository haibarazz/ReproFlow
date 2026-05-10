from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _slugify(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "run"


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for part in path.split("."):
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(part, None)
        else:
            current = getattr(current, part, None)
    return default if current is None else current


def create_run_context(cfg: Any, run_type: str = "train") -> dict[str, Any]:
    tracking_cfg = _cfg_get(cfg, "tracking", {})
    experiment_cfg = _cfg_get(cfg, "experiment", {})
    enabled = bool(tracking_cfg.get("enabled", True)) if hasattr(tracking_cfg, "get") else True

    experiment_name = str(experiment_cfg.get("name", "reproflow_experiment")) if hasattr(experiment_cfg, "get") else "reproflow_experiment"
    data_name = str(_cfg_get(cfg, "data.name", "dataset"))
    model_target = str(_cfg_get(cfg, "model._target_", "model")).split(".")[-1]
    trainer_target = str(_cfg_get(cfg, "trainer._target_", "trainer")).split(".")[-1]
    seed = _cfg_get(cfg, "random.seed", None)

    explicit_experiment_id = tracking_cfg.get("experiment_id") if hasattr(tracking_cfg, "get") else None
    explicit_run_id = tracking_cfg.get("run_id") if hasattr(tracking_cfg, "get") else None
    experiment_id = str(explicit_experiment_id or _slugify(experiment_name))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(
        explicit_run_id
        or f"{timestamp}_{_slugify(data_name)}_{_slugify(model_target)}_seed{seed}_{uuid.uuid4().hex[:8]}"
    )

    output_dir = Path(str(tracking_cfg.get("output_dir", "result/tracking"))) if hasattr(tracking_cfg, "get") else Path("result/tracking")
    run_dir = output_dir / experiment_id / run_id
    if enabled:
        run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "enabled": enabled,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "run_type": run_type,
        "run_dir": str(run_dir),
        "data_name": data_name,
        "model_target": model_target,
        "trainer_target": trainer_target,
        "seed": seed,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_run_metadata(
    run_context: dict[str, Any],
    cfg: Any,
    data_meta: dict[str, Any],
    status: str,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    if not run_context.get("enabled", True):
        return None
    run_dir = Path(str(run_context["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **run_context,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "data_meta": {
            key: value
            for key, value in data_meta.items()
            if key != "preprocessors"
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if extra:
        payload.update(extra)
    path = run_dir / "run_metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
