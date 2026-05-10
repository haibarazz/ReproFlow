from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from Data_pre import load_and_preprocess_data, prepare_data
from reproflow.seed_utils import set_global_seed
from reproflow.tracking import create_run_context, write_run_metadata

os.environ["HYDRA_FULL_ERROR"] = "1"


def create_model(cfg: DictConfig, input_dim: int, data_meta: dict[str, Any]):
    task_type = data_meta["task_type"]
    num_classes = int(data_meta.get("num_classes", 1))
    return hydra.utils.instantiate(
        cfg.model,
        input_dim=input_dim,
        task_type=task_type,
        num_classes=num_classes,
    )


def create_trainer(cfg: DictConfig, model, data_meta: dict[str, Any], run_context: dict[str, Any] | None = None):
    return hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        cfg=cfg,
        data_meta=data_meta,
        run_context=run_context,
        _recursive_=False,
    )


def load_training(
    checkpoint_path: str,
    cfg: DictConfig,
    train_loader,
    test_loader,
    input_dim: int,
    data_meta: dict[str, Any],
):
    print(f"Loading checkpoint: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = create_model(cfg, input_dim=input_dim, data_meta=data_meta)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    run_context = create_run_context(cfg, run_type="resume")
    write_run_metadata(run_context, cfg, data_meta, status="resuming")
    trainer = create_trainer(cfg, model, data_meta, run_context)
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    print(f"Resume from epoch {start_epoch + 1}")
    trainer.train_from_epoch(train_loader, test_loader, start_epoch=start_epoch)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("Starting ReproFlow experiment...")
    seed = set_global_seed(cfg.get("random", {}))
    print(f"Random seed: {seed}")
    df = load_and_preprocess_data(cfg)
    train_loader, test_loader, input_dim, data_meta = prepare_data(df, cfg)
    data_meta["seed"] = seed
    run_context = create_run_context(cfg, run_type="train")
    write_run_metadata(run_context, cfg, data_meta, status="started")

    try:
        model = create_model(cfg, input_dim=input_dim, data_meta=data_meta)
        trainer = create_trainer(cfg, model, data_meta, run_context)
        trainer.train(train_loader, test_loader)
    except Exception as exc:
        write_run_metadata(
            run_context,
            cfg,
            data_meta,
            status="failed",
            extra={"error": repr(exc)},
        )
        raise


if __name__ == "__main__":
    main()
