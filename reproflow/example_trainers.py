"""Example-only trainer scaffolds for paper-method templates.

These trainers demonstrate when a paper reproduction may need a trainer config
instead of only a model config. They are examples, not required default trainers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from engine import BinaryClassificationTrainer


class AuxiliaryHeadBinaryTrainer(BinaryClassificationTrainer):
    """EXAMPLE ONLY: binary trainer that adds BCE losses for auxiliary heads."""

    def __init__(
        self,
        model,
        cfg,
        data_meta: dict[str, Any] | None = None,
        run_context: dict[str, Any] | None = None,
        aux_loss_weight: float = 0.1,
        aux_output_key: str = "auxiliary_logits",
    ):
        super().__init__(model=model, cfg=cfg, data_meta=data_meta, run_context=run_context)
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_output_key = str(aux_output_key)

    def _run_epoch(self, data_loader, training: bool) -> dict[str, Any]:
        total_loss = 0.0
        num_batches = 0
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        desc = "Training" if training else "Evaluating"

        for batch in tqdm(data_loader, desc=desc):
            batch = self._move_batch_to_device(batch)
            outputs = self.model(batch)
            if "logits" not in outputs:
                raise ValueError("Model forward must return a dict containing 'logits'.")

            logits, labels = self._prepare_logits_and_labels(outputs["logits"], batch["label"])
            loss = self.criterion(logits, labels)

            auxiliary = outputs.get(self.aux_output_key, {})
            if isinstance(auxiliary, dict):
                for aux_logits in auxiliary.values():
                    aux_logits, aux_labels = self._prepare_logits_and_labels(aux_logits, batch["label"])
                    loss = loss + self.aux_loss_weight * self.criterion(aux_logits, aux_labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            all_scores.append(self._scores_from_logits(logits))
            all_labels.append(labels.detach().cpu().numpy())

        if num_batches == 0:
            raise ValueError("DataLoader produced no batches.")

        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return {
            "loss": total_loss / num_batches,
            "metrics": self.calculate_metrics(scores, labels),
        }


class PairwiseRankingExampleTrainer(AuxiliaryHeadBinaryTrainer):
    """EXAMPLE ONLY: placeholder for ranking-style papers.

    In a real reproduction, this trainer would consume positive/negative item
    pairs from a custom Dataset. It inherits the auxiliary-head trainer so the
    config is importable while still making the required extension point clear.
    """

    pass
