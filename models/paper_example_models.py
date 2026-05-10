"""Example-only paper model scaffolds for ReproFlow.

These classes are intentionally lightweight examples. They show how an AI agent
can map classic papers into ReproFlow model/config files without turning the
repository into a full paper-code zoo. They are not faithful, benchmark-ready
implementations and should not be treated as reproduced results.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_multiclass(task_type: str) -> bool:
    return str(task_type) in {"multiclass", "multi_class", "multiclass_classification"}


def _is_regression(task_type: str) -> bool:
    return str(task_type) == "regression"


def _output_dim(task_type: str, num_classes: int) -> int:
    return int(num_classes) if _is_multiclass(task_type) else 1


def _format_logits(logits: torch.Tensor, task_type: str) -> torch.Tensor:
    if not _is_multiclass(task_type) and logits.dim() == 2 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    return nn.GELU()


class ExampleMLP(nn.Module):
    """EXAMPLE ONLY: small reusable MLP block for the paper scaffolds."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    _activation(activation),
                    nn.Dropout(float(dropout)),
                ]
            )
            last_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.output_dim = last_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseFeatureTokenizer(nn.Module):
    """EXAMPLE ONLY: converts a flat feature vector into feature tokens."""

    def __init__(self, input_dim: int, num_fields: int = 16, embed_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_fields = int(max(1, num_fields))
        self.value_proj = nn.Linear(1, int(embed_dim))
        self.field_bias = nn.Parameter(torch.zeros(self.num_fields, int(embed_dim)))
        nn.init.normal_(self.field_bias, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.reshape(x.size(0), -1)
        if x.size(1) < self.num_fields:
            pad = x.new_zeros(x.size(0), self.num_fields - x.size(1))
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, : self.num_fields]
        return self.value_proj(x.unsqueeze(-1)) + self.field_bias.unsqueeze(0)


class TransformerEncoderExample(nn.Module):
    """EXAMPLE ONLY: inspired by arXiv:1706.03762, adapted for tabular tokens."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        embed_dim: int = 64,
        num_fields: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_transformer_encoder"
        self.task_type = task_type
        self.tokenizer = DenseFeatureTokenizer(input_dim, num_fields=num_fields, embed_dim=embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, _output_dim(task_type, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(batch["basic_features"])
        pooled = self.norm(self.encoder(tokens).mean(dim=1))
        logits = self.head(pooled)
        return {"logits": _format_logits(logits, self.task_type)}


class BertClassifierExample(nn.Module):
    """EXAMPLE ONLY: BERT-style CLS pooling with an optional auxiliary head."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        embed_dim: int = 64,
        num_fields: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_auxiliary_head: bool = True,
        **_: object,
    ):
        super().__init__()
        self.name = "example_bert_classifier"
        self.task_type = task_type
        self.use_auxiliary_head = bool(use_auxiliary_head)
        self.tokenizer = DenseFeatureTokenizer(input_dim, num_fields=num_fields, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, num_fields + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, _output_dim(task_type, num_classes))
        self.aux_head = nn.Linear(embed_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(batch["basic_features"])
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        hidden = torch.cat([cls, tokens], dim=1) + self.position[:, : tokens.size(1) + 1]
        pooled = self.norm(self.encoder(hidden)[:, 0])
        logits = self.head(pooled)
        output = {"logits": _format_logits(logits, self.task_type)}
        if self.use_auxiliary_head:
            output["auxiliary_logits"] = {"example_aux_head": self.aux_head(pooled).squeeze(-1)}
        return output


class ResidualBlock(nn.Module):
    """EXAMPLE ONLY: residual MLP block used by ResNetMLPExample."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ResNetMLPExample(nn.Module):
    """EXAMPLE ONLY: ResNet idea from arXiv:1512.03385 adapted to tabular data."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_resnet_mlp"
        self.task_type = task_type
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, _output_dim(task_type, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.stem(batch["basic_features"]))
        logits = self.head(hidden)
        return {"logits": _format_logits(logits, self.task_type)}


class WideDeepExample(nn.Module):
    """EXAMPLE ONLY: Wide & Deep style memorization plus generalization branch."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_wide_deep"
        self.task_type = task_type
        hidden_dims = hidden_dims or [128, 64]
        out_dim = _output_dim(task_type, num_classes)
        self.wide = nn.Linear(input_dim, out_dim)
        self.deep = ExampleMLP(input_dim, hidden_dims, dropout)
        self.deep_head = nn.Linear(self.deep.output_dim, out_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["basic_features"]
        logits = self.wide(x) + self.deep_head(self.deep(x))
        return {"logits": _format_logits(logits, self.task_type)}


class NCFExample(nn.Module):
    """EXAMPLE ONLY: NCF-style GMF plus MLP interaction from split feature halves."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_ncf"
        self.task_type = task_type
        hidden_dims = hidden_dims or [128, 64]
        left_dim = max(1, input_dim // 2)
        right_dim = max(1, input_dim - left_dim)
        self.left_dim = left_dim
        self.user_proj = nn.Linear(left_dim, embed_dim)
        self.item_proj = nn.Linear(right_dim, embed_dim)
        self.mlp = ExampleMLP(embed_dim * 2, hidden_dims, dropout)
        self.head = nn.Linear(embed_dim + self.mlp.output_dim, _output_dim(task_type, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["basic_features"]
        user_x = x[:, : self.left_dim]
        item_x = x[:, self.left_dim :]
        if item_x.size(1) == 0:
            item_x = x[:, -1:].clone()
        user = F.gelu(self.user_proj(user_x))
        item = F.gelu(self.item_proj(item_x))
        gmf = user * item
        mlp_out = self.mlp(torch.cat([user, item], dim=-1))
        logits = self.head(torch.cat([gmf, mlp_out], dim=-1))
        return {"logits": _format_logits(logits, self.task_type)}


class DeepFMExample(nn.Module):
    """EXAMPLE ONLY: DeepFM-style wide, FM interaction, and deep branches."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        embed_dim: int = 32,
        num_fields: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_deepfm"
        self.task_type = task_type
        hidden_dims = hidden_dims or [128, 64]
        out_dim = _output_dim(task_type, num_classes)
        self.tokenizer = DenseFeatureTokenizer(input_dim, num_fields=num_fields, embed_dim=embed_dim)
        self.wide = nn.Linear(input_dim, out_dim)
        self.deep = ExampleMLP(num_fields * embed_dim, hidden_dims, dropout)
        self.head = nn.Linear(out_dim + embed_dim + self.deep.output_dim, out_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["basic_features"]
        tokens = self.tokenizer(x)
        sum_square = tokens.sum(dim=1).pow(2)
        square_sum = tokens.pow(2).sum(dim=1)
        fm = 0.5 * (sum_square - square_sum)
        deep = self.deep(tokens.flatten(start_dim=1))
        logits = self.head(torch.cat([self.wide(x), fm, deep], dim=-1))
        return {"logits": _format_logits(logits, self.task_type)}


class CrossLayer(nn.Module):
    """EXAMPLE ONLY: DCN cross layer."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cross = torch.sum(x * self.weight, dim=-1, keepdim=True)
        return x0 * cross + self.bias + x


class DCNExample(nn.Module):
    """EXAMPLE ONLY: Deep & Cross Network style explicit feature crossing."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        cross_layers: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_dcn"
        self.task_type = task_type
        hidden_dims = hidden_dims or [128, 64]
        self.cross = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])
        self.deep = ExampleMLP(input_dim, hidden_dims, dropout)
        self.head = nn.Linear(input_dim + self.deep.output_dim, _output_dim(task_type, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x0 = batch["basic_features"]
        x_cross = x0
        for layer in self.cross:
            x_cross = layer(x0, x_cross)
        deep = self.deep(x0)
        logits = self.head(torch.cat([x_cross, deep], dim=-1))
        return {"logits": _format_logits(logits, self.task_type)}


class AutoIntExample(nn.Module):
    """EXAMPLE ONLY: AutoInt-style self-attentive feature interaction model."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        embed_dim: int = 32,
        num_fields: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_autoint"
        self.task_type = task_type
        self.tokenizer = DenseFeatureTokenizer(input_dim, num_fields=num_fields, embed_dim=embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(num_fields * embed_dim, _output_dim(task_type, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        interactions = self.attention(self.tokenizer(batch["basic_features"]))
        logits = self.head(interactions.flatten(start_dim=1))
        return {"logits": _format_logits(logits, self.task_type)}


class DINExample(nn.Module):
    """EXAMPLE ONLY: DIN-style target-aware attention over behavior tokens."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        embed_dim: int = 32,
        num_fields: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_auxiliary_head: bool = True,
        **_: object,
    ):
        super().__init__()
        self.name = "example_din"
        self.task_type = task_type
        self.use_auxiliary_head = bool(use_auxiliary_head)
        hidden_dims = hidden_dims or [128, 64]
        self.tokenizer = DenseFeatureTokenizer(input_dim, num_fields=num_fields, embed_dim=embed_dim)
        self.att_mlp = ExampleMLP(embed_dim * 4, [64, 32], dropout)
        self.att_head = nn.Linear(self.att_mlp.output_dim, 1)
        self.mlp = ExampleMLP(embed_dim * 3, hidden_dims, dropout)
        self.head = nn.Linear(self.mlp.output_dim, _output_dim(task_type, num_classes))
        self.aux_head = nn.Linear(embed_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(batch["basic_features"])
        target = tokens[:, -1]
        history = tokens[:, :-1]
        target_expand = target.unsqueeze(1).expand_as(history)
        att_input = torch.cat([history, target_expand, history - target_expand, history * target_expand], dim=-1)
        att_scores = self.att_head(self.att_mlp(att_input.flatten(0, 1))).view(tokens.size(0), -1)
        weights = torch.softmax(att_scores, dim=-1)
        interest = torch.sum(history * weights.unsqueeze(-1), dim=1)
        hidden = self.mlp(torch.cat([interest, target, interest * target], dim=-1))
        logits = self.head(hidden)
        output = {"logits": _format_logits(logits, self.task_type)}
        if self.use_auxiliary_head:
            output["auxiliary_logits"] = {"example_interest_aux": self.aux_head(interest).squeeze(-1)}
        return output


class DLRMExample(nn.Module):
    """EXAMPLE ONLY: DLRM-style dense bottom MLP plus feature interactions."""

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int = 1,
        embed_dim: int = 32,
        num_fields: int = 16,
        bottom_dims: list[int] | None = None,
        top_dims: list[int] | None = None,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.name = "example_dlrm"
        self.task_type = task_type
        bottom_dims = bottom_dims or [128, embed_dim]
        top_dims = top_dims or [128, 64]
        self.tokenizer = DenseFeatureTokenizer(input_dim, num_fields=num_fields, embed_dim=embed_dim)
        self.bottom = ExampleMLP(input_dim, bottom_dims, dropout)
        interaction_dim = (num_fields + 1) * num_fields // 2
        self.top = ExampleMLP(self.bottom.output_dim + interaction_dim, top_dims, dropout)
        self.head = nn.Linear(self.top.output_dim, _output_dim(task_type, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["basic_features"]
        dense = self.bottom(x)
        tokens = self.tokenizer(x)
        dense_token = dense.unsqueeze(1)
        features = torch.cat([dense_token, tokens], dim=1)
        interactions = torch.bmm(features, features.transpose(1, 2))
        row, col = torch.triu_indices(interactions.size(1), interactions.size(2), offset=1, device=interactions.device)
        pairwise = interactions[:, row, col]
        logits = self.head(self.top(torch.cat([dense, pairwise], dim=-1)))
        return {"logits": _format_logits(logits, self.task_type)}
