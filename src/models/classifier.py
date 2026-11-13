"""Phase 4 multi-label transformer detector architectures, pooling, chunk aggregators, and loss functions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.data.schema import HALLUCINATION_LABELS

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - local CPU inspection environments without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    AutoConfig = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False


def torch_available() -> bool:
    """Return True when PyTorch is importable in the active runtime."""
    return _TORCH_AVAILABLE


def transformers_available() -> bool:
    """Return True when HuggingFace transformers is importable in the active runtime."""
    return _TRANSFORMERS_AVAILABLE


def require_torch_and_transformers(caller: str) -> None:
    """Raise a clear actionable error when PyTorch/Transformers are absent (e.g. local non-GPU venv)."""
    if not _TORCH_AVAILABLE or not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            f"{caller} requires 'torch' and 'transformers'. On CPU-only local Windows, use "
            "src.models.baseline (TF-IDF) / inspect code & run tests, or execute GPU training in "
            "Google Colab via `python scripts/colab_workflow.py`."
        )


def resolve_backbone_spec(model_cfg: dict[str, Any], backbone_key: str) -> dict[str, Any]:
    """Resolve verified backbone metadata from configs/model.yaml."""
    backbones = model_cfg.get("backbones") or {}
    if backbone_key in backbones:
        return dict(backbones[backbone_key])
    for key, spec in backbones.items():
        if spec.get("hf_model_id") == backbone_key or spec.get("experiment_id") == backbone_key:
            out = dict(spec)
            out["backbone_key"] = key
            return out
    raise KeyError(
        f"Unknown backbone {backbone_key!r}; expected one of {sorted(backbones.keys())}."
    )


def compute_pos_weights_from_matrix(
    y_train: np.ndarray,
    *,
    cap: float = 5.0,
) -> np.ndarray:
    """Compute clipped negative/positive ratio pos_weight vector for imbalanced BCE (Suite A6)."""
    yt = np.asarray(y_train, dtype=np.float64)
    if yt.ndim != 2 or yt.shape[1] != len(HALLUCINATION_LABELS):
        raise ValueError(f"Expected shape (N, 5), got {yt.shape}")
    pos = yt.sum(axis=0)
    neg = np.maximum(0.0, yt.shape[0] - pos)
    raw = np.where(pos > 0.0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(raw, 1.0, float(cap)).astype(np.float32)


def aggregate_chunk_logits_numpy(
    chunk_logits: np.ndarray,
    mode: str = "logsumexp",
) -> np.ndarray:
    """Aggregate `(M, 5)` sliding-window chunk logits into a single `(5,)` vector."""
    arr = np.asarray(chunk_logits, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(HALLUCINATION_LABELS):
        raise ValueError(f"Expected (M, 5) chunk_logits, got {arr.shape}")
    if len(arr) == 0:
        return np.zeros(len(HALLUCINATION_LABELS), dtype=np.float64)
    if mode == "max":
        return arr.max(axis=0)
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "logsumexp":
        m = arr.max(axis=0)
        return m + np.log(np.sum(np.exp(arr - m[None, :]), axis=0)) - np.log(float(len(arr)))
    raise ValueError(f"Unsupported chunk aggregation mode={mode!r}")


def asymmetric_focal_loss_numpy(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
) -> float:
    """Reference NumPy Asymmetric Focal Loss (Ridnik et al. / Suite A6) for parity testing."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    xs_pos = np.clip(p, 1e-8, 1.0)
    xs_neg = np.clip((1.0 - p) + float(clip), 1e-8, 1.0)
    los_pos = y * np.log(xs_pos) * ((1.0 - xs_pos) ** float(gamma_pos))
    los_neg = (1.0 - y) * np.log(xs_neg) * ((1.0 - xs_neg) ** float(gamma_neg))
    return float(-np.mean(los_pos + los_neg))


if _TORCH_AVAILABLE:

    class AsymmetricFocalLoss(nn.Module):  # type: ignore[misc]
        """Multi-label Asymmetric Loss (ASL) mitigating easy negatives on rare labels."""

        def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05) -> None:
            super().__init__()
            self.gamma_neg = float(gamma_neg)
            self.gamma_pos = float(gamma_pos)
            self.clip = float(clip)

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            probs = torch.sigmoid(logits)
            xs_pos = probs.clamp(min=1e-7, max=1.0)
            xs_neg = (1.0 - probs + self.clip).clamp(min=1e-7, max=1.0)
            loss_pos = targets * torch.log(xs_pos) * torch.pow((1.0 - xs_pos).clamp(min=0.0), self.gamma_pos)
            loss_neg = (1.0 - targets) * torch.log(xs_neg) * torch.pow((1.0 - xs_neg).clamp(min=0.0), self.gamma_neg)
            return -(loss_pos + loss_neg).mean()

    class PrimaryIndependentHead(nn.Module):  # type: ignore[misc]
        """Architecture.md Section 3.1 primary independent 5-sigmoid projection head."""

        def __init__(self, hidden_size: int, num_labels: int = 5, dropout: float = 0.15) -> None:
            super().__init__()
            mid = max(64, hidden_size // 2)
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, num_labels),
            )

        def forward(self, pooled: torch.Tensor) -> torch.Tensor:
            return self.net(pooled)

    class ClassifierChainHead(nn.Module):  # type: ignore[misc]
        """Suite A5 sequential Classifier Chain head conditioning category k on prior predicted scores."""

        def __init__(self, hidden_size: int, num_labels: int = 5, dropout: float = 0.15) -> None:
            super().__init__()
            self.num_labels = num_labels
            self.norm = nn.LayerNorm(hidden_size)
            self.drop = nn.Dropout(dropout)
            mid = max(64, hidden_size // 2)
            self.shared_proj = nn.Sequential(nn.Linear(hidden_size, mid), nn.GELU(), nn.Dropout(dropout))
            self.classifiers = nn.ModuleList(
                [nn.Linear(mid + k, 1) for k in range(num_labels)]
            )

        def forward(
            self,
            pooled: torch.Tensor,
            teacher_targets: torch.Tensor | None = None,
        ) -> torch.Tensor:
            h = self.shared_proj(self.drop(self.norm(pooled)))
            logits: list[torch.Tensor] = []
            conds: list[torch.Tensor] = []
            for k in range(self.num_labels):
                if conds:
                    feat = torch.cat([h, *conds], dim=-1)
                else:
                    feat = h
                z_k = self.classifiers[k](feat)
                logits.append(z_k)
                if teacher_targets is not None and self.training:
                    conds.append(teacher_targets[:, k : k + 1].to(h.dtype))
                else:
                    conds.append(torch.sigmoid(z_k))
            return torch.cat(logits, dim=-1)

    class LabelInteractionResidualHead(nn.Module):  # type: ignore[misc]
        """Suite A5 label co-occurrence residual layer modeling pairwise logit dependencies."""

        def __init__(self, hidden_size: int, num_labels: int = 5, dropout: float = 0.15) -> None:
            super().__init__()
            self.base = PrimaryIndependentHead(hidden_size, num_labels=num_labels, dropout=dropout)
            self.interaction = nn.Sequential(
                nn.Linear(num_labels, num_labels * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(num_labels * 4, num_labels),
            )
            # Small residual init so optimization starts near stable independent baseline
            nn.init.zeros_(self.interaction[-1].weight)
            nn.init.zeros_(self.interaction[-1].bias)

        def forward(self, pooled: torch.Tensor) -> torch.Tensor:
            base_logits = self.base(pooled)
            return base_logits + self.interaction(torch.sigmoid(base_logits))

    def pool_encoder_hidden_states(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = "masked_mean",
    ) -> torch.Tensor:
        """Apply masked mean pooling (default) or CLS token pooling."""
        if pooling == "cls":
            return last_hidden_state[:, 0, :]
        if pooling == "masked_mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        raise ValueError(f"Unsupported pooling={pooling!r}; expected 'masked_mean' or 'cls'.")

    class MultiLabelHallucinationClassifier(nn.Module):  # type: ignore[misc]
        """
        Encoder transformer + configurable pooling & multi-label classification head.

        Supports verified HF backbones (`microsoft/deberta-v3-base`, `FacebookAI/roberta-base`,
        `distilbert/distilbert-base-uncased`, `answerdotai/ModernBERT-base`) and Suite A4/A5 variants.
        """

        def __init__(
            self,
            *,
            hf_model_id: str,
            num_labels: int = 5,
            pooling: str = "masked_mean",
            head_type: str = "independent_sigmoid",
            dropout: float = 0.15,
            chunk_aggregation: str = "logsumexp",
            encoder_override: nn.Module | None = None,
            hidden_size_override: int | None = None,
        ) -> None:
            super().__init__()
            if encoder_override is None:
                require_torch_and_transformers("MultiLabelHallucinationClassifier")
                cfg = AutoConfig.from_pretrained(hf_model_id)
                self.encoder = AutoModel.from_pretrained(hf_model_id, config=cfg)
                hidden_size = int(getattr(cfg, "hidden_size", 768))
            else:
                self.encoder = encoder_override
                hidden_size = int(hidden_size_override or getattr(encoder_override, "hidden_size", 768))

            self.hf_model_id = hf_model_id
            self.num_labels = int(num_labels)
            self.pooling = pooling
            self.head_type = head_type
            self.chunk_aggregation = chunk_aggregation

            if head_type == "independent_sigmoid":
                self.head: nn.Module = PrimaryIndependentHead(hidden_size, num_labels=num_labels, dropout=dropout)
            elif head_type == "classifier_chain":
                self.head = ClassifierChainHead(hidden_size, num_labels=num_labels, dropout=dropout)
            elif head_type == "label_interaction":
                self.head = LabelInteractionResidualHead(hidden_size, num_labels=num_labels, dropout=dropout)
            else:
                raise ValueError(f"Unsupported head_type={head_type!r}")

        def encode_pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            return pool_encoder_hidden_states(hidden, attention_mask, pooling=self.pooling)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            targets: torch.Tensor | None = None,
        ) -> torch.Tensor:
            pooled = self.encode_pooled(input_ids, attention_mask)
            if isinstance(self.head, ClassifierChainHead):
                return self.head(pooled, teacher_targets=targets)
            return self.head(pooled)

        def forward_sliding_chunks(
            self,
            chunk_input_ids: list[torch.Tensor],
            chunk_attention_masks: list[torch.Tensor],
        ) -> torch.Tensor:
            """Encode multiple chunks `(B=1, L)` per long sample and aggregate chunk logits."""
            logits_list = [
                self.forward(c_ids, c_mask).squeeze(0)
                for c_ids, c_mask in zip(chunk_input_ids, chunk_attention_masks)
            ]
            stacked = torch.stack(logits_list, dim=0)  # (M, 5)
            if self.chunk_aggregation == "max":
                return stacked.max(dim=0).values
            if self.chunk_aggregation == "mean":
                return stacked.mean(dim=0)
            if self.chunk_aggregation == "logsumexp":
                return torch.logsumexp(stacked, dim=0) - float(np.log(max(1, stacked.shape[0])))
            raise ValueError(f"Unsupported chunk_aggregation={self.chunk_aggregation!r}")

    def build_loss_fn(
        loss_cfg: dict[str, Any],
        y_train: np.ndarray | None = None,
        device: Any = None,
    ) -> nn.Module:
        """Instantiate Suite A6 loss function from training config."""
        ltype = str(loss_cfg.get("type", "bce")).strip().lower()
        if ltype == "bce":
            return nn.BCEWithLogitsLoss()
        if ltype == "weighted_bce":
            if y_train is None:
                raise ValueError("y_train matrix is required to compute capped pos_weight.")
            cap = float(loss_cfg.get("pos_weight_cap", 5.0))
            pw = torch.tensor(compute_pos_weights_from_matrix(y_train, cap=cap), dtype=torch.float32)
            if device is not None:
                pw = pw.to(device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
            if ltype == "asymmetric_focal":
                return AsymmetricFocalLoss(
                    gamma_neg=float(loss_cfg.get("asl_gamma_neg", 4.0)),
                    gamma_pos=float(loss_cfg.get("asl_gamma_pos", 1.0)),
                    clip=float(loss_cfg.get("asl_clip", 0.05)),
                )
        raise ValueError(f"Unsupported loss type={ltype!r}")
else:  # pragma: no cover - clean importability in non-PyTorch CPU environments
    AsymmetricFocalLoss = None  # type: ignore[assignment]
    PrimaryIndependentHead = None  # type: ignore[assignment]
    ClassifierChainHead = None  # type: ignore[assignment]
    LabelInteractionResidualHead = None  # type: ignore[assignment]
    MultiLabelHallucinationClassifier = None  # type: ignore[assignment]

    def pool_encoder_hidden_states(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        require_torch_and_transformers("pool_encoder_hidden_states")

    def build_loss_fn(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        require_torch_and_transformers("build_loss_fn")
