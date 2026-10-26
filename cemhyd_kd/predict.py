"""Inference for the original compact teacher/student MLP checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_metadata(path: str | Path) -> dict[str, float]:
    """Load composition and age fields supplied for a single microstructure."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Metadata JSON must be an object mapping feature names to numeric values.")
    try:
        return {str(key): float(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Every metadata value must be numeric.") from exc


def _torch() -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Neural prediction requires PyTorch. Install with `pip install torch`.") from exc
    return torch, nn


def _make_model(input_dim: int, meta: dict[str, Any], torch: Any, nn: Any) -> Any:
    """Rebuild the original two-layer MLP; reject incompatible checkpoints."""
    args = meta.get("args", {})
    architecture = meta.get("architecture")
    if architecture:
        raise ValueError(
            "This checkpoint uses the final flexible ensemble architecture. "
            "Its public inference adapter has not yet been released; use a compact MLP checkpoint."
        )
    hidden = int(args.get("hidden_dim", 64))
    embed = int(args.get("embed_dim", 32))
    dropout = float(args.get("dropout", 0.10))

    class MLPRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, embed), nn.LayerNorm(embed), nn.ReLU(),
            )
            self.head = nn.Linear(embed, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            embedding = self.backbone(x)
            return self.head(embedding), embedding

    return MLPRegressor()


def _scalers(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    meta = payload.get("meta", {})
    macro = payload.get("macro_scaler", meta.get("macro_scaler"))
    teacher = payload.get("teacher_scaler", meta.get("teacher_scaler"))
    target = payload.get("y_scaler", meta.get("y_scaler"))
    if not all(isinstance(value, dict) and "mean" in value and "std" in value for value in (macro, teacher, target)):
        raise ValueError("Checkpoint lacks required scaler metadata.")
    return macro, teacher, target


def predict_checkpoint(checkpoint: str | Path, feature_values: dict[str, float], role: str) -> float:
    """Predict MPa from a compact teacher or student checkpoint.

    ``role`` is ``teacher`` for macro+micro inference and ``student`` for the
    distilled macro-only student. The latter deliberately does not receive the
    microstructure descriptors, matching the manuscript's deployment setting.
    """
    torch, nn = _torch()
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    macro_cols = list(meta.get("macro_cols", []))
    micro_cols = list(meta.get("micro_cols", []))
    if not macro_cols:
        raise ValueError("Checkpoint metadata has no macro_cols schema.")
    cols = macro_cols + micro_cols if role == "teacher" else macro_cols
    missing = [name for name in cols if name not in feature_values]
    if missing:
        raise ValueError(f"Input is missing required fields: {missing}")
    macro_scaler, teacher_scaler, y_scaler = _scalers(payload)
    scaler = teacher_scaler if role == "teacher" else macro_scaler
    x = np.asarray([[feature_values[name] for name in cols]], dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if x.shape[1] != mean.size:
        raise ValueError("Checkpoint scaler dimensionality does not match its feature schema.")
    model = _make_model(x.shape[1], meta, torch, nn)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        scaled, _ = model(torch.tensor((x - mean) / std, dtype=torch.float32))
    y_mean = float(np.asarray(y_scaler["mean"]).reshape(-1)[0])
    y_std = float(np.asarray(y_scaler["std"]).reshape(-1)[0])
    return float(scaled.item() * y_std + y_mean)
