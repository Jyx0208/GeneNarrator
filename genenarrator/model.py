"""GN-AFT model: dual-tower encoders, QACMF fusion, Weibull AFT head (Methods)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Model (as described in the manuscript)
# ---------------------------------------------------------------------------


class GNAFT(nn.Module):
    """GeneNarrator (GN-AFT) multimodal survival model.

    Architecture:
      - Gene encoder: 1,000 -> 512 -> 256 (BatchNorm, GELU, dropout 0.35)
      - Semantic encoder: 1,024 -> 512 -> 256 (BatchNorm, GELU, dropout 0.21)
      - Quality estimators: 256 -> 64 -> 1 (tanh hidden, sigmoid output)
      - Bidirectional cross-attention: 4 heads, attention dropout 0.1
      - Adaptive weighted residual fusion: [w*g'; w*t'; g'; t'] -> 256
      - Weibull AFT head: 256 -> 128 -> 64 -> 2
    """

    def __init__(
        self,
        gene_dim: int = 1000,
        text_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.gene_gate = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.g2t_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.t2g_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.ln_g = nn.LayerNorm(hidden_dim)
        self.ln_t = nn.LayerNorm(hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.aft_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.constant_(self.aft_head[-1].bias[0], 6.9)
        nn.init.constant_(self.aft_head[-1].bias[1], 0.4)

    def forward(self, gene: torch.Tensor, text: torch.Tensor):
        g = self.gene_encoder(gene)
        t = self.text_encoder(text)

        g_gate = self.gene_gate(g)
        t_gate = self.text_gate(t)
        gate_sum = g_gate + t_gate + 1e-8
        w_g = g_gate / gate_sum
        w_t = t_gate / gate_sum

        g_seq = g.unsqueeze(1)
        t_seq = t.unsqueeze(1)
        g2t, _ = self.g2t_attn(g_seq, t_seq, t_seq)
        t2g, _ = self.t2g_attn(t_seq, g_seq, g_seq)

        g_enhanced = self.ln_g((g_seq + g2t).squeeze(1))
        t_enhanced = self.ln_t((t_seq + t2g).squeeze(1))

        weighted = w_g * g_enhanced + w_t * t_enhanced
        concat = torch.cat([weighted, g_enhanced, t_enhanced], dim=-1)
        fused = self.fusion(concat) + weighted

        params = self.aft_head(fused)
        scale = torch.exp(torch.clamp(params[:, 0], 3.5, 8.5))
        shape = 0.5 + 3.0 * torch.sigmoid(params[:, 1])
        return scale, shape

    def predict_median(self, gene: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Predicted median survival time: t = lambda * (ln 2)^(1/k)."""
        scale, shape = self.forward(gene, text)
        ln2 = torch.log(torch.tensor(2.0, device=scale.device))
        return scale * (ln2 ** (1.0 / shape))


# ---------------------------------------------------------------------------
# Loss (as described in the manuscript, incl. label smoothing and priors)
# ---------------------------------------------------------------------------


def weibull_loss(
    scale: torch.Tensor,
    shape: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    reg: float = 0.01,
) -> torch.Tensor:
    """Negative log-likelihood of the Weibull AFT model.

    Implementation notes (disclosed in the manuscript):
      - event indicator label smoothing: 0.95 * event + 0.025
      - scale prior: (ln lambda - 6.5)^2 ; shape prior: (k - 1.5)^2
    """
    eps = 1e-8
    scale = torch.clamp(scale, min=1.0)
    shape = torch.clamp(shape, min=0.1)
    time = torch.clamp(time, min=1.0)

    z = (time / scale) ** shape
    log_f = (
        torch.log(shape + eps)
        - torch.log(scale + eps)
        + (shape - 1) * (torch.log(time + eps) - torch.log(scale + eps))
        - z
    )
    log_S = -z

    smooth_event = event * 0.95 + 0.025
    nll = -torch.mean(smooth_event * log_f + (1 - smooth_event) * log_S)
    reg_scale = reg * torch.mean((torch.log(scale) - 6.5) ** 2)
    reg_shape = reg * torch.mean((shape - 1.5) ** 2)
    return nll + reg_scale + reg_shape


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _as_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def train_model(
    model: GNAFT,
    fit: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    *,
    epochs: int = 200,
    lr: float = 5e-5,
    batch_size: int = 64,
    patience: int = 30,
    device: Optional[torch.device] = None,
) -> float:
    """Train one model instance; early stopping on the development set.

    Validation C-index is checked every 5 epochs; training stops after
    ``patience // 5`` consecutive checks without improvement (manuscript:
    early stopping patience 30 epochs).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_gene = _as_tensor(fit["gene"]).to(device)
    train_text = _as_tensor(fit["text"]).to(device)
    train_time = _as_tensor(fit["time"]).to(device)
    train_event = _as_tensor(fit["event"]).to(device)
    val_gene = _as_tensor(val["gene"]).to(device)
    val_text = _as_tensor(val["text"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_ci, best_state, bad = 0.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_gene))
        for i in range(0, len(train_gene), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            scale, shape = model(train_gene[idx], train_text[idx])
            loss = weibull_loss(scale, shape, train_time[idx], train_event[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            ci = evaluate_model(model, val, device=device)
            if ci > best_ci:
                best_ci = ci
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= patience // 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ci


def evaluate_model(
    model: GNAFT,
    data: Dict[str, np.ndarray],
    device: Optional[torch.device] = None,
) -> float:
    """Harrell C-index on predicted median survival (risk = 1 / median)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        gene = _as_tensor(data["gene"]).to(device)
        text = _as_tensor(data["text"]).to(device)
        median = model.predict_median(gene, text).cpu().numpy()
    risk = 1.0 / (median + 1e-8)
    return float(concordance_index(data["time"], -risk, data["event"]))


def cross_validate(
    data: Dict[str, np.ndarray],
    *,
    n_folds: int = 5,
    seed: int = 42,
    epochs: int = 200,
    **kwargs,
) -> Tuple[float, float]:
    """Internal 5-fold cross-validation over the development cohort.

    Returns (mean fold C-index, standard deviation). Model selection and
    early stopping are performed inside each fold only.
    """
    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ci: List[float] = []
    for train_idx, val_idx in kf.split(data["gene"]):
        model = GNAFT(gene_dim=data["gene"].shape[1], text_dim=data["text"].shape[1])
        fold = {k: np.asarray(v)[train_idx] for k, v in data.items() if k != "sample_ids"}
        val = {k: np.asarray(v)[val_idx] for k, v in data.items() if k != "sample_ids"}
        fold_ci.append(train_model(model, fold, val, epochs=epochs, **kwargs))
    return float(np.mean(fold_ci)), float(np.std(fold_ci))
