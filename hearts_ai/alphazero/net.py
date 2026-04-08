"""
HeartsNet: neural network for AlphaZero-style Hearts self-play training.

Architecture:
  Input (5088 dims — raw OpenSpiel observation) → Linear projection → N Residual blocks
                  ↓
        ┌─────────┴──────────┐
    Policy head           Value head
   (52 card logits)     (1 scalar, sigmoid)

Input:  The raw 5088-dim OpenSpiel observation tensor for a single player.
        Layout (from openspiel_utils.py):
          [0:4]      pass direction
          [4:56]     dealt hand
          [56:108]   passed cards
          [108:160]  received cards
          [160:212]  current hand
          [212:356]  points per trick / player
          [356:5088] full trick history (13 tricks × 7 segments × 52 cards)

        During minimax depth-cutoff calls (NNWorldSolver._estimate_score), a
        *partial* reconstruction from PlayState is used: the current-hand and
        points slices are populated exactly; the trick-history slice is zeroed.
        Pre-training includes trick-history dropout (p=0.3) so the network is
        robust to both full and partial observations.

Policy head:  predicts a probability distribution over all 52 cards, masked to
              the legal actions.  Training target = MCTS visit distribution.

Value head:   predicts the agent's future point contribution from the current
              position, normalised to [0, 1] by dividing by 26.
              Pre-training target = evaluate_hand() heuristic score.
              AlphaZero target    = actual game outcome.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Raw OpenSpiel observation length for Hearts
OBS_DIM   = 5088
NUM_CARDS = 52
MAX_POINTS = 26.0  # Maximum points any player can take in Hearts


# ── Building blocks ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear + skip.
    Pre-norm (norm before transform) generally trains more stably than post-norm
    for deep MLPs and is the convention in transformer-based architectures.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim * 2)
        self.fc2   = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First sub-layer
        h = self.norm1(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        x = x + h
        # Second sub-layer (extra capacity without increasing depth)
        h = self.norm2(x)
        return x + self.drop(h)


# ── Main network ───────────────────────────────────────────────────────────

class HeartsNet(nn.Module):
    """
    Dual-head network for Hearts card-play evaluation.

    Args:
        input_dim:  Dimensionality of the input vector (default: 5088, the raw
                    OpenSpiel observation).  Can be set to a smaller value for
                    testing or if a pre-processed feature vector is preferred.
        hidden_dim: Width of all hidden layers (default: 256).
        n_blocks:   Number of residual blocks in the shared trunk (default: 4).
        dropout:    Dropout rate applied inside residual blocks (default: 0.1).
    """

    def __init__(
        self,
        input_dim: int = OBS_DIM,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_blocks   = n_blocks

        # Input projection: lift raw features into the hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Shared trunk
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Policy head → 52 logits (one per card)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, NUM_CARDS),
        )

        # Value head → scalar in [0, 1]  (scale by 26 to recover predicted points)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; bias initialised to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: float tensor of shape (batch, input_dim)

        Returns:
            policy_logits: (batch, 52) — raw logits before softmax
            value:         (batch, 1)  — predicted points / 26, in [0, 1]
        """
        h = self.input_proj(x)
        h = self.trunk(h)
        return self.policy_head(h), self.value_head(h)

    # ── Inference helpers (no_grad wrappers) ──────────────────────────────

    def _device(self) -> torch.device:
        """Return the device the model parameters currently live on."""
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(
        self,
        features: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Single-example inference.

        Args:
            features:    (input_dim,) float32 observation vector.
            legal_mask:  Optional (52,) bool array; illegal card logits are
                         set to -inf before softmax.

        Returns:
            policy_probs: (52,) float32 — probability distribution over cards.
            value_points: float — predicted points the agent will take (0–26).
        """
        self.eval()
        device = self._device()
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = self(x)
        logits = logits.squeeze(0)

        if legal_mask is not None:
            mask = torch.tensor(legal_mask, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~mask, float("-inf"))

        policy_probs = F.softmax(logits, dim=-1).cpu().numpy()
        value_points = float(value.squeeze().cpu()) * MAX_POINTS
        return policy_probs, value_points

    @torch.no_grad()
    def predict_value(self, features: np.ndarray) -> float:
        """
        Predict the agent's expected future point contribution from this position.

        This is the drop-in replacement for ``evaluate_hand`` used inside
        ``NNWorldSolver._estimate_score``.  Returns predicted future points
        (not accumulated points — those are tracked separately in PlayState).

        Args:
            features: (input_dim,) float32 observation vector.  During minimax
                      depth-cutoff calls this is a partial reconstruction from
                      PlayState; during self-play it is the full OpenSpiel obs.

        Returns:
            float — predicted remaining points the agent will take.
        """
        self.eval()
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self._device())
        _, value = self(x)
        return float(value.squeeze().cpu()) * MAX_POINTS

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save weights and hyperparameters to a single checkpoint file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim":  self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "n_blocks":   self.n_blocks,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "HeartsNet":
        """Load a previously saved HeartsNet checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        net  = cls(**cfg)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net
