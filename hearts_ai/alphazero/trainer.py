"""
Training loop for the HeartsNet AlphaZero agent.

Loss function
─────────────
  L = L_policy + L_value

  L_policy = CrossEntropy(softmax(logits), mcts_visit_distribution)
           = -Σ  π_mcts(a) · log softmax(logits(a))

  The MCTS policy target π is a proper probability distribution over the 52
  cards (zero probability for illegal cards), so using it directly as a soft
  label in cross-entropy is equivalent to KL(π || NN_policy) + const.

  L_value  = MSE(value_pred, z / 26)
           where z is the agent's raw final score (0–26 points).

  The two losses are equally weighted.  If you find one dominates, add a
  configurable scalar multiplier (``policy_weight`` / ``value_weight``).

Optimiser
─────────
  Adam with a cosine-annealing learning rate schedule over the training
  steps within each iteration.  Gradient norm is clipped to 1.0 to prevent
  occasional large updates when the buffer is first populated.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import HeartsNet, MAX_POINTS
from .replay_buffer import ReplayBuffer, TrainingExample


class Trainer:
    """
    Manages the optimizer and runs one training epoch over the replay buffer.

    Args:
        net:            The HeartsNet being trained (mutated in-place).
        lr:             Initial learning rate for Adam.
        weight_decay:   L2 regularisation coefficient.
        policy_weight:  Relative weight of the policy cross-entropy loss.
        value_weight:   Relative weight of the value MSE loss.
        grad_clip:      Maximum gradient L2 norm (0 = no clipping).
        device:         Torch device string; defaults to CUDA if available.
    """

    def __init__(
        self,
        net: HeartsNet,
        lr: float             = 1e-3,
        weight_decay: float   = 1e-4,
        policy_weight: float  = 1.0,
        value_weight: float   = 1.0,
        grad_clip: float      = 1.0,
        device: str | None    = None,
    ):
        self.net           = net
        self.policy_weight = policy_weight
        self.value_weight  = value_weight
        self.grad_clip     = grad_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    # ── Scheduler helpers ─────────────────────────────────────────────────

    def set_cosine_schedule(self, total_steps: int, eta_min: float = 1e-5) -> None:
        """
        Attach a cosine-annealing LR schedule over ``total_steps`` gradient updates.
        Call this once per pipeline iteration (before ``train_epoch``) so the
        schedule aligns with the number of batches you intend to run.
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=eta_min
        )

    # ── Training ──────────────────────────────────────────────────────────

    def train_epoch(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 256,
        n_batches: int  = 200,
    ) -> Dict[str, float]:
        """
        Run one epoch of supervised learning over the replay buffer.

        Args:
            buffer:     ReplayBuffer containing recent self-play examples.
            batch_size: Samples per gradient step.
            n_batches:  Number of gradient steps (batches) in this epoch.
                        If the buffer is smaller than batch_size, fewer steps
                        are actually taken.

        Returns:
            Dict with keys ``loss``, ``policy_loss``, ``value_loss``, ``lr``.
        """
        if len(buffer) < max(batch_size, 32):
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "lr": 0.0}

        self.net.train()
        total_loss   = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        steps_done   = 0

        for _ in range(n_batches):
            batch = buffer.sample_batch(batch_size)
            if not batch:
                break

            p_loss, v_loss = self._compute_losses(batch)
            loss = self.policy_weight * p_loss + self.value_weight * v_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss   += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            steps_done   += 1

        n = max(steps_done, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {
            "loss":         total_loss   / n,
            "policy_loss":  total_p_loss / n,
            "value_loss":   total_v_loss / n,
            "lr":           current_lr,
        }

    def _compute_losses(
        self, batch: List[TrainingExample]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy (cross-entropy) and value (MSE) losses for one batch.

        Policy cross-entropy:
            -Σ_a  π_mcts(a) · log(softmax(logits)(a))
          We sum over all 52 cards, but cards with π_mcts = 0 contribute
          nothing, so only visited actions influence the gradient.

        Value MSE:
            ||σ(value_head) - z / 26||²
          where σ is the sigmoid applied inside the network.
        """
        import numpy as _np
        features = torch.from_numpy(
            _np.stack([ex.features for ex in batch])
        ).float().to(self.device)

        # Policy target: (batch, 52)
        policies = torch.from_numpy(
            _np.stack([ex.policy for ex in batch])
        ).float().to(self.device)

        # Value target: (batch, 1) — normalized to [0, 1]
        values = torch.from_numpy(
            _np.array([[ex.value / MAX_POINTS] for ex in batch], dtype=_np.float32)
        ).to(self.device)
        # Clamp in case of shoot-the-moon producing negative targets
        values = values.clamp(0.0, 1.0)

        policy_logits, value_pred = self.net(features)

        # Cross-entropy with soft MCTS targets
        # F.cross_entropy expects class indices; use manual KL-form instead
        log_softmax = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(policies * log_softmax).sum(dim=-1).mean()

        # MSE value loss
        value_loss = F.mse_loss(value_pred, values)

        return policy_loss, value_loss

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def save_checkpoint(self, path: str, iteration: int) -> None:
        """Save network weights + optimizer state for resuming training."""
        torch.save(
            {
                "iteration":       iteration,
                "net_state_dict":  self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "net_config": {
                    "input_dim":  self.net.input_dim,
                    "hidden_dim": self.net.hidden_dim,
                    "n_blocks":   self.net.n_blocks,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        """
        Restore network weights and optimizer state.
        Returns the iteration number stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        return int(ckpt.get("iteration", 0))
