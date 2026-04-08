"""
Replay buffer for AlphaZero Hearts training.

Stores (features, policy, value) tuples collected during self-play.
Oldest examples are evicted once the buffer reaches capacity, so the agent
always trains on the most recent experience.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class TrainingExample:
    """
    One supervised learning example produced by self-play.

    Attributes:
        features: (85,) float32 — agent-perspective state encoding.
        policy:   (52,) float32 — normalized MCTS visit distribution.
                  Non-legal cards have probability 0.
        value:    float — the agent's actual final score for the game
                  (points taken, range 0–26).
    """

    features: np.ndarray
    policy:   np.ndarray
    value:    float


class ReplayBuffer:
    """
    Circular buffer that retains the most recent ``max_size`` training examples.

    Thread-safety note: this class is NOT thread-safe.  If you add multi-process
    self-play in the future, wrap access with a lock or use a queue.

    Args:
        max_size: Maximum number of examples to keep.  When the buffer is full,
                  the oldest example is discarded on each ``add`` call.
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self._buf: deque[TrainingExample] = deque()

    # ── Mutation ──────────────────────────────────────────────────────────

    def add(self, example: TrainingExample) -> None:
        """Append one example, evicting the oldest if over capacity."""
        if len(self._buf) >= self.max_size:
            self._buf.popleft()
        self._buf.append(example)

    def extend(self, examples: Sequence[TrainingExample]) -> None:
        """Bulk-add a sequence of examples (e.g., all turns from one game)."""
        for ex in examples:
            self.add(ex)

    def clear(self) -> None:
        """Empty the buffer (useful when starting a fresh training run)."""
        self._buf.clear()

    # ── Sampling ──────────────────────────────────────────────────────────

    def sample_batch(self, batch_size: int) -> List[TrainingExample]:
        """
        Sample a random mini-batch without replacement.

        Returns at most ``min(batch_size, len(self))`` examples.
        """
        n = min(batch_size, len(self._buf))
        return random.sample(list(self._buf), n)

    # ── Introspection ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self._buf)}, max_size={self.max_size})"

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the buffer to disk as a compressed numpy archive."""
        if not self._buf:
            return
        features = np.stack([ex.features for ex in self._buf])
        policies = np.stack([ex.policy   for ex in self._buf])
        values   = np.array([ex.value    for ex in self._buf], dtype=np.float32)
        np.savez_compressed(path, features=features, policies=policies, values=values)

    @classmethod
    def load(cls, path: str, max_size: int = 100_000) -> "ReplayBuffer":
        """Reload a buffer previously saved with ``save``."""
        buf = cls(max_size=max_size)
        data = np.load(path)
        features = data["features"]
        policies = data["policies"]
        values   = data["values"]
        for f, p, v in zip(features, policies, values):
            buf.add(TrainingExample(features=f, policy=p, value=float(v)))
        return buf
