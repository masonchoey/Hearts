"""
hearts_ai.alphazero — AlphaZero-style self-play training for Hearts.

Architecture overview
─────────────────────
The 5088-dim observation comes EXCLUSIVELY from OpenSpiel.  It is NEVER
reconstructed from a PlayState.  PlayState is used only to run DMCTS (via
evaluate_hand at depth cutoffs) and to derive evaluate_hand labels for
Phase-1 supervised pre-training.

  Phase 1 — supervised pre-training:
    (obs_5088_from_OpenSpiel, evaluate_hand_label)  →  value-only MSE loss

  Phase 2 — AlphaZero self-play:
    ROOT: obs_5088_from_OpenSpiel  →  NN policy + DMCTS blended policy
    DEPTH CUTOFF: evaluate_hand(PlayState)  →  no NN call here
    Training: (obs_5088, blended_policy, game_outcome)

Public API
──────────
HeartsNet               Neural network (5088-dim raw obs input, policy + value heads).
OBS_DIM                 Raw OpenSpiel observation length (5088).
ReplayBuffer            Circular buffer of self-play training examples.
TrainingExample         Single (features, policy, value) data point.
NNWorldSolver           WorldSolver; depth cutoffs use evaluate_hand (not NN).
run_self_play_game      Run one complete Hearts game (OpenSpiel) and return examples.
Trainer                 Manages the PyTorch optimiser and training loop.
pit_networks            Compare two networks head-to-head (OpenSpiel-driven).
evaluate_vs_heuristic   Compare NN against evaluate_hand baseline.
AlphaZeroPipeline       Top-level self-play → train → pit → repeat orchestrator.
generate_heuristic_data Generate (obs_5088, evaluate_hand_score) pairs for pre-training.
pretrain                Phase-1 supervised pre-training function.
"""

from .features      import OBS_DIM
from .net           import HeartsNet
from .pretrain      import generate_heuristic_data, pretrain
from .replay_buffer import ReplayBuffer, TrainingExample
from .self_play     import NNWorldSolver, run_self_play_game
from .trainer       import Trainer
from .evaluator     import pit_networks, evaluate_vs_heuristic
from .pipeline      import AlphaZeroPipeline

__all__ = [
    "OBS_DIM",
    "generate_heuristic_data",
    "pretrain",
    "HeartsNet",
    "ReplayBuffer",
    "TrainingExample",
    "NNWorldSolver",
    "run_self_play_game",
    "Trainer",
    "pit_networks",
    "evaluate_vs_heuristic",
    "AlphaZeroPipeline",
]
