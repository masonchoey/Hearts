"""
Feature extraction for NNWorldSolver depth-cutoff evaluation.

The 5088-dim observation vector used as NN input ALWAYS comes directly from
the OpenSpiel game engine (``state.observation_tensor(agent_id)``).  It is
NEVER reconstructed from a PlayState.

  • Pre-training data generation  (pretrain.py):
        obs = np.array(ts.observations["info_state"][cp])   ← from OpenSpiel
        label = evaluate_hand(...)                          ← from PlayState
        TrainingExample(features=obs, ...)

  • Self-play ROOT decisions  (self_play.py):
        obs = np.array(ts.observations["info_state"][cp])   ← from OpenSpiel
        # NN policy head called here with real obs
        # DMCTS votes come from WorldSolver w/ evaluate_hand depth cutoffs

  • DMCTS depth-cutoff leaf nodes  (NNWorldSolver._estimate_score):
        Uses evaluate_hand directly on PlayState — the NN is NOT called here
        because no real OpenSpiel observation is available at leaf nodes.

This module only re-exports the OBS_DIM constant for convenience.
"""
from .net import OBS_DIM

__all__ = ["OBS_DIM"]
