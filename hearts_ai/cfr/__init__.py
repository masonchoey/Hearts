"""
hearts_ai.cfr — Counterfactual Regret Minimization for Hearts.

Algorithm
─────────
Uses OpenSpiel's Outcome Sampling Monte Carlo CFR (OS-MCCFR).  Both tabular
CFR and External Sampling MCCFR are impractical for Hearts (tabular CFR
cannot enumerate the ~10^100+ information states; ES-MCCFR has factorial
branching in hand size).  OS-MCCFR samples one complete outcome per player
per iteration — O(game_length) ≈ O(52) work — making it the only practical
choice.  The information-state table grows incrementally and the average
strategy converges to an approximate Nash equilibrium.

Public API
──────────
HeartsCFRSolver
    Core solver.  Wraps ``ExternalSamplingSolver`` from OpenSpiel, adds
    ``train()``, ``save()``, and ``from_checkpoint()`` for convenient
    iteration, checkpointing, and resumption.

CFRAgent
    Play-time agent.  Takes a trained solver (or checkpoint path) and
    exposes ``act(state)`` (sampled) and ``act_greedy(state)`` (argmax).
    Falls back to uniform random for information states not seen in training.

evaluate_vs_random
    Pit the CFR agent against uniform-random opponents over N games.

evaluate_vs_heuristic
    Pit the CFR agent against rule-based heuristic opponents over N games.

Quick start
───────────
    from hearts_ai.cfr import HeartsCFRSolver, CFRAgent

    solver = HeartsCFRSolver()
    solver.train(10_000, checkpoint_every=1000, checkpoint_dir="cfr_checkpoints")

    agent = CFRAgent(solver)
    # use agent.act(state) in a pyspiel game loop

    # Or load from checkpoint:
    agent = CFRAgent("cfr_checkpoints/cfr_0010000.pkl")
"""

from .solver    import HeartsCFRSolver
from .agent     import CFRAgent
from .evaluator import evaluate_vs_random, evaluate_vs_heuristic

__all__ = [
    "HeartsCFRSolver",
    "CFRAgent",
    "evaluate_vs_random",
    "evaluate_vs_heuristic",
]
