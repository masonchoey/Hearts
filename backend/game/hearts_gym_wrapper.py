"""
Gymnasium Wrapper for Backend Integration
Bridges the HeartsGame class with the new Gymnasium environment for AI inference.
"""

import sys
import os
import numpy as np
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .hearts_env_human_vs_ai import HeartsGymEnvHumanVsAI
from ..models.dmcts_opponent_controller import DMCTSOpponentController


class HeartsGymWrapper:
    """
    Wrapper that integrates the Gymnasium environment with the backend.

    The opponent seats are driven by ``DMCTSOpponentController``, which runs
    three independent ``HeartsAgent`` players sharing an AlphaZero
    ``HeartsNet`` checkpoint (``ALPHAZERO_CHECKPOINT`` in the environment).
    Search hyperparameters (``N_WORLDS``, ``TIME_LIMIT_MS``, ``MAX_DEPTH``)
    are also read from the environment, mirroring ``dmcts_vs_bots.py``.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, human_player_id: int = 0, eager_load: bool = True):
        """
        Initialize the Gymnasium wrapper.

        Args:
            checkpoint_path: Unused.  Opponent configuration is read from
                the environment (see ``DMCTSOpponentController``).  Kept in
                the signature for backwards compatibility with existing
                callers (e.g. ``GameStateManager``).
            human_player_id: Which player is the human (0-3).
            eager_load: Unused for DMCTS (there is no heavy lazy load step).
                Kept for signature compatibility.
        """
        self.human_player_id = human_player_id
        del checkpoint_path, eager_load  # retained for compat; see docstring

        self.ai_model = DMCTSOpponentController(human_player_id=human_player_id)

        env_config = {
            "ai_model": self.ai_model,
            "human_player_id": human_player_id,
        }
        self.env = HeartsGymEnvHumanVsAI(env_config=env_config)
        
        # Current state
        self._current_obs = None
        self._current_info = None
        self._game_active = False
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Start a new game.
        
        Returns:
            Dict with:
                - legal_actions: List of legal action indices
                - observation: Observation vector for human player
                - is_human_turn: Whether it's currently human's turn
                - current_player: Current player ID
        """
        self._current_obs, self._current_info = self.env.reset(seed=seed)
        self._game_active = True
        
        return {
            "legal_actions": self._current_info["legal_actions"],
            "observation": self._current_obs["observations"],
            "action_mask": self._current_obs["action_mask"],
            "is_human_turn": self._current_info["is_human_turn"],
            "current_player": self._current_info["current_player"]
        }
    
    def step(self, action: int) -> Dict[str, Any]:
        """
        Apply human player's action and advance game state.
        
        Args:
            action: Action index chosen by human player
            
        Returns:
            Dict with:
                - legal_actions: List of legal action indices for next turn
                - observation: Observation vector for human player
                - reward: Reward received
                - terminated: Whether game is over
                - is_human_turn: Whether it's currently human's turn
                - current_player: Current player ID
                - all_rewards: Rewards for all players
                - game_history: Complete game history of (player_id, action) tuples
        """
        if not self._game_active:
            raise RuntimeError("No active game. Call reset() first.")
        
        # Take step in environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._current_obs = obs
        self._current_info = info
        
        if terminated:
            self._game_active = False
        
        return {
            "legal_actions": info["legal_actions"],
            "observation": obs["observations"],
            "action_mask": obs["action_mask"],
            "reward": reward,
            "terminated": terminated,
            "is_human_turn": info["is_human_turn"],
            "current_player": info["current_player"],
            "all_rewards": info["all_player_rewards"],
            "game_history": info["game_history"]
        }
    
    def get_legal_actions(self) -> List[int]:
        """Get list of currently legal action indices."""
        if not self._game_active or self._current_info is None:
            return []
        return self._current_info["legal_actions"]
    
    def get_observation(self) -> np.ndarray:
        """Get current observation vector for human player."""
        if not self._game_active or self._current_obs is None:
            return np.zeros(5088, dtype=np.float32)
        return self._current_obs["observations"]
    
    def get_action_mask(self) -> np.ndarray:
        """Get current action mask (1 for legal actions, 0 for illegal)."""
        if not self._game_active or self._current_obs is None:
            return np.zeros(52, dtype=np.int8)
        return self._current_obs["action_mask"]
    
    def is_game_active(self) -> bool:
        """Check if game is currently active."""
        return self._game_active
    
    def is_human_turn(self) -> bool:
        """Check if it's currently the human player's turn."""
        if not self._game_active or self._current_info is None:
            return False
        return self._current_info["is_human_turn"]
    
    def current_player(self) -> int:
        """Get current player ID (0-3, or -1 if game over)."""
        if not self._game_active or self._current_info is None:
            return -1
        return self._current_info["current_player"]
    
    def get_game_history(self) -> List[tuple]:
        """Get complete game history of (player_id, action) tuples."""
        return self.env.get_game_history()
    
    def get_all_rewards(self) -> List[float]:
        """Get accumulated rewards for all players."""
        return self.env.get_all_rewards()
    
    def get_human_reward(self) -> float:
        """Get accumulated reward for human player."""
        return self.env.get_human_reward()
    
    def shutdown(self):
        """Cleanup resources."""
        if self.ai_model:
            self.ai_model.shutdown()
        self._game_active = False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# Example usage function
# LEGACY: example_usage - Not used in current implementation
# This function exists for backward compatibility and documentation purposes
def example_usage():
    """Example of how to use the HeartsGymWrapper in the backend."""
    print("\n" + "="*60)
    print("Example: Using HeartsGymWrapper in Backend")
    print("="*60)
    
    # Get checkpoint path from environment or use default
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    
    if not checkpoint_path:
        print("\n⚠ No MODEL_CHECKPOINT_PATH set, using random policy")
    
    # Initialize wrapper
    wrapper = HeartsGymWrapper(checkpoint_path=checkpoint_path, human_player_id=0)
    
    # Start new game
    print("\n1. Starting new game...")
    state = wrapper.reset(seed=42)
    print(f"   ✓ Game started")
    print(f"   - Current player: {state['current_player']}")
    print(f"   - Legal actions: {len(state['legal_actions'])} available")
    print(f"   - Is human turn: {state['is_human_turn']}")
    
    # Play a few turns
    turn = 1
    while wrapper.is_game_active() and turn <= 5:
        if wrapper.is_human_turn():
            print(f"\n{turn}. Human player's turn")
            
            # Get legal actions
            legal_actions = wrapper.get_legal_actions()
            print(f"   - Legal actions: {legal_actions[:5]}... ({len(legal_actions)} total)")
            
            # Choose random action (in real backend, this comes from frontend)
            action = np.random.choice(legal_actions)
            print(f"   - Choosing action: {action}")
            
            # Apply action
            result = wrapper.step(action)
            print(f"   - Reward: {result['reward']}")
            print(f"   - Game over: {result['terminated']}")
            
            if result['terminated']:
                print(f"\n✓ Game completed!")
                print(f"   - Final rewards: {result['all_rewards']}")
                print(f"   - Human reward: {wrapper.get_human_reward()}")
                break
            
            turn += 1
        else:
            print(f"\n{turn}. Waiting for AI players...")
            # In the gymnasium environment, AI players play automatically
            # This should not happen since step() advances until human's turn
            break
    
    # Cleanup
    wrapper.shutdown()
    print(f"\n✓ Example completed successfully")


if __name__ == "__main__":
    example_usage()

