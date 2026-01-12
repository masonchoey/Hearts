"""
Hearts Gymnasium Environment for Human vs AI
A Gymnasium wrapper around OpenSpiel Hearts with 1 human player and 3 AI players.

This environment supports real-time inference where:
- Player 0 is controlled by the human (via step() calls)
- Players 1-3 are controlled by the trained AI model
- The environment automatically plays AI turns between human moves
"""

import gymnasium as gym
import numpy as np
import pyspiel
from gymnasium import spaces
from open_spiel.python.rl_environment import Environment as OSPSingle
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()
sparse_reward = os.getenv("SPARSE_REWARD")
print(f"SPARSE_REWARD: {sparse_reward}")
if sparse_reward is None:
    ERROR("SPARSE_REWARD is not set")
    exit(1)

class HeartsGymEnvHumanVsAI(gym.Env):
    """A Gymnasium wrapper for Hearts with 1 human player vs 3 AI players.
    
    The environment handles the game flow automatically:
    - Human player (player 0) provides actions through step()
    - AI players (players 1-3) are automatically controlled by the provided model
    - step() returns when it's the human player's turn
    - Action masking ensures only legal actions are available
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, env_config=None):
        """Initialize the environment.
        
        Args:
            env_config: Optional dict containing:
                - ai_model: Instance of HeartsAIModel for AI players
                - human_player_id: Which player is human (default: 0)
        """
        # The base OpenSpiel environment (4-player Hearts)
        self._base_env = OSPSingle(pyspiel.load_game("hearts"), players=4)
        
        # Store configuration
        self._env_config = env_config or {}
        self._ai_model = self._env_config.get("ai_model", None)
        self._human_player_id = self._env_config.get("human_player_id", 0)
        
        # Get observation and action space dimensions
        obs_size = self._base_env.observation_spec()["info_state"][0]
        num_actions = self._base_env.action_spec()["num_actions"]
        
        # Extended observation space to include action mask
        self.observation_space = spaces.Dict({
            "observations": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_size,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(num_actions,),
                dtype=np.int8,
            )
        })
        
        self.action_space = spaces.Discrete(num_actions)
        self._num_actions = num_actions
        
        # Track the last TimeStep from OpenSpiel
        self._last_timestep = None
        
        # Track complete game history for debugging/analysis
        self._game_history = []  # List of (player_id, action) tuples
        
        # Track rewards for all players during the episode
        self._episode_rewards = [0.0, 0.0, 0.0, 0.0]
        
        if self._ai_model is None:
            print("Warning: No AI model provided. AI players will use random policy.")
    
    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.
        
        Returns:
            observation: Dict with 'observations' and 'action_mask' for human player
            info: Dict with game state information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the game to the initial state
        ts = self._base_env.reset()
        self._last_timestep = ts
        self._episode_rewards = [0.0, 0.0, 0.0, 0.0]
        self._game_history = []
        
        # If the human player is not first, play AI turns until human's turn
        while not self._is_human_turn() and not self._last_timestep.last():
            self._play_ai_turn()
        
        obs = self._get_human_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int):
        """Apply the human player's action and play AI turns until next human turn.
        
        Args:
            action: The action index chosen by the human player
            
        Returns:
            observation: Dict with 'observations' and 'action_mask' for human player
            reward: Reward for the human player
            terminated: Whether the game is over
            truncated: Whether the episode was truncated (always False)
            info: Dict with game state information
        """
        if self._last_timestep.last():
            raise RuntimeError("Cannot step in a terminated game. Call reset() first.")
        
        if not self._is_human_turn():
            raise RuntimeError(f"Not human player's turn. Current player: {self._last_timestep.observations['current_player']}")
        
        # Validate that the action is legal
        current_player = self._last_timestep.observations["current_player"]
        legal_actions = self._last_timestep.observations["legal_actions"][current_player]
        
        if action not in legal_actions:
            raise ValueError(f"Illegal action {action}. Legal actions: {legal_actions}")
        
        # Apply human player's action
        action = int(action)
        self._game_history.append((current_player, action))
        ts = self._base_env.step([action])
        
        # Accumulate rewards
        if ts.rewards is not None:
            for i in range(4):
                self._episode_rewards[i] += ts.rewards[i]
        
        self._last_timestep = ts
        
        if not sparse_reward:
        # Track reward from human player's action (before AI moves)
        # The reward from ts (after human's action) is what we want to return
            if ts.rewards is not None:
                human_reward = ts.rewards[self._human_player_id]
            else:
                human_reward = 0.0
        
        # Play AI turns until it's human's turn again or game ends
        while not self._is_human_turn() and not self._last_timestep.last():
            self._play_ai_turn()
        
        # Check if game is terminated
        terminated = self._last_timestep.last()
        truncated = False
        
        if terminated:
            if sparse_reward:
                human_reward = self._episode_rewards[self._human_player_id]
            obs = {
                "observations": np.zeros(self.observation_space["observations"].shape, dtype=np.float32),
                "action_mask": np.zeros(self._num_actions, dtype=np.int8)
            }
        else:
            if sparse_reward:
            # Game continues - return 0 reward during play (sparse reward at end)
                human_reward = 0.0
            obs = self._get_human_observation()
        
        info = self._get_info()
        
        return obs, human_reward, terminated, truncated, info
    
    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    
    def _is_human_turn(self) -> bool:
        """Check if it's currently the human player's turn."""
        if self._last_timestep is None or self._last_timestep.last():
            return False
        current_player = self._last_timestep.observations["current_player"]
        return current_player == self._human_player_id
    
    def _play_ai_turn(self):
        """Play one turn for an AI player."""
        if self._last_timestep.last():
            return
        
        current_player = self._last_timestep.observations["current_player"]
        
        if current_player == self._human_player_id:
            raise RuntimeError("Cannot play AI turn for human player")
        
        # Get AI player's observation and legal actions
        obs = np.array(self._last_timestep.observations["info_state"][current_player], dtype=np.float32)
        legal_actions = self._last_timestep.observations["legal_actions"][current_player]
        
        # Get action from AI model
        if self._ai_model is not None:
            try:
                action = self._ai_model.get_action(obs, legal_actions)
            except Exception as e:
                print(f"Error getting AI action: {e}. Using random action.")
                action = np.random.choice(legal_actions)
        else:
            # Fallback to random if no model
            action = np.random.choice(legal_actions)
        
        # Apply AI action
        self._game_history.append((current_player, action))
        ts = self._base_env.step([action])
        
        # Accumulate rewards
        if ts.rewards is not None:
            for i in range(4):
                self._episode_rewards[i] += ts.rewards[i]
        
        self._last_timestep = ts
    
    def _get_human_observation(self) -> Dict[str, np.ndarray]:
        """Get observation dict for the human player."""
        if self._last_timestep is None or self._last_timestep.last():
            return {
                "observations": np.zeros(self.observation_space["observations"].shape, dtype=np.float32),
                "action_mask": np.zeros(self._num_actions, dtype=np.int8)
            }
        
        player = self._human_player_id
        obs_vec = self._last_timestep.observations["info_state"][player]
        
        # Create action mask (1 for legal actions, 0 for illegal)
        action_mask = np.zeros(self._num_actions, dtype=np.int8)
        legal_actions = self._last_timestep.observations["legal_actions"][player]
        action_mask[legal_actions] = 1
        
        return {
            "observations": np.array(obs_vec, dtype=np.float32),
            "action_mask": action_mask
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with current game state."""
        if self._last_timestep is None or self._last_timestep.last():
            return {
                "legal_actions": [],
                "current_player": -1,
                "all_player_rewards": self._episode_rewards.copy(),
                "game_history": self._game_history.copy(),
                "is_human_turn": False
            }
        
        current_player = self._last_timestep.observations["current_player"]
        return {
            "legal_actions": self._last_timestep.observations["legal_actions"][current_player],
            "current_player": current_player,
            "all_player_rewards": self._episode_rewards.copy(),
            "game_history": self._game_history.copy(),
            "is_human_turn": self._is_human_turn()
        }
    
    def get_game_history(self) -> List[tuple]:
        """Return the complete game history of all actions played by all players."""
        return self._game_history.copy()
    
    def get_all_rewards(self) -> List[float]:
        """Get accumulated rewards for all players."""
        return self._episode_rewards.copy()
    
    def get_human_reward(self) -> float:
        """Get accumulated reward for the human player."""
        return self._episode_rewards[self._human_player_id]
    
    def render(self, mode="human"):
        """Render the current game state (optional)."""
        if mode == "human":
            print(f"\n=== Hearts Game State ===")
            print(f"Current Player: {self._last_timestep.observations['current_player']}")
            print(f"Episode Rewards: {self._episode_rewards}")
            print(f"Game History Length: {len(self._game_history)}")
            return None
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")

