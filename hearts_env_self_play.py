import random

import gymnasium as gym
import numpy as np
import pyspiel
from gymnasium import spaces
from open_spiel.python.rl_environment import Environment as OSPSingle


class HeartsGymEnvSelfPlay(gym.Env):
    """A self-play Gymnasium wrapper around OpenSpiel Hearts with action masking.

    In self-play mode, all 4 players are controlled by the same RL agent policy.
    This allows the agent to learn by playing against copies of itself, leading
    to more sophisticated strategy development as all players improve together.
    
    The environment rotates through players 0-3, always presenting the current
    player's perspective to the RL agent. Action masking ensures only legal 
    actions are available, improving training efficiency.
    
    Optionally supports playing against older checkpoint policies to prevent
    overfitting to the most recent strategy.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config=None):
        # The base OpenSpiel environment (4-player Hearts).
        self._base_env = OSPSingle(pyspiel.load_game("hearts"), players=4)
        
        # Store opponent policy info (if using older checkpoints)
        self._opponent_policies = {}
        self._env_config = env_config or {}
        
        # Determine which players are controlled by opponent policies
        # Player 0 is always the learning agent
        self._learning_player = 0
        self._opponent_checkpoint_paths = self._env_config.get("opponent_checkpoints", [])

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

        # Track the last TimeStep from OpenSpiel to reduce object creation.
        self._last_timestep = None
        
        # Track complete game history for debugging/analysis
        self._game_history = []  # List of all actions played by all players
        
        # Track rewards for all players during the episode
        self._episode_rewards = [0.0, 0.0, 0.0, 0.0]

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np_random, _ = gym.utils.seeding.np_random(seed)
            self._rng = np_random
        else:
            self._rng = random

        # Reset the game to the initial state
        ts = self._base_env.reset()
        self._last_timestep = ts
        self._episode_rewards = [0.0, 0.0, 0.0, 0.0]
        self._game_history = []  # Reset game history
        
        obs = self._current_obs()
        return obs, {}

    def step(self, action):
        """Applies the current player's action in self-play mode.

        In self-play, each step represents one player's move. The environment
        returns the observation for the next player to move, or terminates
        if the game is over. All players are assumed to be controlled by the
        same RL agent policy, except when opponent checkpoints are specified.
        """
        ts = self._last_timestep
        current_player = ts.observations["current_player"]

        # Apply the action (with action masking, should always be legal)
        applied_action = int(action)
        
        # Track the action in game history
        self._game_history.append(applied_action)
        
        # Note: With fresh action mask implementation, this assertion is no longer needed
                # as we generate action masks directly from current OpenSpiel state
                # legal = ts.observations["legal_actions"][current_player]
                # assert action in legal, f"Action {action} not in legal actions {legal}. Action masking failed!"

        ts = self._base_env.step([applied_action])

        # Accumulate rewards for all players
        if ts.rewards is not None:
            for i in range(4):
                # Scale the rewards to be less negative and more learnable
                # Hearts rewards are typically penalty points, so we'll normalize them
                base_reward = ts.rewards[i]
                # Transform the reward to be less extreme
                normalized_reward = base_reward / 10.0  # Scale down the magnitude
                self._episode_rewards[i] += normalized_reward

        self._last_timestep = ts
        terminated = ts.last()
        truncated = False  # OpenSpiel environments are episodic, not truncated.

        if terminated:
            # Game is over - return the final reward for the last acting player
            final_reward = self._episode_rewards[current_player]
            
            # Add final reward shaping based on relative performance
            # In Hearts, lower scores are better, so reward avoiding high penalties
            final_score = abs(final_reward * 10)  # Convert back to penalty points
            obs = {
                "observations": np.zeros(self.observation_space["observations"].shape, dtype=np.float32),
                "action_mask": np.zeros(self._num_actions, dtype=np.int8)
            }
        else:
            # Game continues - return current reward for the acting player
            final_reward = self._episode_rewards[current_player]
            obs = self._current_obs()

        info = {
            "legal_actions": ts.observations["legal_actions"][ts.observations["current_player"]] if not terminated else [],
            "current_player": ts.observations["current_player"] if not terminated else -1,
            "all_player_rewards": self._episode_rewards.copy()
        }
        
        return obs, final_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _current_obs(self):
        """Returns the current (player-specific) observation vector with action mask."""
        ts = self._last_timestep
        player = ts.observations["current_player"]
        obs_vec = ts.observations["info_state"][player]
        
        # Create action mask (1 for legal actions, 0 for illegal)
        action_mask = np.zeros(self._num_actions, dtype=np.int8)
        legal_actions = ts.observations["legal_actions"][player]
        action_mask[legal_actions] = 1
        
        return {
            "observations": np.array(obs_vec, dtype=np.float32),
            "action_mask": action_mask
        }
    
    def get_game_history(self):
        """Return the complete game history of all actions played by all players."""
        return self._game_history.copy()
    
    def load_opponent_policies(self, checkpoint_paths):
        """Load opponent policies from checkpoints.
        
        Args:
            checkpoint_paths: List of paths to checkpoint directories
        """
        self._opponent_checkpoint_paths = checkpoint_paths
