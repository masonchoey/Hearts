import random

import gymnasium as gym
import numpy as np
import pyspiel
from gymnasium import spaces
from open_spiel.python.rl_environment import Environment as OSPSingle


class HeartsGymEnvConservative(gym.Env):
    """A Gymnasium wrapper around OpenSpiel Hearts with conservative bot opponents.

    In this mode, Player 0 is controlled by the RL agent, while Players 1, 2, and 3
    are controlled by conservative bots that play the lowest legal card strategy.
    This provides a consistent baseline for testing RL agent performance.
    
    The environment handles the turn rotation internally and only presents decisions
    to the external RL agent when it's Player 0's turn to act.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config=None):
        # The base OpenSpiel environment (4-player Hearts).
        self._base_env = OSPSingle(pyspiel.load_game("hearts"), players=4)

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
        
        # Process bot moves until it's Player 0's turn or game ends
        obs = self._advance_to_rl_turn()
        return obs, {}

    def step(self, action):
        """Applies Player 0's action and advances game until next Player 0 turn.

        This method:
        1. Applies the RL agent's action (Player 0)
        2. Lets conservative bots play their turns (Players 1, 2, 3)
        3. Returns control when it's Player 0's turn again or game ends
        """
        ts = self._last_timestep
        current_player = ts.observations["current_player"]
        
        # Ensure it's actually Player 0's turn
        if current_player != 0:
            raise ValueError(f"Expected Player 0's turn, but it's Player {current_player}'s turn")

        # Apply the RL agent's action (Player 0)
        applied_action = int(action)
        
        # Track the action in game history
        self._game_history.append(applied_action)
        
        ts = self._base_env.step([applied_action])

        # Accumulate rewards for all players
        if ts.rewards is not None:
            for i in range(4):
                # Scale the rewards to be less negative and more learnable
                base_reward = ts.rewards[i]
                normalized_reward = base_reward / 10.0  # Scale down the magnitude
                self._episode_rewards[i] += normalized_reward

        self._last_timestep = ts
        
        # Check if game ended after Player 0's move
        if ts.last():
            terminated = True
            final_reward = self._episode_rewards[0]  # Player 0's cumulative reward
            obs = {
                "observations": np.zeros(self.observation_space["observations"].shape, dtype=np.float32),
                "action_mask": np.zeros(self._num_actions, dtype=np.int8)
            }
        else:
            # Game continues - advance through bot turns until Player 0's next turn
            obs = self._advance_to_rl_turn()
            terminated = self._last_timestep.last()
            final_reward = self._episode_rewards[0]  # Player 0's cumulative reward

        truncated = False  # OpenSpiel environments are episodic, not truncated.

        info = {
            "legal_actions": self._last_timestep.observations["legal_actions"][0] if not terminated else [],
            "current_player": self._last_timestep.observations["current_player"] if not terminated else -1,
            "all_player_rewards": self._episode_rewards.copy()
        }
        
        return obs, final_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _advance_to_rl_turn(self):
        """Advance the game through bot turns until it's Player 0's turn or game ends."""
        ts = self._last_timestep
        
        # Keep playing bot moves until it's Player 0's turn or game ends
        while not ts.last() and ts.observations["current_player"] != 0:
            current_player = ts.observations["current_player"]
            legal_actions = ts.observations["legal_actions"][current_player]
            
            # Conservative bot strategy: play the lowest legal card
            bot_action = self._conservative_bot_action(legal_actions)
            
            # Track the bot's action in game history
            self._game_history.append(bot_action)
            
            # Apply bot action
            ts = self._base_env.step([bot_action])
            
            # Accumulate rewards for all players
            if ts.rewards is not None:
                for i in range(4):
                    base_reward = ts.rewards[i]
                    normalized_reward = base_reward / 10.0
                    self._episode_rewards[i] += normalized_reward
            
            self._last_timestep = ts
        
        # Return observation for Player 0 or terminal state
        if ts.last():
            # Game ended
            return {
                "observations": np.zeros(self.observation_space["observations"].shape, dtype=np.float32),
                "action_mask": np.zeros(self._num_actions, dtype=np.int8)
            }
        else:
            # It's Player 0's turn
            return self._current_obs()

    def _conservative_bot_action(self, legal_actions):
        """Conservative bot strategy: play the lowest legal card.
        
        This implements a simple conservative strategy where bots always
        play the lowest value card from their legal actions.
        
        Args:
            legal_actions: List of legal action indices
            
        Returns:
            Selected action (lowest legal card)
        """
        if not legal_actions:
            return 0  # Fallback, should not happen
        
        # Simply return the lowest legal action (most conservative)
        return min(legal_actions)

    def _current_obs(self):
        """Returns the current observation for Player 0 with action mask."""
        ts = self._last_timestep
        player = ts.observations["current_player"]
        
        # Should always be Player 0 when this is called
        if player != 0:
            raise ValueError(f"Expected Player 0, but current player is {player}")
        
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
