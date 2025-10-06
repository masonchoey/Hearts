import random

import gymnasium as gym
import numpy as np
import pyspiel
from gymnasium import spaces
from open_spiel.python.rl_environment import Environment as OSPSingle


class HeartsGymEnv(gym.Env):
    """A single-agent Gymnasium wrapper around OpenSpiel Hearts with action masking.

    Only the first player (player_id=0) is controlled by the RL agent.  All
    other players take *uniformly random* legal actions.  The environment
    exposes a *vector* observation equal to the OpenSpiel ``info_state`` tensor
    of the *current* player and uses a discrete action space with action masking.
    
    Action masking ensures that the RL agent only receives legal actions to choose from,
    eliminating the need for illegal action handling and improving training efficiency.
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

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np_random, _ = gym.utils.seeding.np_random(seed)
            self._rng = np_random
        else:
            self._rng = random

        # Advance the game until it is player 0's turn (agent-controlled).
        ts = self._base_env.reset()
        while ts.observations["current_player"] != 0 and not ts.last():
            legal = ts.observations["legal_actions"][ts.observations["current_player"]]
            ts = self._base_env.step([random.choice(legal)])

        self._last_timestep = ts
        obs = self._current_obs()
        return obs, {}

    def step(self, action):
        """Applies the agent's ``action`` and then plays out random opponents.

        With action masking enabled, the agent should only provide legal actions.
        We keep advancing the underlying OpenSpiel environment until it is
        again player 0's turn **or** until the episode terminates.  Rewards are
        accumulated for player 0 across these internal steps.
        """
        accumulated_reward = 0.0
        terminated = False

        ts = self._last_timestep

        while True:
            current_player = ts.observations["current_player"]

            if current_player == 0:
                # With action masking, the RL agent should only receive legal actions
                # So we can directly use the action without validation
                applied_action = int(action)
                
                # Note: With fresh action mask implementation, this assertion is no longer needed
                # as we generate action masks directly from current OpenSpiel state
                # legal = ts.observations["legal_actions"][current_player]
                # assert action in legal, f"Action {action} not in legal actions {legal}. Action masking failed!"
            else:
                legal = ts.observations["legal_actions"][current_player]
                applied_action = int(self._rng.choice(legal))

            ts = self._base_env.step([applied_action])

            if ts.rewards is not None:
                # Scale the rewards to be less negative and more learnable
                # Hearts rewards are typically penalty points, so we'll normalize them
                base_reward = ts.rewards[0]
                # Transform the reward to be less extreme
                normalized_reward = base_reward / 10.0  # Scale down the magnitude
                accumulated_reward += normalized_reward

            # Break when episode ends or we are back to player 0's turn
            if ts.last() or ts.observations["current_player"] == 0:
                break

        self._last_timestep = ts
        terminated = ts.last()
        truncated = False  # OpenSpiel environments are episodic, not truncated.

        if terminated:
            obs = {
                "observations": np.zeros(self.observation_space["observations"].shape, dtype=np.float32),
                "action_mask": np.zeros(self._num_actions, dtype=np.int8)
            }
            # Add a final reward shaping based on relative performance
            # In Hearts, lower scores are better, so reward avoiding high penalties
            final_score = abs(accumulated_reward * 10)  # Convert back to penalty points
            if final_score == 0:
                accumulated_reward += 10.0  # Bonus for perfect game
            elif final_score <= 5:
                accumulated_reward += 5.0   # Bonus for excellent game
            elif final_score >= 20:
                accumulated_reward -= 5.0   # Penalty for poor game
        else:
            obs = self._current_obs()

        info = {"legal_actions": ts.observations["legal_actions"][0] if not terminated else []}
        return obs, accumulated_reward, terminated, truncated, info

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