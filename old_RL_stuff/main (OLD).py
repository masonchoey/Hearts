import numpy as np
import pyspiel
import os
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from hearts_env import HeartsGymEnv
import torch
import torch.nn as nn
from gymnasium import spaces as gym_spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

# game = pyspiel.load_game("hearts")
# state = game.new_initial_state()

# while not state.is_terminal():
#   print("legal actions: ", state.legal_actions())
#   state.apply_action(np.random.choice(state.legal_actions()))
#   print(str(state) + '\n')


# ---------------------------------------------------------------------------
# Register a proper RLlib environment creator. RLlib expects a *callable* that
# returns a **fresh** environment instance every time it is invoked (one per
# worker). Passing an *already-created* environment instance leads to shared
# state across workers and can break internal checks. We therefore register a
# factory with Tune's registry and reference it by name in the config.
# ---------------------------------------------------------------------------


def env_creator(env_config):
    """Factory that builds an OpenSpiel Hearts environment for RLlib."""
    return HeartsGymEnv(env_config)


register_env("hearts_env", env_creator)


# ---------------------------------------------------------------------------
# RLlib 2.x enables its *new API stack* by default. Mixing the old `model`
# configuration (used below) with the new stack leads to the AttributeError
# you observed (``'NoneType' object has no attribute 'enable_rl_module_and_learner'``).
# We therefore explicitly *disable* the new stack via `api_stack(...)` so we
# can continue using the classic ``model`` dict without changes.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Custom Torch model that applies action masking to logits
# ---------------------------------------------------------------------------


class ActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_outputs = num_outputs

        # Read hidden sizes from config or use sensible defaults
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        layers = []

        # Determine observation dimensionality robustly (Dict or Box)
        base_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(base_space, gym_spaces.Dict) and "observations" in base_space.spaces:
            obs_dim = int(np.prod(base_space["observations"].shape))
        else:
            # Fallback: already flattened Box
            obs_dim = int(np.prod(base_space.shape))

        last_dim = obs_dim
        for hidden_size in hiddens:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.ReLU())
            last_dim = hidden_size

        self.policy_net = nn.Sequential(*layers)
        self.logits_layer = nn.Linear(last_dim, num_outputs)
        self.value_net = nn.Sequential(
            nn.Linear(last_dim, max(128, last_dim)),
            nn.ReLU(),
            nn.Linear(max(128, last_dim), 1),
        )

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs_tensor = input_dict["obs"]
        # Support both Dict obs and flattened Tensor obs
        if isinstance(obs_tensor, dict) and "observations" in obs_tensor:
            obs = obs_tensor["observations"].float()
            action_mask = obs_tensor.get("action_mask", None)
            if action_mask is not None:
                action_mask = action_mask.float()
        else:
            obs = obs_tensor.float()
            action_mask = None

        features = self.policy_net(obs)
        logits = self.logits_layer(features)

        if action_mask is not None:
            # log(0) -> -inf, log(1) -> 0
            inf_mask = torch.clamp(torch.log(action_mask), min=torch.finfo(torch.float32).min)
            logits = logits + inf_mask

        # Store value output
        self._value_out = self.value_net(features).squeeze(-1)

        return logits, state

    def value_function(self):
        return self._value_out


# Register the custom model so it can be referenced by name in the config
ModelCatalog.register_custom_model("masked_action_model", ActionMaskModel)


ppo_config = (
    PPOConfig()
    # Disable the new API stack to avoid compatibility issues with the
    # legacy `model` specification used here.
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment("hearts_env")
    .framework("torch")
    .training(
        model={
            # Use custom action-masking model; disable RLlib LSTM wrapper to avoid conflicts
            "custom_model": "masked_action_model",
            "fcnet_hiddens": [256, 256],
        },
        # Improved training hyperparameters for better learning
        num_epochs=15,          # Increased from 10 for more learning per batch
        minibatch_size=64,      # Reduced from 128 for more gradient updates
        train_batch_size=8000,  # Increased from 4000 for more diverse samples
        lr=3e-4,               # Reduced learning rate for more stable learning
        entropy_coeff=0.01,     # Encourage exploration
        vf_loss_coeff=1.0,      # Increase value function learning
        clip_param=0.2,         # Standard PPO clipping
        grad_clip=0.5,          # Gradient clipping for stability
        # GAE parameters for better value estimation
        use_gae=True,
        lambda_=0.95,
        gamma=0.99,             # Standard discount factor
    )
    .env_runners(
        num_env_runners=2,      # Increased for more parallel experience collection
        num_envs_per_env_runner=1,
    )
    .evaluation(
        evaluation_interval=10,  # More frequent evaluation
        evaluation_duration=100, # Reduced to speed up evaluation
        evaluation_duration_unit="episodes",
        evaluation_config={"explore": False}
    )
    # Add debugging and logging
    .debugging(
        log_level="INFO"
    )
)

# The ONLY way to limit training iterations is through tune.RunConfig:
print("Starting PPO training...")
print("Press Ctrl+C at any time to stop training early.")
print("-" * 70)

try:
    results = tune.Tuner(
        "PPO", 
        param_space=ppo_config,
        run_config=tune.RunConfig(
            stop={
                "training_iteration": 2,  # Reduce for quick debug run
                # Other stopping criteria options:
                # "timesteps_total": 50000,     # Stop after 50k timesteps
                # "env_runners/episode_reward_mean": -20,   # Stop when avg reward reaches -20 (corrected key)
            },
            # Add checkpoint configuration to save model properly
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="env_runners/episode_reward_mean",  # Corrected key
                checkpoint_frequency=1,   # Save checkpoint every iteration for debug
                num_to_keep=3,           # Keep 3 most recent checkpoints
                checkpoint_at_end=True,  # Ensure final checkpoint is saved
            ),
        )
    ).fit()
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("Final results:")
    print(f"Best trial: {results.get_best_result()}")
    
except KeyboardInterrupt:
    print("\n" + "=" * 70)
    print("Training interrupted by user (Ctrl+C)")
    print("Checkpoints have been saved and can be found in the Ray results directory.")
    print("You can resume training later or use the saved checkpoints for evaluation.")
    
except Exception as e:
    print(f"\n" + "=" * 70)
    print(f"Training stopped due to an error: {e}")
    print("Check the logs and checkpoints in the Ray results directory.")
    
finally:
    print("Training session ended.")
    print("-" * 70)