#!/usr/bin/env python3
"""
PPO Self-Play Training Script - Optimized for NVIDIA T4 GPU
"""

import numpy as np
import pyspiel
import os
import argparse
import glob
import csv
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from hearts_env_self_play import HeartsGymEnvSelfPlay
import torch
import torch.nn as nn
from gymnasium import spaces as gym_spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


def env_creator_self_play(env_config):
    """Factory that builds a self-play OpenSpiel Hearts environment for RLlib."""
    return HeartsGymEnvSelfPlay(env_config)


register_env("hearts_env_self_play", env_creator_self_play)


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


def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='PPO Self-Play Training for Hearts (Optimized for NVIDIA T4 GPU)')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint directory to resume training from')
    parser.add_argument('--resume-from-latest', type=str, default=None,
                       help='Path to results directory - will automatically find latest checkpoint')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Number of training iterations to run (default: 500)')
    parser.add_argument('--use-mixed-precision', action='store_true',
                       help='Enable mixed precision training for T4 Tensor cores')
    
    return parser.parse_args()


def main():
    """Main training function optimized for NVIDIA T4 GPU."""
    # Parse command line arguments
    args = parse_arguments()

    # PPO Configuration - Optimized for NVIDIA T4 GPU
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("hearts_env_self_play")
        .framework("torch")
        .resources(
            # T4 GPU Configuration - Maximize GPU utilization
            num_gpus=1,                    # Use the T4 GPU for neural network training
            num_cpus_for_main_process=1,   # Reserve 1 CPU for main process
        )
        .training(
            model={
                "custom_model": "masked_action_model",
                "fcnet_hiddens": [1024, 1024, 512],  # Larger 3-layer network for T4
                "use_lstm": False,
                "max_seq_len": 1,
            },
            # T4-optimized training hyperparameters
            num_epochs=30,              # More epochs to fully utilize GPU
            minibatch_size=128,         # Large minibatches for T4 efficiency (16GB VRAM)
            train_batch_size=32000,     # Very large batch size leveraging T4's 16GB memory
            lr=5e-5,                   # Lower LR for stability with very large batches
            lr_schedule=None,          # Constant learning rate
            entropy_coeff=0.01,        # Slightly lower entropy for more focused learning
            vf_loss_coeff=1.0,
            clip_param=0.2,
            grad_clip=1.0,             # Slightly higher grad clip for large batches
            use_gae=True,
            lambda_=0.95,
            gamma=0.99,
            # Enable mixed precision if requested (T4 Tensor cores)
            **({"mixed_precision": True} if args.use_mixed_precision else {}),
        )
        .env_runners(
            # Scale environment runners based on available CPUs
            num_env_runners=2,          # Use 2 runners if you have more CPUs available
            num_envs_per_env_runner=4,  # Run 4 environments per runner for high throughput
            num_cpus_per_env_runner=1,  # 1 CPU per runner
        )
        .evaluation(
            evaluation_interval=25,     # Regular evaluation
            evaluation_duration=50,     # Thorough evaluation
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False}
        )
        .debugging(
            log_level="INFO"
        )
    )

    print("=" * 80)
    print("🚀 NVIDIA T4 GPU OPTIMIZED PPO Self-Play Training")
    print("=" * 80)
    print("T4 GPU Specifications:")
    print("  • 16 GB GDDR6 Memory")
    print("  • 2,560 CUDA Cores")
    print("  • 320 Tensor Cores")
    print("  • 65 TFLOPS FP16 Performance")
    print()
    print("Optimized Configuration:")
    print("  • Environment Runners: 2 (with 4 envs per runner = 8 total envs)")
    print("  • CPU Usage: 1 (main) + 2 (env runners) = 3 CPUs")
    print("  • GPU Usage: 1 T4 GPU for neural network training")
    print("  • Batch Size: 32,000 (leveraging T4's 16GB memory)")
    print("  • Network Size: 1024→1024→512 (3-layer deep network)")
    print("  • Training Epochs: 30 (maximize GPU utilization)")
    print("  • Minibatch Size: 128 (optimal for T4)")
    if args.use_mixed_precision:
        print("  • Mixed Precision: ENABLED (using Tensor cores)")
    else:
        print("  • Mixed Precision: DISABLED (use --use-mixed-precision to enable)")
    print("=" * 80)

    try:
        # Use Ray Tune for training
        run_config = tune.RunConfig(
            stop={"training_iteration": args.iterations},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="env_runners/episode_reward_mean",
                checkpoint_frequency=5,
                num_to_keep=5,
                checkpoint_at_end=True,
            ),
        )
        
        tuner = tune.Tuner("PPO", param_space=ppo_config, run_config=run_config)
        results = tuner.fit()
        
        print("\n" + "=" * 80)
        print("✅ T4 GPU Optimized Self-Play Training completed successfully!")
        print("Final results:")
        print(f"Best trial: {results.get_best_result()}")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Training interrupted by user (Ctrl+C)")
        print("Checkpoints have been saved and can be found in the Ray results directory.")
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ Training stopped due to an error: {e}")
        print("Check the logs and checkpoints in the Ray results directory.")
        
    finally:
        print("🏁 T4 GPU optimized training session ended.")
        print("💡 Pro Tip: Use 'nvidia-smi' to monitor GPU utilization during training")
        print("=" * 80)


if __name__ == "__main__":
    main()
