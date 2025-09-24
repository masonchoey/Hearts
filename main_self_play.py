#!/usr/bin/env python3
"""
PPO Self-Play Training Script with Resume Functionality

This script trains a PPO agent to play Hearts using self-play, with the ability
to resume training from previously saved checkpoints, including completed trials.
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


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint directories (checkpoint_XXXXXX format)
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number (extract number from checkpoint_XXXXXX)
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    return checkpoint_dirs[-1]  # Return the latest one


def find_experiment_checkpoint_dir(base_dir):
    """Find the experiment checkpoint directory within a base results directory."""
    if not os.path.exists(base_dir):
        return None
    
    # Look for PPO experiment directories
    pattern = os.path.join(base_dir, "PPO_hearts_env_self_play_*")
    experiment_dirs = glob.glob(pattern)
    
    if not experiment_dirs:
        return None
    
    # Return the first (and typically only) experiment directory
    return experiment_dirs[0]


def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='PPO Self-Play Training for Hearts')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint directory to resume training from')
    parser.add_argument('--resume-from-latest', type=str, default=None,
                       help='Path to results directory - will automatically find latest checkpoint')
    parser.add_argument('--iterations', type=int, default=250,
                       help='Number of training iterations to run (default: 250)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_arguments()

    # PPO Configuration
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("hearts_env_self_play")
        .framework("torch")
        .training(
            model={
                "custom_model": "masked_action_model",
                "fcnet_hiddens": [256, 256],
            },
            num_epochs=20,
            minibatch_size=32,
            train_batch_size=12000,
            lr=2e-4,
            entropy_coeff=0.02,
            vf_loss_coeff=1.0,
            clip_param=0.2,
            grad_clip=0.5,
            use_gae=True,
            lambda_=0.95,
            gamma=0.99,
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=1,
        )
        .evaluation(
            evaluation_interval=15,
            evaluation_duration=50,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False}
        )
        .debugging(
            log_level="INFO"
        )
    )

    # Determine if we're resuming from a checkpoint
    resume_from_checkpoint = None
    current_iteration = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            resume_from_checkpoint = os.path.abspath(args.resume)
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Error: Checkpoint path does not exist: {args.resume}")
            exit(1)
    elif args.resume_from_latest:
        # Find the experiment directory and latest checkpoint
        experiment_dir = find_experiment_checkpoint_dir(args.resume_from_latest)
        if experiment_dir:
            latest_checkpoint = find_latest_checkpoint(experiment_dir)
            if latest_checkpoint:
                resume_from_checkpoint = os.path.abspath(latest_checkpoint)
                print(f"Resuming training from latest checkpoint: {latest_checkpoint}")
                
                # Read current iteration from progress.csv
                progress_file = os.path.join(experiment_dir, "progress.csv")
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, 'r') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                current_iteration = int(row['training_iteration'])
                        print(f"Current training iteration: {current_iteration}")
                    except Exception as e:
                        print(f"Warning: Could not read current iteration: {e}")
            else:
                print(f"No checkpoints found in experiment directory: {experiment_dir}")
                exit(1)
        else:
            print(f"No experiment directory found in: {args.resume_from_latest}")
            exit(1)

    # Print training information
    if resume_from_checkpoint:
        print("Resuming PPO Self-Play Training...")
        print(f"Continuing from iteration {current_iteration}")
    else:
        print("Starting PPO Self-Play Training...")
        print("In self-play mode, the agent learns by playing against copies of itself.")
        print("This should lead to more sophisticated strategies over time.")

    print("Press Ctrl+C at any time to stop training early.")
    print("-" * 70)

    try:
        if resume_from_checkpoint and current_iteration > 0:
            # For completed trials, use manual training approach
            target_iterations = args.iterations
            
            # Ensure we're extending beyond current iteration
            if target_iterations <= current_iteration:
                target_iterations = current_iteration + 50
                print(f"⚠️  Target iterations ({args.iterations}) <= current iteration ({current_iteration})")
                print(f"📈 Extending training to {target_iterations} iterations")
            else:
                print(f"📈 Extending training to {target_iterations} iterations")
            
            print(f"🔄 Loading algorithm from checkpoint...")
            
            # Initialize Ray if not already initialized
            import ray
            if not ray.is_initialized():
                ray.init()
            
            # Load the algorithm from checkpoint
            algo = PPO.from_checkpoint(resume_from_checkpoint)
            print(f"✅ Successfully loaded algorithm from checkpoint")
            
            # Train manually for better control
            print(f"🏃 Starting training from iteration {current_iteration + 1} to {target_iterations}")
            
            for i in range(current_iteration + 1, target_iterations + 1):
                print(f"Iteration {i}/{target_iterations}")
                result = algo.train()
                
                # Save checkpoint every 5 iterations
                if i % 5 == 0 or i == target_iterations:
                    checkpoint = algo.save()
                    print(f"💾 Saved checkpoint at iteration {i}")
                
                # Print some metrics
                if 'env_runners/episode_reward_mean' in result:
                    reward = result['env_runners/episode_reward_mean']
                    print(f"  Episode reward mean: {reward:.3f}")
            
            print(f"✅ Training completed! Reached {target_iterations} iterations")
            
        else:
            # Use Ray Tune for new training or unfinished trials
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
            
            print("\n" + "=" * 70)
            print("Self-Play Training completed successfully!")
            print("Final results:")
            print(f"Best trial: {results.get_best_result()}")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Self-Play Training interrupted by user (Ctrl+C)")
        print("Checkpoints have been saved and can be found in the Ray results directory.")
        print("You can resume training later using:")
        print("  --resume <path_to_specific_checkpoint>")
        print("  --resume-from-latest <path_to_results_directory>")
        
    except Exception as e:
        print(f"\n" + "=" * 70)
        print(f"Self-Play Training stopped due to an error: {e}")
        print("Check the logs and checkpoints in the Ray results directory.")
        print("If checkpoints exist, you can resume training using the --resume or --resume-from-latest options.")
        
    finally:
        training_type = "resumed" if resume_from_checkpoint else "new"
        print(f"Self-Play Training session ({training_type}) ended.")
        print("-" * 70)
        
        # Print usage examples
        if not resume_from_checkpoint:
            print("\nTo resume this training later, use one of these commands:")
            print("  python main_self_play.py --resume-from-latest <results_directory>")
            print("  python main_self_play.py --resume <specific_checkpoint_path>")
            print("  python main_self_play.py --iterations 500  # for longer training")
            print("-" * 70)

 
if __name__ == "__main__":
    main()
