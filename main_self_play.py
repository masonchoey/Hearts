#!/usr/bin/env python3
"""
PPO Self-Play Training Script with Resume Functionality and Checkpoint Rotation

This script trains a PPO agent to play Hearts using self-play, with the ability
to resume training from previously saved checkpoints, including completed trials.

Key Features:
- Centralized configuration: All hyperparameters and settings declared at the top
- Checkpoint rotation: Periodically trains against older checkpoint policies to 
  prevent overfitting to the most recent strategy
- Configurable opponent probability: Control how often older checkpoints are used
- Automatic checkpoint discovery: Finds recent checkpoints up to N iterations back
- W&B integration: Logs checkpoint usage and training metrics

Configuration:
  All hyperparameters and training settings are declared as constants at the top 
  of this file (after imports). To modify any aspect of training, simply edit the 
  constants in the configuration sections:
  
  - Training Hyperparameters (learning rate, epochs, batch sizes, etc.)
  - Environment Settings (number of runners, environments per runner)
  - Model Architecture (embedding dimensions, attention heads, etc.)
  - Evaluation Settings (frequency, duration)
  - Checkpoint Settings (frequency, number to keep)
  - Resource Settings (GPU/CPU allocation)
  - Training Configuration (resume settings, iterations, W&B)
  - Checkpoint Rotation Settings (opponent probability, lookback)

Usage:
  1. Configure all settings by editing the constants at the top of this file
  2. Run the script:
     python main_self_play.py
  
  For resuming training, set these constants:
  - RESUME_CHECKPOINT: Path to specific checkpoint directory
  - RESUME_FROM_LATEST_DIR: Path to results directory to auto-find latest checkpoint
  - TOTAL_ITERATIONS: Extend training to this many iterations
"""

import numpy as np
import pyspiel
import os
import glob
import csv
import random as py_random
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from hearts_env_self_play import HeartsGymEnvSelfPlay
import torch
import torch.nn as nn
from gymnasium import spaces as gym_spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from attention_model import AttentionMaskModel
import math
import wandb
from datetime import datetime


def env_creator_self_play(env_config):
    """Factory that builds a self-play OpenSpiel Hearts environment for RLlib."""
    return HeartsGymEnvSelfPlay(env_config)


register_env("hearts_env_self_play", env_creator_self_play)

# ============================================================================
# HYPERPARAMETERS - Centralized configuration
# ============================================================================
# Declare all hyperparameters once here to avoid duplication

# Training Hyperparameters
NUM_EPOCHS = 10                  # Number of epochs per training iteration
MINIBATCH_SIZE = 128             # 32 for CPU, 128 for GPU/MPS
TRAIN_BATCH_SIZE = 16000         # 12000 for CPU, 16000 for GPU/MPS
LEARNING_RATE = 5e-4             # Learning rate for optimizer
ENTROPY_COEFF = 0.2              # Entropy regularization coefficient
VF_LOSS_COEFF = 2.0              # Value function loss coefficient
CLIP_PARAM = 0.3                 # PPO clipping parameter
GRAD_CLIP = 0.5                  # Gradient clipping threshold
GAMMA = 0.99                     # Discount factor
LAMBDA = 0.95                    # GAE lambda parameter

# Environment Settings
NUM_ENV_RUNNERS = 2              # Number of parallel environment runners
NUM_ENVS_PER_RUNNER = 1          # Environments per runner

# Model Architecture
EMBED_DIM = 64                  # Embedding dimension for attention model
NUM_ATTENTION_HEADS = 2          # Number of attention heads
NUM_ATTENTION_LAYERS = 1         # Number of transformer layers
FCNET_HIDDENS = [128, 128]       # Hidden layer sizes for fully connected network

# Evaluation Settings
EVALUATION_INTERVAL = 15         # Evaluate every N iterations
EVALUATION_DURATION = 300        # Number of episodes per evaluation

# Checkpoint Settings
CHECKPOINT_FREQUENCY = 25        # Save checkpoint every N iterations
NUM_CHECKPOINTS_TO_KEEP = 5      # Number of recent checkpoints to keep

# Resource Settings
NUM_GPUS = 0                     # Ray GPU allocation (0 for MPS/CPU)
NUM_CPUS_PER_LEARNER = 1         # CPU cores per learner worker

# Training Configuration
RESUME_CHECKPOINT = None         # Path to checkpoint directory to resume training from (None for new training)
RESUME_FROM_LATEST_DIR = None    # Path to results directory to find latest checkpoint (None to disable)
TOTAL_ITERATIONS = 200           # Total number of training iterations to run
USE_WANDB = True                 # Enable Weights & Biases logging

# Checkpoint Rotation Settings
OPPONENT_PROBABILITY = 0.5       # Probability of training against older checkpoint (0.0-1.0)
MAX_CHECKPOINT_LOOKBACK = 3      # Maximum number of checkpoint iterations to look back

# ============================================================================

# Register the custom model so it can be referenced by name in the config
ModelCatalog.register_custom_model("masked_attention_model", AttentionMaskModel)

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

def find_latest_ppo_directory(base_dir=os.path.expanduser("~/ray_results")):
    """Find the most recent PPO training directory."""
    if not os.path.exists(base_dir):
        return None
    
    pattern = os.path.join(base_dir, "PPO_*")
    ppo_dirs = glob.glob(pattern)
    
    if not ppo_dirs:
        return None
    
    # Sort by modification time, most recent first
    ppo_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return ppo_dirs[0]

def get_available_checkpoints(experiment_dir, max_lookback=3):
    """Get list of available checkpoint paths, limited to recent ones.
    
    Args:
        experiment_dir: Directory containing checkpoints
        max_lookback: Maximum number of checkpoint iterations to look back
        
    Returns:
        List of checkpoint paths, sorted by iteration (oldest to newest)
    """
    if not os.path.exists(experiment_dir):
        return []
    
    checkpoint_pattern = os.path.join(experiment_dir, "checkpoint_*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return []
    
    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    # Only keep the most recent max_lookback checkpoints
    return checkpoint_dirs[-max_lookback:] if len(checkpoint_dirs) > max_lookback else checkpoint_dirs

# Note: All configuration is now done via constants at the top of this file.
# Previously used argparse, but migrated to centralized configuration.

class CheckpointRotationManager:
    """Manages rotation of older checkpoints as opponents during training.
    
    This class helps prevent overfitting by periodically introducing older
    checkpoint policies into the training loop. It maintains a pool of recent
    checkpoints (up to max_lookback iterations back) and probabilistically
    selects them to create training diversity.
    
    How it works:
    1. Every iteration, checks if older checkpoints should be used (opponent_prob)
    2. If yes, randomly selects from available recent checkpoints
    3. Loads the older checkpoint's policy weights
    4. Applies those weights to a subset of environment workers
    5. Training continues with this mixed opponent pool
    6. When switching back, restores all workers to the current policy
    
    This creates a training environment where the agent plays against both
    its current self and older versions, preventing overfitting to the
    most recent strategy.
    """
    
    def __init__(self, experiment_dir, max_lookback=3, opponent_prob=0.25):
        """
        Args:
            experiment_dir: Directory where checkpoints are saved
            max_lookback: Maximum number of checkpoint iterations to look back
            opponent_prob: Probability of using an older checkpoint as opponent
        """
        self.experiment_dir = experiment_dir
        self.max_lookback = max_lookback
        self.opponent_prob = opponent_prob
        self.checkpoint_cache = []
        self.last_opponent_checkpoint = None
        
    def update_checkpoint_list(self):
        """Update the list of available checkpoints."""
        self.checkpoint_cache = get_available_checkpoints(
            self.experiment_dir, 
            self.max_lookback
        )
        
    def should_use_opponent(self):
        """Determine if we should train against an older checkpoint this iteration."""
        # Don't use opponents if no checkpoints available yet
        if not self.checkpoint_cache:
            return False
        return py_random.random() < self.opponent_prob
    
    def get_opponent_checkpoint(self):
        """Select a random older checkpoint to use as opponent.
        
        Returns:
            Path to checkpoint directory, or None if no checkpoints available
        """
        if not self.checkpoint_cache:
            return None
        
        # Randomly select from available checkpoints
        checkpoint = py_random.choice(self.checkpoint_cache)
        self.last_opponent_checkpoint = checkpoint
        return checkpoint
    
    def get_stats(self):
        """Get statistics about checkpoint usage."""
        return {
            "num_available_checkpoints": len(self.checkpoint_cache),
            "last_opponent_checkpoint": self.last_opponent_checkpoint,
            "opponent_probability": self.opponent_prob
        }

def main():
    """Main training function."""
    # Configure PyTorch to use MPS if available (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        # Enable MPS fallback to CPU for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Tell PyTorch to prefer MPS
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        print("🚀 MPS (Apple Silicon GPU) is available and will be used for training")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   MPS available: {torch.backends.mps.is_available()}")
        print(f"   MPS built: {torch.backends.mps.is_built()}")
        print("   Note: Ray RLlib's MPS support is experimental. If you encounter issues,")
        print("         you can disable GPU training by modifying the num_gpus setting.")
    else:
        print("⚠️  MPS not available, using CPU")

    # Initialize Weights & Biases
    if USE_WANDB:
        wandb.init(
            project="hearts-ppo-selfplay_runs",
            entity="masonchoey-ucla",
            name="hearts_training_run_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            config={
            # Training hyperparameters
            "algorithm": "PPO",
            "framework": "torch",
            "num_epochs": NUM_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "entropy_coeff": ENTROPY_COEFF,
            "vf_loss_coeff": VF_LOSS_COEFF,
            "clip_param": CLIP_PARAM,
            "grad_clip": GRAD_CLIP,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            
            # Environment settings
            "env": "hearts_env_self_play",
            "num_env_runners": NUM_ENV_RUNNERS,
            "num_envs_per_env_runner": NUM_ENVS_PER_RUNNER,
            
            # Model architecture
            "model_type": "masked_attention_model",
            "fcnet_hiddens": FCNET_HIDDENS,
            "custom_model_config": {
                "embed_dim": EMBED_DIM,
                "num_attention_heads": NUM_ATTENTION_HEADS,
                "num_attention_layers": NUM_ATTENTION_LAYERS,
            },
            
            # Training settings
            "total_iterations": TOTAL_ITERATIONS,
            "evaluation_interval": EVALUATION_INTERVAL,
            "evaluation_duration": EVALUATION_DURATION,
            "checkpoint_frequency": CHECKPOINT_FREQUENCY,
            
            # Opponent rotation settings
            "opponent_prob": OPPONENT_PROBABILITY,
            "max_lookback": MAX_CHECKPOINT_LOOKBACK,
            
            # Resume settings
            "resumed_from_checkpoint": RESUME_CHECKPOINT or RESUME_FROM_LATEST_DIR is not None,
            },
            tags=["ppo", "self-play", "hearts", "transformer", "checkpoint-rotation"],
            notes="PPO training with attention-based model for Hearts card game, using checkpoint rotation to prevent overfitting",
        )
    else:
        wandb.init(mode="disabled")

    # Check if MPS is available and will be used by PyTorch
    if torch.backends.mps.is_available():
        print(f"🚀 MPS (GPU) is available! PyTorch will automatically use Apple Silicon GPU.")
        print(f"   Note: Ray resource manager doesn't track MPS, but your model will still use the GPU.")
    else:
        print(f"⚠️  MPS not available, training will use CPU.")
    
    # PPO Configuration
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("hearts_env_self_play")
        .framework("torch")
        .framework(
            framework="torch",
            torch_compile_learner=False,  # Disable torch.compile for MPS compatibility
        )
        .training(
            model={
                "custom_model": "masked_attention_model",
                "fcnet_hiddens": FCNET_HIDDENS,
                "custom_model_config": {
                    "embed_dim": EMBED_DIM,
                    "num_attention_heads": NUM_ATTENTION_HEADS,
                    "num_attention_layers": NUM_ATTENTION_LAYERS,
                }
            },
            num_epochs=NUM_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            train_batch_size=TRAIN_BATCH_SIZE,
            lr=LEARNING_RATE,
            entropy_coeff=ENTROPY_COEFF,
            vf_loss_coeff=VF_LOSS_COEFF,
            clip_param=CLIP_PARAM,
            grad_clip=GRAD_CLIP,
            use_gae=True,
            lambda_=LAMBDA,
            gamma=GAMMA,
        )
        .env_runners(
            num_env_runners=NUM_ENV_RUNNERS,
            num_envs_per_env_runner=NUM_ENVS_PER_RUNNER,
        )
        .resources(
            # Note: Ray doesn't recognize MPS as a GPU resource, but PyTorch will
            # still use MPS automatically when available. We just don't request
            # GPU resources from Ray's resource manager.
            num_gpus=NUM_GPUS,
            num_cpus_per_learner_worker=NUM_CPUS_PER_LEARNER,
        )
        .evaluation(
            evaluation_interval=EVALUATION_INTERVAL,
            evaluation_duration=EVALUATION_DURATION,
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
    
    if RESUME_CHECKPOINT:
        if os.path.exists(RESUME_CHECKPOINT):
            resume_from_checkpoint = os.path.abspath(RESUME_CHECKPOINT)
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Error: Checkpoint path does not exist: {RESUME_CHECKPOINT}")
            exit(1)
    elif RESUME_FROM_LATEST_DIR:
        # Find the experiment directory and latest checkpoint
        experiment_dir = find_experiment_checkpoint_dir(RESUME_FROM_LATEST_DIR)
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
            print(f"No experiment directory found in: {RESUME_FROM_LATEST_DIR}")
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
            target_iterations = TOTAL_ITERATIONS
            
            # Ensure we're extending beyond current iteration
            if target_iterations <= current_iteration:
                target_iterations = current_iteration + 50
                print(f"⚠️  Target iterations ({TOTAL_ITERATIONS}) <= current iteration ({current_iteration})")
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
            
            # Determine the experiment directory for saving new checkpoints
            # Extract the experiment directory from the checkpoint path
            experiment_dir = os.path.dirname(resume_from_checkpoint)
            print(f"📁 New checkpoints will be saved to: {experiment_dir}")
            
            # Initialize checkpoint rotation manager
            checkpoint_manager = CheckpointRotationManager(
                experiment_dir=experiment_dir,
                max_lookback=MAX_CHECKPOINT_LOOKBACK,
                opponent_prob=OPPONENT_PROBABILITY
            )
            print(f"🔄 Checkpoint rotation enabled: {OPPONENT_PROBABILITY*100:.0f}% chance of using older checkpoints")
            print(f"📊 Looking back up to {MAX_CHECKPOINT_LOOKBACK} checkpoint iterations")
            
            # Train manually for better control
            print(f"🏃 Starting training from iteration {current_iteration + 1} to {target_iterations}")
            
            # Track checkpoint usage for diversity
            checkpoints_pool = []  # Pool of recent checkpoint paths
            current_opponent_iter = None
            iterations_with_opponent = 0
            
            for i in range(current_iteration + 1, target_iterations + 1):
                print(f"Iteration {i}/{target_iterations}")
                
                # Every iteration, potentially rotate to an older checkpoint
                # Update available checkpoints list
                checkpoint_manager.update_checkpoint_list()
                print(f"Available checkpoints: {checkpoint_manager.checkpoint_cache}")
                
                # Decide if we should use an older checkpoint as opponent
                # This is decided probabilistically each iteration
                use_opponent = checkpoint_manager.should_use_opponent()
                
                if use_opponent:
                    opponent_checkpoint = checkpoint_manager.get_opponent_checkpoint()
                    if opponent_checkpoint and opponent_checkpoint != current_opponent_iter:
                        checkpoint_name = os.path.basename(opponent_checkpoint)
                        print(f"  🎯 Training against older checkpoint: {checkpoint_name}")
                        
                        try:
                            # Load the older checkpoint
                            opponent_algo = PPO.from_checkpoint(opponent_checkpoint)
                            opponent_weights = opponent_algo.get_policy().get_weights()
                            
                            # In self-play mode, we create training diversity by periodically
                            # synchronizing some workers to use an older checkpoint's policy.
                            # This prevents overfitting to the most recent strategy.
                            
                            # Synchronize worker policies to use the older checkpoint
                            # We do this by setting the policy weights in the workers
                            # RLlib's workers use the main policy, so we temporarily update it
                            
                            # Get list of workers
                            workers = algo.workers.remote_workers()
                            if workers:
                                # Update approximately 50% of workers to use older policy
                                num_opponent_workers = max(1, len(workers) // 2)
                                opponent_workers = workers[:num_opponent_workers]
                                
                                for worker in opponent_workers:
                                    # Set worker policy to older checkpoint
                                    worker.foreach_policy.remote(
                                        lambda p, pid: p.set_weights(opponent_weights)
                                    )
                                
                                print(f"     Updated {num_opponent_workers}/{len(workers)} workers to use {checkpoint_name}")
                            else:
                                print(f"     No remote workers available for policy mixing")
                            
                            current_opponent_iter = checkpoint_name
                            
                            # Log this event
                            wandb.log({
                                "opponent_checkpoint_active": 1,
                                "opponent_checkpoint_name": checkpoint_name,
                                "training_iteration": i,
                                "num_available_checkpoints": len(checkpoint_manager.checkpoint_cache),
                                "num_opponent_workers": num_opponent_workers if workers else 0
                            })
                            
                            del opponent_algo  # Clean up
                            
                        except Exception as e:
                            print(f"  ⚠️  Could not load opponent checkpoint: {e}")
                            current_opponent_iter = None
                    elif not opponent_checkpoint:
                        # No older checkpoints available yet
                        if i == current_iteration + 1:  # Only print on first iteration
                            print(f"  ℹ️  No older checkpoints available yet - training in pure self-play mode")
                        if current_opponent_iter:
                            wandb.log({
                                "opponent_checkpoint_active": 0,
                                "training_iteration": i
                            })
                        current_opponent_iter = None
                elif current_opponent_iter:
                    # Switching back to pure self-play
                    print(f"  🔙 Returning to pure self-play training")
                    
                    # Restore all workers to current policy
                    try:
                        current_weights = algo.get_policy().get_weights()
                        workers = algo.workers.remote_workers()
                        if workers:
                            for worker in workers:
                                worker.foreach_policy.remote(
                                    lambda p, pid: p.set_weights(current_weights)
                                )
                            print(f"     Restored all {len(workers)} workers to current policy")
                    except Exception as e:
                        print(f"  ⚠️  Could not restore worker policies: {e}")
                    
                    wandb.log({
                        "opponent_checkpoint_active": 0,
                        "training_iteration": i
                    })
                    current_opponent_iter = None
                
                result = algo.train()
                
                # Log metrics to W&B
                wandb_metrics = {
                    "training_iteration": i,
                    # Performance metrics
                    "episode_reward_mean": result.get('env_runners/episode_reward_mean', 0),
                    "episode_reward_max": result.get('env_runners/episode_reward_max', 0),
                    "episode_reward_min": result.get('env_runners/episode_reward_min', 0),
                    "episode_len_mean": result.get('env_runners/episode_len_mean', 0),
                    
                    # Training metrics
                    "policy_loss": result.get('info/learner/default_policy/learner_stats/policy_loss', 0),
                    "vf_loss": result.get('info/learner/default_policy/learner_stats/vf_loss', 0),
                    "entropy": result.get('info/learner/default_policy/learner_stats/entropy', 0),
                    "kl": result.get('info/learner/default_policy/learner_stats/kl', 0),
                    
                    # Learning dynamics
                    "curr_lr": result.get('info/learner/default_policy/learner_stats/curr_lr', 0),
                    "grad_gnorm": result.get('info/learner/default_policy/learner_stats/grad_gnorm', 0),
                    "vf_explained_var": result.get('info/learner/default_policy/learner_stats/vf_explained_var', 0),
                    
                    # System metrics
                    "num_env_steps_sampled": result.get('num_env_steps_sampled', 0),
                    "num_env_steps_trained": result.get('num_env_steps_trained', 0),
                    "time_this_iter_s": result.get('time_this_iter_s', 0),
                }
                
                # Add evaluation metrics if available
                if 'evaluation' in result:
                    wandb_metrics["eval_episode_reward_mean"] = result.get('evaluation/env_runners/episode_reward_mean', 0)
                    wandb_metrics["eval_episode_len_mean"] = result.get('evaluation/env_runners/episode_len_mean', 0)
                
                wandb.log(wandb_metrics)
                
                # Save checkpoint at regular intervals
                if i % CHECKPOINT_FREQUENCY == 0 or i == target_iterations:
                    # Create checkpoint directory path
                    checkpoint_dir = os.path.join(experiment_dir, f"checkpoint_{i:06d}")
                    checkpoint = algo.save(checkpoint_dir)
                    print(f"💾 Saved checkpoint at iteration {i}: {checkpoint}")
                    
                    # Log checkpoint to W&B
                    wandb.log({"checkpoint_saved": i})
                
                # Print some metrics
                if 'env_runners/episode_reward_mean' in result:
                    reward = result['env_runners/episode_reward_mean']
                    print(f"  Episode reward mean: {reward:.3f}")
            
            print(f"✅ Training completed! Reached {target_iterations} iterations")
            
        else:
            # Use Ray Tune for new training or unfinished trials
            print("Note: Checkpoint rotation is only available when resuming training.")
            print("Start training first, then resume with checkpoint rotation enabled.")
            
            # Configure W&B callback for Ray Tune (if enabled)
            callbacks = []
            if USE_WANDB:
                wandb_callback = WandbLoggerCallback(
                    project="hearts-ppo-selfplay_runs",
                    entity="masonchoey-ucla",
                    name="hearts_training_run_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    log_config=True,  # Log the full config
                    save_checkpoints=False,  # We handle checkpoints ourselves
                )
                callbacks.append(wandb_callback)
            
            run_config = tune.RunConfig(
                stop={"training_iteration": TOTAL_ITERATIONS},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute="env_runners/episode_reward_mean",
                    checkpoint_frequency=CHECKPOINT_FREQUENCY,
                    num_to_keep=NUM_CHECKPOINTS_TO_KEEP,
                    checkpoint_at_end=True,
                ),
                callbacks=callbacks if callbacks else None,
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
        # Finish W&B run
        wandb.finish()
        
        training_type = "resumed" if resume_from_checkpoint else "new"
        print(f"Self-Play Training session ({training_type}) ended.")
        print("-" * 70)
        
        # Print configuration instructions
        if not resume_from_checkpoint:
            print("\nTo resume this training later:")
            print("  1. Open main_self_play.py")
            print("  2. Set RESUME_FROM_LATEST_DIR to the results directory path")
            print("     OR set RESUME_CHECKPOINT to a specific checkpoint path")
            print("  3. Optionally adjust TOTAL_ITERATIONS for longer training")
            print("  4. Run: python main_self_play.py")
            print("-" * 70)

 
if __name__ == "__main__":
    main()
