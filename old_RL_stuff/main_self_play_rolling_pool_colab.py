#!/usr/bin/env python3
"""
PPO Self-Play Training with Rolling Pool - Google Colab Edition

This script trains a PPO agent with rolling checkpoint pool optimized for Google Colab.
Stores all checkpoints and data in Google Drive for persistence across sessions.

Key Features:
- Google Drive integration for checkpoint persistence
- Rolling checkpoint pool (20 checkpoints)
- KL divergence penalty for stability
- Automatic session recovery
- Optimized for Colab Pro (T4 GPU)

Usage in Colab:
    1. Upload this file and supporting files to Colab
    2. Run setup cell to mount Drive and install dependencies
    3. Run training cell
    
Quick Setup:
    !wget https://raw.githubusercontent.com/[your-repo]/colab_utils.py
    !wget https://raw.githubusercontent.com/[your-repo]/checkpoint_pool_manager.py
    # ... (or upload files manually)
    
    %run main_self_play_rolling_pool_colab.py
"""

import numpy as np
import pyspiel
import os
import sys
from pathlib import Path
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from hearts_env_self_play import HeartsGymEnvSelfPlay
from ray.rllib.models import ModelCatalog
from attention_model import AttentionMaskModel
from datetime import datetime

# Import Colab utilities
from colab_utils import (
    is_colab,
    setup_colab_environment,
    check_colab_resources,
    get_checkpoint_storage_path,
    save_training_state,
    load_training_state,
    install_colab_dependencies
)

# Import rolling pool components
from checkpoint_pool_manager import CheckpointPoolManager
from rolling_training_callback import RollingTrainingCallback


def env_creator_self_play(env_config):
    """Factory that builds a self-play OpenSpiel Hearts environment for RLlib."""
    return HeartsGymEnvSelfPlay(env_config)


class HeartsCallbacks(DefaultCallbacks):
    """Custom callbacks for Hearts training to properly manage model state.
    
    This callback ensures that the attention model's history buffer is properly
    reset between episodes, preventing the model from seeing observations from
    previous games in the current game.
    """
    
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Called at the beginning of each episode to reset model history."""
        # Reset history buffer for all policies that have the reset_history method
        for policy_id, policy in policies.items():
            if hasattr(policy, 'model') and hasattr(policy.model, 'reset_history'):
                policy.model.reset_history()

register_env("hearts_env_self_play", env_creator_self_play)

# ============================================================================
# HYPERPARAMETERS - Optimized for Google Colab Pro (2 CPU + T4 GPU)
# ============================================================================

# Training Hyperparameters
NUM_EPOCHS = 12                  # More epochs to maximize GPU utilization
MINIBATCH_SIZE = 128             # Minibatch size for T4 efficiency
TRAIN_BATCH_SIZE = 4000          # Balanced for 2-CPU sample collection
LEARNING_RATE = 3e-4             # Learning rate
ENTROPY_COEFF = 0.05             # Entropy coefficient
VF_LOSS_COEFF = 1.0              # Value function loss coefficient
CLIP_PARAM = 0.2                 # PPO clipping parameter
GRAD_CLIP = 0.5                  # Gradient clipping
GAMMA = 0.99                     # Discount factor
LAMBDA = 0.95                    # GAE lambda parameter

# Environment Settings (Colab Pro: 2 CPUs)
NUM_ENV_RUNNERS = 1              # Single runner
NUM_ENVS_PER_RUNNER = 4          # 4 parallel environments
NUM_CPUS_PER_RUNNER = 1.5        # 1.5 CPUs for runner

# Model Architecture (T4 GPU Optimized)
EMBED_DIM = 128                  # Embedding dimension
NUM_ATTENTION_HEADS = 4          # Attention heads
NUM_ATTENTION_LAYERS = 2         # Transformer layers
FCNET_HIDDENS = [1024, 1024, 512]  # 3-layer network

# Evaluation Settings
EVALUATION_INTERVAL = 15         # Evaluate every N iterations
EVALUATION_DURATION = 300        # Evaluation episodes
EVALUATION_DURATION_UNIT = "episodes"

# Checkpoint Settings (Colab-specific)
CHECKPOINT_FREQUENCY = 25        # Save to Ray every N iterations
NUM_CHECKPOINTS_TO_KEEP = 3      # Keep only 3 in Ray (save space)

# Resource Settings
NUM_GPUS = 1                     # Use T4 GPU
NUM_CPUS_FOR_MAIN = 0.5          # Reserve 0.5 CPU for main

# Training Configuration
TOTAL_ITERATIONS = 250           # Total training iterations
USE_WANDB = True                 # Enable W&B logging
USE_MIXED_PRECISION = False      # Mixed precision training

# ============================================================================
# GOOGLE DRIVE / ROLLING POOL CONFIGURATION
# ============================================================================

# Google Drive Settings
DRIVE_PROJECT_NAME = "Hearts_RL"            # Folder name in MyDrive
AUTO_MOUNT_DRIVE = True                     # Auto-mount Drive

# Pool Settings (stored in Drive)
MAX_POOL_SIZE = 20                          # Maximum checkpoints in pool
CHECKPOINT_SAVE_TO_POOL_FREQUENCY = 10      # Save to pool every N iterations
MAX_CACHED_POLICIES = 3                     # Fewer cached (save memory)

# KL Divergence Settings
KL_PENALTY_BETA = 0.01                      # KL penalty weight

# Opponent Sampling Settings
OPPONENT_SAMPLE_METHOD = "random"           # 'random' or 'performance_weighted'
EXCLUDE_RECENT_OPPONENTS = 1                # Exclude N most recent
OPPONENT_SAMPLE_FREQUENCY = 10              # Sample every N episodes

# Device Settings
DEVICE = "cuda" if NUM_GPUS > 0 else "cpu"

# Session Recovery
AUTO_SAVE_STATE = True                      # Auto-save training state
STATE_SAVE_FREQUENCY = 5                    # Save state every N iterations

# ============================================================================

# Register the custom model
ModelCatalog.register_custom_model("masked_attention_model", AttentionMaskModel)


def setup_colab():
    """
    Set up Google Colab environment for training.
    
    Returns:
        Tuple of (project_path, pool_path)
    """
    print("\n" + "=" * 80)
    print("🚀 GOOGLE COLAB SETUP")
    print("=" * 80)
    
    # Check if in Colab
    if not is_colab():
        print("⚠️  WARNING: Not running in Google Colab")
        print("   This script is optimized for Colab with Google Drive")
        print("   For local training, use main_self_play_rolling_pool.py instead")
        
        # Use local paths as fallback
        project_path = Path.cwd()
        pool_path = project_path / "models" / "pool"
        pool_path.mkdir(parents=True, exist_ok=True)
        return project_path, pool_path
    
    # Check resources
    check_colab_resources()
    
    # Set up environment (mounts Drive and creates directories)
    project_path, pool_path = setup_colab_environment(
        project_name=DRIVE_PROJECT_NAME,
        mount_drive=AUTO_MOUNT_DRIVE
    )
    
    return project_path, pool_path


def main():
    """Main training function for Google Colab."""
    
    # Set up Colab environment
    project_path, pool_path = setup_colab()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=NUM_CPUS_FOR_MAIN + NUM_ENV_RUNNERS * NUM_CPUS_PER_RUNNER,
            num_gpus=NUM_GPUS,
            ignore_reinit_error=True,
            include_dashboard=False,  # Dashboard doesn't work well in Colab
        )
        print("✅ Ray initialized successfully")
    
    # Initialize checkpoint pool manager (with Drive path)
    print("\n" + "=" * 80)
    print("INITIALIZING ROLLING CHECKPOINT POOL (GOOGLE DRIVE)")
    print("=" * 80)
    
    pool_manager = CheckpointPoolManager(
        pool_dir=str(pool_path),
        max_pool_size=MAX_POOL_SIZE
    )
    
    pool_stats = pool_manager.get_pool_statistics()
    print(f"Pool Status:")
    print(f"  Location: {pool_path}")
    print(f"  Current size: {pool_stats['pool_size']}/{pool_stats['max_pool_size']}")
    
    if pool_stats['pool_size'] == 0:
        print("\n⚠️  WARNING: Checkpoint pool is empty!")
        print("   Training will proceed, but consider collecting checkpoints")
        print("   from past runs to populate the pool.")
    else:
        print(f"  ✅ Pool contains {pool_stats['pool_size']} checkpoints")
    
    # Load previous training state (if resuming)
    previous_state = load_training_state(project_path)
    start_iteration = previous_state.get('last_iteration', 0)
    
    if start_iteration > 0:
        print(f"\n🔄 RESUMING from iteration {start_iteration}")
        print("   Previous training state loaded from Drive")
    
    # Get checkpoint storage path in Drive
    checkpoint_storage = get_checkpoint_storage_path(project_path)
    
    # PPO Configuration with Rolling Pool
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("hearts_env_self_play")
        .framework("torch")
        .resources(
            num_gpus=NUM_GPUS,
            num_cpus_for_main_process=NUM_CPUS_FOR_MAIN,
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
            lr_schedule=None,
            entropy_coeff=ENTROPY_COEFF,
            vf_loss_coeff=VF_LOSS_COEFF,
            clip_param=CLIP_PARAM,
            grad_clip=GRAD_CLIP,
            use_gae=True,
            lambda_=LAMBDA,
            gamma=GAMMA,
            **({"mixed_precision": True} if USE_MIXED_PRECISION else {}),
        )
        .env_runners(
            num_env_runners=NUM_ENV_RUNNERS,
            num_envs_per_env_runner=NUM_ENVS_PER_RUNNER,
            num_cpus_per_env_runner=NUM_CPUS_PER_RUNNER,
        )
        .callbacks(HeartsCallbacks)
        .evaluation(
            evaluation_interval=EVALUATION_INTERVAL,
            evaluation_duration=EVALUATION_DURATION,
            evaluation_duration_unit=EVALUATION_DURATION_UNIT,
            evaluation_config={"explore": False}
        )
        .callbacks(RollingTrainingCallback)
        .debugging(
            log_level="INFO"
        )
    )
    
    # Add pool configuration
    ppo_config["pool_config"] = {
        "pool_dir": str(pool_path),
        "max_pool_size": MAX_POOL_SIZE,
        "checkpoint_save_frequency": CHECKPOINT_SAVE_TO_POOL_FREQUENCY,
        "max_cached_policies": MAX_CACHED_POLICIES,
        "kl_penalty_beta": KL_PENALTY_BETA,
        "opponent_sample_method": OPPONENT_SAMPLE_METHOD,
        "exclude_recent_opponents": EXCLUDE_RECENT_OPPONENTS,
        "opponent_sample_frequency": OPPONENT_SAMPLE_FREQUENCY,
        "device": DEVICE,
    }

    print("\n" + "=" * 80)
    print("🚀 PPO ROLLING POOL TRAINING - GOOGLE COLAB EDITION")
    print("=" * 80)
    print("Colab Configuration:")
    print(f"  GPU: {NUM_GPUS} T4 GPU")
    print(f"  CPUs: {NUM_ENV_RUNNERS * NUM_CPUS_PER_RUNNER + NUM_CPUS_FOR_MAIN}")
    print(f"  Environments: {NUM_ENV_RUNNERS * NUM_ENVS_PER_RUNNER}")
    print(f"  Batch Size: {TRAIN_BATCH_SIZE:,}")
    
    print("\nGoogle Drive Storage:")
    print(f"  Project: {project_path}")
    print(f"  Pool: {pool_path}")
    print(f"  Checkpoints: {checkpoint_storage}")
    
    print("\nRolling Pool:")
    print(f"  Pool Size: {MAX_POOL_SIZE} checkpoints")
    print(f"  Current: {pool_stats['pool_size']}/{MAX_POOL_SIZE}")
    print(f"  KL Penalty Beta: {KL_PENALTY_BETA}")
    print(f"  Save Frequency: Every {CHECKPOINT_SAVE_TO_POOL_FREQUENCY} iterations")
    
    if USE_WANDB:
        print(f"\n  W&B Logging: ENABLED (project: hearts-ppo-colab-rolling-pool)")
    print("=" * 80)

    try:
        # Configure W&B callback
        callbacks = []
        if USE_WANDB:
            wandb_callback = WandbLoggerCallback(
                project="hearts-ppo-colab-rolling-pool",
                entity="masonchoey-ucla",
                api_key="",  # Will use WANDB_API_KEY env var if set
                log_config=True,
                save_checkpoints=False,  # We manage checkpoints via Drive
            )
            callbacks.append(wandb_callback)
        
        # Configure Ray Tune with Drive storage
        run_config = tune.RunConfig(
            name="colab_rolling_pool",
            local_dir=str(checkpoint_storage),  # Store in Drive
            stop={"training_iteration": TOTAL_ITERATIONS},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="env_runners/episode_reward_mean",
                checkpoint_frequency=CHECKPOINT_FREQUENCY,
                num_to_keep=NUM_CHECKPOINTS_TO_KEEP,  # Keep fewer in Ray
                checkpoint_at_end=True,
            ),
            callbacks=callbacks if callbacks else None,
        )
        
        # Create tuner
        tuner = tune.Tuner(
            "PPO",
            param_space=ppo_config,
            run_config=run_config
        )
        
        # Run training
        print("\n🚀 Starting training...")
        print("💡 TIP: Training state is automatically saved to Drive")
        print("   You can safely disconnect and resume later\n")
        
        results = tuner.fit()
        
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        
        # Print final results
        best_result = results.get_best_result()
        print("\nBest Results:")
        print(f"  Reward: {best_result.metrics.get('env_runners/episode_reward_mean', 'N/A')}")
        print(f"  Iteration: {best_result.metrics.get('training_iteration', 'N/A')}")
        
        # Print final pool statistics
        final_stats = pool_manager.get_pool_statistics()
        print("\nFinal Pool Statistics:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        # Save final training state
        if AUTO_SAVE_STATE:
            save_training_state(
                project_path,
                iteration=best_result.metrics.get('training_iteration', 0),
                metrics={
                    'reward_mean': best_result.metrics.get('env_runners/episode_reward_mean'),
                    'pool_size': final_stats['pool_size']
                }
            )
        
        print("\n💾 All data saved to Google Drive:")
        print(f"   {project_path}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Training interrupted by user")
        print("💾 Checkpoints and pool are saved in Google Drive")
        print("   You can resume training by running this script again")
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ Training error: {e}")
        print("\n💾 Partial progress saved in Google Drive")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()
            print("\n✅ Ray shutdown successfully")
        
        print("\n" + "=" * 80)
        print("🏁 Colab training session ended")
        print(f"📂 All data in Drive: {project_path}")
        print("=" * 80)


def print_colab_tips():
    """Print helpful tips for Colab users."""
    print("\n" + "=" * 80)
    print("💡 GOOGLE COLAB TIPS")
    print("=" * 80)
    print("1. Connect to GPU runtime:")
    print("   Runtime → Change runtime type → GPU (T4)")
    print()
    print("2. Keep session alive:")
    print("   Click window periodically or use Colab Pro")
    print()
    print("3. Monitor GPU usage:")
    print("   !nvidia-smi")
    print()
    print("4. Check Drive space:")
    print("   !df -h /content/drive/MyDrive")
    print()
    print("5. Resume training:")
    print("   Just run this script again - it auto-resumes")
    print()
    print("6. View checkpoints:")
    print("   ls /content/drive/MyDrive/Hearts_RL/models/pool/")
    print("=" * 80)


if __name__ == "__main__":
    # Print helpful tips
    if is_colab():
        print_colab_tips()
    
    # Run training
    main()

