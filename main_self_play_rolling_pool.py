#!/usr/bin/env python3
"""
PPO Self-Play Training with Rolling Checkpoint Pool

This script trains a PPO agent with the rolling training pool system:
- Maintains a pool of historical checkpoints
- Samples opponents from the pool during training
- Adds KL divergence penalty to prevent drift from stable policies
- Periodically evaluates against historical checkpoints
- Saves checkpoints to the pool with FIFO management

Key Features:
- All features from main_self_play_optimized.py
- Rolling checkpoint pool for robust training
- KL divergence penalty for stability
- Historical opponent sampling
- Performance tracking against past versions

Usage:
    1. First, collect existing checkpoints to the pool:
       python collect_checkpoints_to_pool.py --max-pool-size 20
    
    2. Then run training with rolling pool:
       python main_self_play_rolling_pool.py
"""

import numpy as np
import pyspiel
import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from hearts_env_self_play import HeartsGymEnvSelfPlay
from ray.rllib.models import ModelCatalog
from attention_model import AttentionMaskModel
from datetime import datetime

# Import rolling pool components
from checkpoint_pool_manager import CheckpointPoolManager
from rolling_training_callback import RollingTrainingCallback

def env_creator_self_play(env_config):
    """Factory that builds a self-play OpenSpiel Hearts environment for RLlib."""
    return HeartsGymEnvSelfPlay(env_config)


register_env("hearts_env_self_play", env_creator_self_play)

# ============================================================================
# HYPERPARAMETERS - Centralized configuration
# ============================================================================

# Training Hyperparameters (Colab Pro: 2 CPU + T4 GPU)
NUM_EPOCHS = 12                  # More epochs to maximize GPU utilization per batch
MINIBATCH_SIZE = 128             # Minibatch size for T4 efficiency
TRAIN_BATCH_SIZE = 4000          # Balanced for 2-CPU sample collection speed
LEARNING_RATE = 3e-4             # Learning rate
ENTROPY_COEFF = 0.05             # Entropy coefficient for exploration
VF_LOSS_COEFF = 1.0              # Value function loss coefficient
CLIP_PARAM = 0.2                 # PPO clipping parameter
GRAD_CLIP = 0.5                  # Gradient clipping
GAMMA = 0.99                     # Discount factor
LAMBDA = 0.95                    # GAE lambda parameter

# Environment Settings (Colab Pro: 2 CPUs total)
NUM_ENV_RUNNERS = 1              # Single runner (limited by 2 CPUs)
NUM_ENVS_PER_RUNNER = 4          # 4 parallel environments for faster sample collection
NUM_CPUS_PER_RUNNER = 1.5        # Use 1.5 CPUs for env runner (0.5 reserved for main)

# Model Architecture (T4 GPU Optimized - Larger Network)
EMBED_DIM = 128                  # Larger embedding dimension for T4
NUM_ATTENTION_HEADS = 4          # More attention heads for T4
NUM_ATTENTION_LAYERS = 2         # More transformer layers for T4
FCNET_HIDDENS = [1024, 1024, 512]  # Larger 3-layer network for T4

# Evaluation Settings
EVALUATION_INTERVAL = 15         # Regular evaluation
EVALUATION_DURATION = 300        # Thorough evaluation
EVALUATION_DURATION_UNIT = "episodes"

# Checkpoint Settings
CHECKPOINT_FREQUENCY = 25         # Save checkpoint every N iterations
NUM_CHECKPOINTS_TO_KEEP = 5      # Number of recent checkpoints to keep

# Resource Settings (Colab Pro: 2 CPUs + T4 GPU)
NUM_GPUS = 1                     # Use the T4 GPU for neural network training
NUM_CPUS_FOR_MAIN = 0.5          # Reserve 0.5 CPU for main process (2 CPUs total)

# Training Configuration
TOTAL_ITERATIONS = 250           # Total number of training iterations to run
USE_WANDB = True                 # Enable Weights & Biases logging
USE_MIXED_PRECISION = False      # Enable mixed precision training for T4 Tensor cores

# ============================================================================
# ROLLING POOL CONFIGURATION
# ============================================================================

# Pool Settings
POOL_DIR = "models/pool"                    # Directory for checkpoint pool
MAX_POOL_SIZE = 20                          # Maximum checkpoints in pool (FIFO)
CHECKPOINT_SAVE_TO_POOL_FREQUENCY = 10      # Save to pool every N iterations
MAX_CACHED_POLICIES = 5                     # Maximum opponent policies to cache

# KL Divergence Settings
KL_PENALTY_BETA = 0.01                      # Weight for KL divergence penalty
                                            # Higher = stronger regularization
                                            # Lower = more policy freedom

# Opponent Sampling Settings
OPPONENT_SAMPLE_METHOD = "random"           # 'random' or 'performance_weighted'
EXCLUDE_RECENT_OPPONENTS = 1                # Exclude N most recent checkpoints
OPPONENT_SAMPLE_FREQUENCY = 10              # Sample new opponents every N episodes

# Evaluation Settings
EVALUATE_VS_POOL_FREQUENCY = 50             # Evaluate vs pool every N iterations
POOL_EVALUATION_OPPONENTS = 5               # Number of pool opponents to evaluate against
POOL_EVALUATION_GAMES = 50                  # Games per opponent during evaluation

# Device Settings
DEVICE = "cuda" if NUM_GPUS > 0 else "cpu"  # Device for opponent policies

# ============================================================================

# Register the custom model so it can be referenced by name in the config
ModelCatalog.register_custom_model("masked_attention_model", AttentionMaskModel)


def main():
    """Main training function with rolling checkpoint pool."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=NUM_CPUS_FOR_MAIN + NUM_ENV_RUNNERS * NUM_CPUS_PER_RUNNER,
            num_gpus=NUM_GPUS,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        print("✅ Ray initialized successfully")
    
    # Initialize checkpoint pool manager
    print("\n" + "=" * 80)
    print("INITIALIZING ROLLING CHECKPOINT POOL")
    print("=" * 80)
    
    pool_manager = CheckpointPoolManager(
        pool_dir=POOL_DIR,
        max_pool_size=MAX_POOL_SIZE
    )
    
    pool_stats = pool_manager.get_pool_statistics()
    print(f"Pool Status:")
    print(f"  Current size: {pool_stats['pool_size']}/{pool_stats['max_pool_size']}")
    print(f"  Total added: {pool_stats['total_added']}")
    print(f"  Total evicted: {pool_stats['total_evicted']}")
    
    if pool_stats['pool_size'] == 0:
        print("\n⚠️  WARNING: Checkpoint pool is empty!")
        print("   Consider running: python collect_checkpoints_to_pool.py")
        print("   Training will proceed but without opponent diversity until pool is populated.")
    
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
    
    # Add pool configuration to the config
    # This will be accessible by the callback
    ppo_config["pool_config"] = {
        "pool_dir": POOL_DIR,
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
    print("🚀 PPO SELF-PLAY TRAINING WITH ROLLING CHECKPOINT POOL")
    print("=" * 80)
    print("Training Configuration:")
    print(f"  Environment Runners: {NUM_ENV_RUNNERS} × {NUM_ENVS_PER_RUNNER} = {NUM_ENV_RUNNERS * NUM_ENVS_PER_RUNNER} parallel environments")
    print(f"  Batch Size: {TRAIN_BATCH_SIZE:,} samples")
    print(f"  Network: {' → '.join(map(str, FCNET_HIDDENS))}")
    print(f"  Attention: {NUM_ATTENTION_HEADS} heads, {NUM_ATTENTION_LAYERS} layers, {EMBED_DIM} dim")
    print(f"  GPU: {NUM_GPUS} GPU(s)")
    
    print("\nRolling Pool Configuration:")
    print(f"  Pool Size: {MAX_POOL_SIZE} checkpoints (FIFO)")
    print(f"  Save Frequency: Every {CHECKPOINT_SAVE_TO_POOL_FREQUENCY} iterations")
    print(f"  KL Penalty Beta: {KL_PENALTY_BETA}")
    print(f"  Opponent Sampling: {OPPONENT_SAMPLE_METHOD}")
    print(f"  Max Cached Opponents: {MAX_CACHED_POLICIES}")
    
    if USE_WANDB:
        print(f"\n  W&B Logging: ENABLED (project: hearts-ppo-rolling-pool)")
    print("=" * 80)

    try:
        # Configure W&B callback for Ray Tune (if enabled)
        callbacks = []
        if USE_WANDB:
            wandb_callback = WandbLoggerCallback(
                project="hearts-ppo-rolling-pool",
                entity="masonchoey-ucla",
                api_key="",
                log_config=True,
                save_checkpoints=False,
            )
            callbacks.append(wandb_callback)
        
        # Use Ray Tune for training
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
        
        print("\n" + "=" * 80)
        print("✅ Rolling Pool Training completed successfully!")
        print("=" * 80)
        
        # Print final pool statistics
        final_stats = pool_manager.get_pool_statistics()
        print("Final Pool Statistics:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        print("\nBest trial results:")
        best_result = results.get_best_result()
        print(f"  Reward: {best_result.metrics.get('env_runners/episode_reward_mean', 'N/A')}")
        print(f"  Iteration: {best_result.metrics.get('training_iteration', 'N/A')}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Training interrupted by user (Ctrl+C)")
        print("Checkpoints have been saved to the pool.")
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ Training stopped due to an error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()
            print("✅ Ray shutdown successfully")
        print("🏁 Rolling pool training session ended.")
        print("=" * 80)


if __name__ == "__main__":
    main()

