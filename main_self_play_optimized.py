#!/usr/bin/env python3
"""
PPO Self-Play Training Script - Optimized for Google Colab Pro (2 CPU + T4 GPU)

This script trains a PPO agent to play Hearts using self-play, optimized for
Google Colab Pro with 2 CPUs and NVIDIA T4 GPU (16GB VRAM).

Key Features:
- Centralized configuration: All hyperparameters and settings declared at the top
- Colab Pro optimizations: Efficient CPU usage, balanced batch sizes for 2-CPU constraint
- W&B integration: Comprehensive experiment tracking and logging
- Mixed precision training: Optional Tensor core acceleration

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
  - Training Configuration (iterations, W&B, mixed precision)

Usage:
  1. Configure all settings by editing the constants at the top of this file
  2. Run the script:
     python main_self_play_optimized.py
"""

import numpy as np
import pyspiel
import os
import glob
import csv
import ray
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
from datetime import datetime

def env_creator_self_play(env_config):
    """Factory that builds a self-play OpenSpiel Hearts environment for RLlib."""
    return HeartsGymEnvSelfPlay(env_config)


register_env("hearts_env_self_play", env_creator_self_play)

# ============================================================================
# HYPERPARAMETERS - Centralized configuration for Colab Pro (2 CPU + T4 GPU)
# ============================================================================
# Declare all hyperparameters once here to avoid duplication

# Training Hyperparameters (Colab Pro: 2 CPU + T4 GPU)
NUM_EPOCHS = 12                  # More epochs to maximize GPU utilization per batch
MINIBATCH_SIZE = 128             # Minibatch size for T4 efficiency
TRAIN_BATCH_SIZE = 4000          # Balanced for 2-CPU sample collection speed
LEARNING_RATE = 5e-4             # Learning rate
ENTROPY_COEFF = 0.2              # Entropy coefficient for exploration
VF_LOSS_COEFF = 2.0              # Value function loss coefficient
CLIP_PARAM = 0.3                 # PPO clipping parameter
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

# Register the custom model so it can be referenced by name in the config
ModelCatalog.register_custom_model("masked_attention_model", AttentionMaskModel)


def main():
    """Main training function optimized for Google Colab Pro (2 CPU + T4 GPU)."""
    # Initialize Ray (important for Google Colab environments)
    if not ray.is_initialized():
        ray.init(
            num_cpus=NUM_CPUS_FOR_MAIN + NUM_ENV_RUNNERS * NUM_CPUS_PER_RUNNER,
            num_gpus=NUM_GPUS,
            ignore_reinit_error=True,
            include_dashboard=False,  # Disable dashboard for Colab
        )
        print("✅ Ray initialized successfully")
    
    # Note: W&B initialization is handled by WandbLoggerCallback in Ray Tune
    # No need to call wandb.init() here as it will cause conflicts

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
            # T4-optimized training hyperparameters
            num_epochs=NUM_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            train_batch_size=TRAIN_BATCH_SIZE,
            lr=LEARNING_RATE,
            lr_schedule=None,          # Constant learning rate
            entropy_coeff=ENTROPY_COEFF,
            vf_loss_coeff=VF_LOSS_COEFF,
            clip_param=CLIP_PARAM,
            grad_clip=GRAD_CLIP,
            use_gae=True,
            lambda_=LAMBDA,
            gamma=GAMMA,
            # Enable mixed precision if configured (T4 Tensor cores)
            **({"mixed_precision": True} if USE_MIXED_PRECISION else {}),
        )
        .env_runners(
            # Scale environment runners based on available CPUs
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
        .debugging(
            log_level="INFO"
        )
    )

    print("=" * 80)
    print("🚀 GOOGLE COLAB PRO PPO Self-Play Training (2 CPU + T4 GPU)")
    print("=" * 80)
    print("Colab Pro Specifications:")
    print("  • 2 vCPUs")
    print("  • ~13-25 GB RAM")
    print("  • T4 GPU: 16 GB GDDR6 Memory, 2,560 CUDA Cores, 320 Tensor Cores")
    print()
    print("Optimized Configuration:")
    print(f"  • Environment Runners: {NUM_ENV_RUNNERS} runner × {NUM_ENVS_PER_RUNNER} envs = {NUM_ENV_RUNNERS * NUM_ENVS_PER_RUNNER} parallel environments")
    print(f"  • CPU Usage: {NUM_CPUS_FOR_MAIN} (main) + {NUM_ENV_RUNNERS} × {NUM_CPUS_PER_RUNNER} (runners) = {NUM_CPUS_FOR_MAIN + NUM_ENV_RUNNERS * NUM_CPUS_PER_RUNNER} CPUs")
    print(f"  • GPU Usage: {NUM_GPUS} T4 GPU for neural network training & inference")
    print(f"  • Batch Size: {TRAIN_BATCH_SIZE:,} samples (balanced for 2-CPU collection)")
    print(f"  • Network Size: {' → '.join(map(str, FCNET_HIDDENS))} (3-layer deep network)")
    print(f"  • Training Epochs: {NUM_EPOCHS} (maximize GPU work per batch)")
    print(f"  • Minibatch Size: {MINIBATCH_SIZE}")
    print(f"  • Attention Heads: {NUM_ATTENTION_HEADS} | Layers: {NUM_ATTENTION_LAYERS} | Embed Dim: {EMBED_DIM}")
    if USE_MIXED_PRECISION:
        print("  • Mixed Precision: ENABLED (using Tensor cores)")
    else:
        print("  • Mixed Precision: DISABLED (set USE_MIXED_PRECISION = True to enable)")
    if USE_WANDB:
        print(f"  • W&B Logging: ENABLED (project: hearts-ppo-t4gpu)")
    else:
        print("  • W&B Logging: DISABLED")
    print("=" * 80)

    try:
        # Configure W&B callback for Ray Tune (if enabled)
        callbacks = []
        if USE_WANDB:
            wandb_callback = WandbLoggerCallback(
                project="hearts-ppo-t4gpu",
                entity="masonchoey-ucla",
                api_key="",
                log_config=True,  # Log the full config
                save_checkpoints=False,  # We handle checkpoints ourselves
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
        print("✅ Colab Pro Self-Play Training completed successfully!")
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
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()
            print("✅ Ray shutdown successfully")
        # W&B cleanup is handled automatically by WandbLoggerCallback
        print("🏁 Colab Pro training session ended.")
        print("💡 Pro Tip: Use '!nvidia-smi' in Colab to monitor GPU utilization during training")
        print("=" * 80)


if __name__ == "__main__":
    main()
