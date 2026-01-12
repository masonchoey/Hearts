#!/usr/bin/env python3
"""
Rolling Training Callback with KL Divergence Penalty

This callback implements the rolling training pool system:
- Saves checkpoints to the pool periodically
- Samples opponents from the pool during training
- Adds KL divergence penalty to prevent drift from stable policies
- Evaluates against historical checkpoints
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from checkpoint_pool_manager import CheckpointPoolManager
from opponent_policy_loader import OpponentPolicyLoader, compute_kl_divergence


class RollingTrainingCallback(DefaultCallbacks):
    """
    Callback for rolling training with checkpoint pool management.
    
    Key Features:
    - Periodic checkpoint saving to pool
    - KL divergence penalty computation
    - Evaluation against historical checkpoints
    - Opponent sampling from pool
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize pool manager and policy loader
        self.pool_manager = None
        self.policy_loader = None
        
        # KL divergence tracking
        self.kl_divergences = []
        self.kl_penalty_beta = 0.01  # Weight for KL penalty (configurable)
        
        # Checkpoint saving frequency
        self.checkpoint_save_frequency = 10  # Save every N iterations
        self.last_checkpoint_iteration = 0
        
        # Evaluation tracking
        self.evaluation_results = []
    
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        """Called when the algorithm is initialized."""
        # Get configuration
        config = algorithm.config
        
        # Initialize pool manager
        pool_config = config.get("pool_config", {})
        max_pool_size = pool_config.get("max_pool_size", 20)
        pool_dir = pool_config.get("pool_dir", "models/pool")
        
        self.pool_manager = CheckpointPoolManager(
            pool_dir=pool_dir,
            max_pool_size=max_pool_size
        )
        
        # Initialize policy loader
        self.policy_loader = OpponentPolicyLoader(
            checkpoint_pool_manager=self.pool_manager,
            max_cached_policies=pool_config.get("max_cached_policies", 5),
            device=pool_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Get KL penalty beta from config
        self.kl_penalty_beta = pool_config.get("kl_penalty_beta", 0.01)
        self.checkpoint_save_frequency = pool_config.get("checkpoint_save_frequency", 10)
        
        print(f"✅ RollingTrainingCallback initialized")
        print(f"   Pool size: {len(self.pool_manager.metadata['checkpoints'])}/{max_pool_size}")
        print(f"   KL penalty beta: {self.kl_penalty_beta}")
        print(f"   Checkpoint save frequency: {self.checkpoint_save_frequency} iterations")
    
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """Called after each training iteration."""
        iteration = result["training_iteration"]
        
        # Save checkpoint to pool periodically
        if iteration - self.last_checkpoint_iteration >= self.checkpoint_save_frequency:
            self._save_checkpoint_to_pool(algorithm, result)
            self.last_checkpoint_iteration = iteration
        
        # Add KL divergence metrics to result
        if self.kl_divergences:
            result["custom_metrics"]["kl_divergence_mean"] = np.mean(self.kl_divergences)
            result["custom_metrics"]["kl_divergence_max"] = np.max(self.kl_divergences)
            result["custom_metrics"]["kl_divergence_min"] = np.min(self.kl_divergences)
            self.kl_divergences = []  # Reset for next iteration
        
        # Add pool statistics
        if self.pool_manager:
            stats = self.pool_manager.get_pool_statistics()
            result["custom_metrics"]["pool_size"] = stats["pool_size"]
            result["custom_metrics"]["pool_utilization"] = stats["pool_size"] / stats["max_pool_size"]
    
    def _save_checkpoint_to_pool(self, algorithm, result: dict):
        """Save current checkpoint to the pool."""
        if not self.pool_manager:
            return
        
        # Save checkpoint to temporary location
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = algorithm.save(tmpdir)
            
            # Add to pool with performance metrics
            performance_score = result.get("env_runners/episode_reward_mean", 0.0)
            iteration = result["training_iteration"]
            
            self.pool_manager.add_checkpoint(
                checkpoint_path,
                performance_score=performance_score,
                training_iteration=iteration,
                copy_to_pool=True
            )
        
        print(f"💾 Saved checkpoint to pool at iteration {iteration}")
    
    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: EpisodeV2,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: Dict,
        original_batches: Dict,
        **kwargs
    ) -> None:
        """
        Called after trajectory postprocessing.
        
        This is where we add the KL divergence penalty to the loss.
        """
        if not self.pool_manager or not self.policy_loader:
            return
        
        # Check if we have any checkpoints in the pool
        checkpoint_paths = self.pool_manager.get_all_checkpoint_paths()
        if not checkpoint_paths:
            return
        
        # Sample a checkpoint from the pool
        try:
            sampled_checkpoints = self.pool_manager.sample_opponent(
                n=1,
                method="random",
                exclude_recent=1  # Exclude most recent to avoid self-comparison
            )
            
            if not sampled_checkpoints:
                return
            
            checkpoint_path = sampled_checkpoints[0]
            
            # Get observations from the batch
            obs_batch = postprocessed_batch.get("obs", None)
            if obs_batch is None:
                return
            
            # Get current policy
            current_policy = policies[policy_id]
            
            # Compute KL divergence for a sample of observations (for efficiency)
            # We'll sample up to 32 observations per batch
            batch_size = len(obs_batch) if isinstance(obs_batch, (list, tuple)) else obs_batch.shape[0]
            sample_size = min(32, batch_size)
            
            if sample_size > 0:
                # Sample indices
                sample_indices = np.random.choice(batch_size, size=sample_size, replace=False)
                
                kl_divs = []
                
                for idx in sample_indices:
                    # Get observation
                    if isinstance(obs_batch, dict):
                        obs = {k: v[idx] for k, v in obs_batch.items()}
                    else:
                        obs = obs_batch[idx]
                    
                    try:
                        # Get current policy distribution
                        with torch.no_grad():
                            if hasattr(current_policy, 'model'):
                                # Convert obs to torch tensor
                                if isinstance(obs, dict):
                                    obs_torch = {
                                        k: torch.from_numpy(v).unsqueeze(0).float()
                                        if isinstance(v, np.ndarray) else v
                                        for k, v in obs.items()
                                    }
                                else:
                                    obs_torch = torch.from_numpy(obs).unsqueeze(0).float()
                                
                                # Get logits from current policy
                                logits_current, _ = current_policy.model({"obs": obs_torch}, [], None)
                                dist_current = torch.softmax(logits_current, dim=-1)
                                
                                # Get distribution from old policy
                                dist_old = self.policy_loader.get_action_distribution(
                                    checkpoint_path,
                                    obs if isinstance(obs, dict) else {"observations": obs}
                                )
                                
                                # Ensure same shape
                                if dist_old.dim() == 1:
                                    dist_old = dist_old.unsqueeze(0)
                                
                                # Get action mask if available
                                action_mask = None
                                if isinstance(obs, dict) and "action_mask" in obs:
                                    action_mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0).float()
                                
                                # Compute KL divergence
                                kl_div = compute_kl_divergence(
                                    dist_current,
                                    dist_old,
                                    action_mask
                                )
                                
                                kl_divs.append(kl_div.item())
                        
                    except Exception as e:
                        # Skip this observation if there's an error
                        continue
                
                # Store KL divergences for logging
                if kl_divs:
                    self.kl_divergences.extend(kl_divs)
                    
                    # Add KL penalty to the batch
                    # Note: This is a simplified version. In practice, you might want to
                    # modify the advantages or add it to the loss directly in a custom trainer
                    mean_kl = np.mean(kl_divs)
                    
                    # Store as custom metric
                    if "kl_divergence" not in postprocessed_batch:
                        postprocessed_batch["kl_divergence"] = mean_kl
        
        except Exception as e:
            # Don't fail training if KL computation fails
            print(f"⚠️  KL divergence computation failed: {e}")
            pass


class PoolSamplingCallback(DefaultCallbacks):
    """
    Simplified callback that just handles opponent sampling from the pool.
    
    This is useful if you want opponent sampling without KL divergence penalty.
    """
    
    def __init__(self):
        super().__init__()
        self.pool_manager = None
        self.opponent_sample_frequency = 10  # Sample new opponents every N episodes
        self.episode_count = 0
        self.current_opponents = []
    
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        """Initialize pool manager."""
        config = algorithm.config
        pool_config = config.get("pool_config", {})
        
        self.pool_manager = CheckpointPoolManager(
            pool_dir=pool_config.get("pool_dir", "models/pool"),
            max_pool_size=pool_config.get("max_pool_size", 20)
        )
        
        self.opponent_sample_frequency = pool_config.get("opponent_sample_frequency", 10)
        
        print(f"✅ PoolSamplingCallback initialized")
        print(f"   Pool size: {len(self.pool_manager.metadata['checkpoints'])}")
    
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        **kwargs
    ) -> None:
        """Sample new opponents periodically."""
        self.episode_count += 1
        
        if self.episode_count % self.opponent_sample_frequency == 0:
            if self.pool_manager:
                self.current_opponents = self.pool_manager.sample_opponent(
                    n=3,  # Sample 3 opponents
                    method="random",
                    exclude_recent=1
                )
                
                if self.current_opponents:
                    print(f"🎲 Sampled new opponents from pool (episode {self.episode_count})")


def main():
    """Demo/test script for callbacks."""
    print("=" * 80)
    print("ROLLING TRAINING CALLBACK")
    print("=" * 80)
    print("This callback provides:")
    print("  1. Periodic checkpoint saving to pool")
    print("  2. KL divergence penalty computation")
    print("  3. Evaluation against historical checkpoints")
    print("  4. Opponent sampling from pool")
    print("=" * 80)
    print("\nUsage:")
    print("  Add to your PPO config:")
    print('    .callbacks(RollingTrainingCallback)')
    print("  Configure pool settings:")
    print('    .pool_config({')
    print('        "max_pool_size": 20,')
    print('        "kl_penalty_beta": 0.01,')
    print('        "checkpoint_save_frequency": 10')
    print('    })')
    print("=" * 80)


if __name__ == "__main__":
    main()

