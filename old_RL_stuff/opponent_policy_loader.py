#!/usr/bin/env python3
"""
Opponent Policy Loader for Rolling Training

This module loads and manages opponent policies from checkpoints for training:
- Loads frozen policies from checkpoint pool
- Provides interface for sampling actions from opponent policies
- Manages multiple opponent policies simultaneously
- Caches loaded policies for efficiency
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
import tempfile
import pickle


class OpponentPolicyLoader:
    """
    Loads and manages opponent policies from checkpoints.
    
    Key Features:
    - Lazy loading of policies (only load when needed)
    - Policy caching for efficiency
    - Frozen policy inference (no gradient computation)
    - Support for multiple simultaneous opponents
    """
    
    def __init__(
        self,
        checkpoint_pool_manager=None,
        max_cached_policies: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize the opponent policy loader.
        
        Args:
            checkpoint_pool_manager: CheckpointPoolManager instance
            max_cached_policies: Maximum number of policies to keep in memory
            device: Device to load policies on ('cpu' or 'cuda')
        """
        self.checkpoint_pool_manager = checkpoint_pool_manager
        self.max_cached_policies = max_cached_policies
        self.device = device
        
        # Cache for loaded policies: {checkpoint_path: policy}
        self.policy_cache = {}
        self.cache_order = []  # Track access order for LRU eviction
        
        print(f"✅ OpponentPolicyLoader initialized")
        print(f"   Max cached policies: {max_cached_policies}")
        print(f"   Device: {device}")
    
    def load_policy_from_checkpoint(
        self,
        checkpoint_path: str,
        cache: bool = True
    ) -> Policy:
        """
        Load a policy from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            cache: Whether to cache the loaded policy
            
        Returns:
            Loaded policy object
        """
        checkpoint_path = str(checkpoint_path)
        
        # Check cache first
        if checkpoint_path in self.policy_cache:
            # Move to end (most recently used)
            self.cache_order.remove(checkpoint_path)
            self.cache_order.append(checkpoint_path)
            return self.policy_cache[checkpoint_path]
        
        # Load the policy from checkpoint
        try:
            # Load the full algorithm to get the policy
            algo = PPO.from_checkpoint(checkpoint_path)
            policy = algo.get_policy()
            
            # Move model to specified device
            if hasattr(policy, 'model'):
                policy.model.to(self.device)
            
            # Set to eval mode (no training)
            if hasattr(policy, 'model'):
                policy.model.eval()
            
            # Cache the policy if requested
            if cache:
                self._add_to_cache(checkpoint_path, policy)
            
            print(f"✅ Loaded policy from: {Path(checkpoint_path).name}")
            
            return policy
            
        except Exception as e:
            print(f"❌ Failed to load policy from {checkpoint_path}: {e}")
            raise
    
    def _add_to_cache(self, checkpoint_path: str, policy: Policy):
        """Add a policy to the cache with LRU eviction."""
        # Evict oldest if cache is full
        if len(self.policy_cache) >= self.max_cached_policies:
            oldest = self.cache_order.pop(0)
            del self.policy_cache[oldest]
            print(f"🗑️  Evicted policy from cache: {Path(oldest).name}")
        
        # Add new policy
        self.policy_cache[checkpoint_path] = policy
        self.cache_order.append(checkpoint_path)
    
    def get_action(
        self,
        checkpoint_path: str,
        observation: Dict,
        deterministic: bool = True
    ) -> Tuple[int, Dict]:
        """
        Get an action from an opponent policy.
        
        Args:
            checkpoint_path: Path to the checkpoint
            observation: Observation dict (with 'observations' and 'action_mask')
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, info_dict)
        """
        policy = self.load_policy_from_checkpoint(checkpoint_path)
        
        # Compute action (no gradients)
        with torch.no_grad():
            # Policy expects batched input
            if isinstance(observation, dict):
                obs_batch = {
                    k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else v
                    for k, v in observation.items()
                }
            else:
                obs_batch = np.expand_dims(observation, axis=0)
            
            # Compute action
            action, state, info = policy.compute_single_action(
                obs_batch,
                explore=not deterministic
            )
        
        return int(action), info
    
    def get_action_distribution(
        self,
        checkpoint_path: str,
        observation: Dict
    ) -> torch.Tensor:
        """
        Get action probability distribution from an opponent policy.
        
        This is useful for computing KL divergence between policies.
        
        Args:
            checkpoint_path: Path to the checkpoint
            observation: Observation dict
            
        Returns:
            Action probabilities as a tensor
        """
        policy = self.load_policy_from_checkpoint(checkpoint_path)
        
        with torch.no_grad():
            # Convert observation to torch tensor
            if isinstance(observation, dict):
                obs_torch = {
                    k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                    if isinstance(v, np.ndarray) else v
                    for k, v in observation.items()
                }
            else:
                obs_torch = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            
            # Get model logits
            if hasattr(policy, 'model'):
                logits, _ = policy.model({"obs": obs_torch}, [], None)
                probs = torch.softmax(logits, dim=-1)
                return probs.squeeze(0)  # Remove batch dimension
            else:
                raise NotImplementedError("Policy does not have a model attribute")
    
    def sample_opponent_checkpoints(
        self,
        n: int = 1,
        method: str = "random",
        exclude_recent: int = 1
    ) -> List[str]:
        """
        Sample opponent checkpoint paths from the pool.
        
        Args:
            n: Number of opponents to sample
            method: Sampling method ('random', 'performance_weighted')
            exclude_recent: Exclude N most recent checkpoints
            
        Returns:
            List of checkpoint paths
        """
        if self.checkpoint_pool_manager is None:
            return []
        
        return self.checkpoint_pool_manager.sample_opponent(
            n=n,
            method=method,
            exclude_recent=exclude_recent
        )
    
    def clear_cache(self):
        """Clear the policy cache."""
        self.policy_cache.clear()
        self.cache_order.clear()
        print("🗑️  Cleared policy cache")
    
    def get_cache_info(self) -> Dict:
        """Get information about the policy cache."""
        return {
            "cached_policies": len(self.policy_cache),
            "max_cached": self.max_cached_policies,
            "cache_order": [Path(p).name for p in self.cache_order]
        }


class PolicyDistributionCache:
    """
    Cache for policy distributions to speed up KL divergence computation.
    
    Stores pre-computed action distributions for observations that have been
    seen before, avoiding redundant forward passes through the network.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the distribution cache.
        
        Args:
            max_size: Maximum number of distributions to cache
        """
        self.max_size = max_size
        self.cache = {}  # {(checkpoint_path, obs_hash): distribution}
        self.access_order = []
    
    def _hash_observation(self, observation: Dict) -> int:
        """Create a hash of an observation for cache lookup."""
        if isinstance(observation, dict):
            # Hash the observations array
            obs_array = observation.get("observations", observation.get("obs"))
            if isinstance(obs_array, np.ndarray):
                return hash(obs_array.tobytes())
            elif isinstance(obs_array, torch.Tensor):
                return hash(obs_array.cpu().numpy().tobytes())
        return hash(str(observation))
    
    def get(
        self,
        checkpoint_path: str,
        observation: Dict
    ) -> Optional[torch.Tensor]:
        """Get a cached distribution if available."""
        key = (checkpoint_path, self._hash_observation(observation))
        return self.cache.get(key)
    
    def put(
        self,
        checkpoint_path: str,
        observation: Dict,
        distribution: torch.Tensor
    ):
        """Add a distribution to the cache."""
        key = (checkpoint_path, self._hash_observation(observation))
        
        # LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = distribution
        self.access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


def compute_kl_divergence(
    policy_current_dist: torch.Tensor,
    policy_old_dist: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute KL divergence between two policy distributions.
    
    KL(current || old) = sum(current * log(current / old))
    
    Args:
        policy_current_dist: Current policy distribution [batch_size, num_actions]
        policy_old_dist: Old policy distribution [batch_size, num_actions]
        action_mask: Optional mask for legal actions [batch_size, num_actions]
        
    Returns:
        KL divergence (scalar or per-sample)
    """
    # Apply action mask if provided
    if action_mask is not None:
        # Mask out illegal actions
        policy_current_dist = policy_current_dist * action_mask
        policy_old_dist = policy_old_dist * action_mask
        
        # Renormalize
        policy_current_dist = policy_current_dist / (policy_current_dist.sum(dim=-1, keepdim=True) + 1e-8)
        policy_old_dist = policy_old_dist / (policy_old_dist.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Compute KL divergence
    # Add small epsilon for numerical stability
    epsilon = 1e-8
    kl_div = torch.sum(
        policy_current_dist * torch.log((policy_current_dist + epsilon) / (policy_old_dist + epsilon)),
        dim=-1
    )
    
    return kl_div


def main():
    """Demo script for opponent policy loader."""
    from checkpoint_pool_manager import CheckpointPoolManager
    
    # Initialize manager
    pool_manager = CheckpointPoolManager(max_pool_size=10)
    loader = OpponentPolicyLoader(pool_manager)
    
    # Print cache info
    print("\n" + "="*80)
    print("OPPONENT POLICY LOADER INFO")
    print("="*80)
    info = loader.get_cache_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Sample some opponents
    opponents = loader.sample_opponent_checkpoints(n=3, method="random")
    print(f"\nSampled {len(opponents)} opponents:")
    for i, opp in enumerate(opponents, 1):
        print(f"  {i}. {Path(opp).name}")


if __name__ == "__main__":
    main()

