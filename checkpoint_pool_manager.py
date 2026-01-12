#!/usr/bin/env python3
"""
Checkpoint Pool Manager for Rolling Training

This module manages a pool of model checkpoints for robust self-play training:
- Collects checkpoints from past training runs
- Maintains a fixed-size pool with FIFO eviction
- Samples opponents from the pool during training
- Evaluates current policy against historical checkpoints
- Prevents overfitting by maintaining diverse opponents
"""

import os
import glob
import shutil
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np


class CheckpointPoolManager:
    """
    Manages a pool of model checkpoints for rolling self-play training.
    
    Key Features:
    - FIFO checkpoint management (keep last N checkpoints)
    - Random or performance-weighted opponent sampling
    - Metadata tracking (performance, timestamp, training iteration)
    - Automatic cleanup of old checkpoints
    """
    
    def __init__(
        self,
        pool_dir: str = "models/pool",
        max_pool_size: int = 20,
        metadata_file: str = "pool_metadata.json"
    ):
        """
        Initialize the checkpoint pool manager.
        
        Args:
            pool_dir: Directory to store pooled checkpoints
            max_pool_size: Maximum number of checkpoints to keep (FIFO)
            metadata_file: JSON file to store checkpoint metadata
        """
        self.pool_dir = Path(pool_dir)
        self.max_pool_size = max_pool_size
        self.metadata_file = self.pool_dir / metadata_file
        
        # Create pool directory if it doesn't exist
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
        print(f"✅ CheckpointPoolManager initialized")
        print(f"   Pool directory: {self.pool_dir}")
        print(f"   Max pool size: {self.max_pool_size}")
        print(f"   Current pool size: {len(self.metadata['checkpoints'])}")
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "checkpoints": [],  # List of checkpoint info dicts
                "creation_time": datetime.now().isoformat(),
                "total_added": 0,
                "total_evicted": 0
            }
    
    def _save_metadata(self):
        """Save checkpoint metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_checkpoint(
        self,
        checkpoint_path: str,
        performance_score: Optional[float] = None,
        training_iteration: Optional[int] = None,
        copy_to_pool: bool = True
    ) -> str:
        """
        Add a checkpoint to the pool with FIFO eviction if full.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            performance_score: Optional performance metric (e.g., average reward)
            training_iteration: Optional training iteration number
            copy_to_pool: If True, copy checkpoint to pool; if False, move it
            
        Returns:
            Path to the checkpoint in the pool
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        # Generate unique checkpoint name with microseconds and original name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_name = checkpoint_path.name
        checkpoint_name = f"checkpoint_{timestamp}_{original_name}"
        pool_checkpoint_path = self.pool_dir / checkpoint_name
        
        # Copy or move checkpoint to pool
        if copy_to_pool:
            shutil.copytree(checkpoint_path, pool_checkpoint_path)
        else:
            shutil.move(str(checkpoint_path), str(pool_checkpoint_path))
        
        # Create checkpoint metadata
        checkpoint_info = {
            "name": checkpoint_name,
            "path": str(pool_checkpoint_path),
            "timestamp": timestamp,
            "performance_score": performance_score,
            "training_iteration": training_iteration,
            "added_at": datetime.now().isoformat()
        }
        
        # Add to metadata
        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["total_added"] += 1
        
        # Apply FIFO eviction if pool is full
        if len(self.metadata["checkpoints"]) > self.max_pool_size:
            self._evict_oldest()
        
        self._save_metadata()
        
        print(f"✅ Added checkpoint to pool: {checkpoint_name}")
        print(f"   Performance: {performance_score}")
        print(f"   Iteration: {training_iteration}")
        print(f"   Pool size: {len(self.metadata['checkpoints'])}/{self.max_pool_size}")
        
        return str(pool_checkpoint_path)
    
    def _evict_oldest(self):
        """Remove the oldest checkpoint from the pool (FIFO)."""
        if not self.metadata["checkpoints"]:
            return
        
        # Remove oldest checkpoint (first in list)
        oldest = self.metadata["checkpoints"].pop(0)
        checkpoint_path = Path(oldest["path"])
        
        # Delete checkpoint directory
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        
        self.metadata["total_evicted"] += 1
        
        print(f"🗑️  Evicted oldest checkpoint: {oldest['name']}")
        print(f"   Freed space for new checkpoints")
    
    def sample_opponent(
        self,
        n: int = 1,
        method: str = "random",
        exclude_recent: int = 0
    ) -> List[str]:
        """
        Sample opponent checkpoint(s) from the pool.
        
        Args:
            n: Number of opponents to sample
            method: Sampling method ('random', 'performance_weighted', 'uniform')
            exclude_recent: Number of most recent checkpoints to exclude
            
        Returns:
            List of checkpoint paths
        """
        if not self.metadata["checkpoints"]:
            return []
        
        # Get available checkpoints (excluding recent if specified)
        available = self.metadata["checkpoints"][:-exclude_recent] if exclude_recent > 0 else self.metadata["checkpoints"]
        
        if not available:
            available = self.metadata["checkpoints"]  # Fallback to all if none available
        
        n = min(n, len(available))  # Don't sample more than available
        
        if method == "random" or method == "uniform":
            # Random uniform sampling
            sampled = random.sample(available, n)
        
        elif method == "performance_weighted":
            # Sample based on performance scores
            # Higher performance = higher probability of being selected
            scores = [cp.get("performance_score", 0.0) for cp in available]
            
            # Handle negative scores (Hearts uses negative rewards)
            # Convert to weights: higher is better
            if all(s is not None for s in scores):
                min_score = min(scores)
                weights = [s - min_score + 1.0 for s in scores]  # Shift to positive
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                
                indices = np.random.choice(len(available), size=n, replace=False, p=weights)
                sampled = [available[i] for i in indices]
            else:
                # Fallback to random if scores not available
                sampled = random.sample(available, n)
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return [cp["path"] for cp in sampled]
    
    def get_all_checkpoint_paths(self) -> List[str]:
        """Get paths to all checkpoints in the pool."""
        return [cp["path"] for cp in self.metadata["checkpoints"]]
    
    def get_pool_statistics(self) -> Dict:
        """Get statistics about the checkpoint pool."""
        checkpoints = self.metadata["checkpoints"]
        
        stats = {
            "pool_size": len(checkpoints),
            "max_pool_size": self.max_pool_size,
            "total_added": self.metadata["total_added"],
            "total_evicted": self.metadata["total_evicted"],
            "oldest_checkpoint": checkpoints[0]["timestamp"] if checkpoints else None,
            "newest_checkpoint": checkpoints[-1]["timestamp"] if checkpoints else None,
        }
        
        # Performance statistics
        scores = [cp.get("performance_score") for cp in checkpoints if cp.get("performance_score") is not None]
        if scores:
            stats["avg_performance"] = np.mean(scores)
            stats["best_performance"] = max(scores)
            stats["worst_performance"] = min(scores)
        
        return stats
    
    def collect_from_training_runs(
        self,
        base_dir: str = ".",
        pattern: str = "PPO_2025-*",
        max_per_run: int = 3,
        select_method: str = "evenly_spaced"
    ) -> int:
        """
        Collect checkpoints from past training runs.
        
        Args:
            base_dir: Base directory to search for training runs
            pattern: Glob pattern to match training run directories
            max_per_run: Maximum checkpoints to collect per training run
            select_method: How to select checkpoints ('latest', 'evenly_spaced', 'all')
            
        Returns:
            Number of checkpoints collected
        """
        base_dir = Path(base_dir)
        training_runs = sorted(base_dir.glob(pattern))
        
        print(f"🔍 Scanning for training runs matching '{pattern}'...")
        print(f"   Found {len(training_runs)} training run directories")
        
        collected_count = 0
        
        for run_dir in training_runs:
            print(f"\n📁 Processing: {run_dir.name}")
            
            # Find all checkpoint directories in this run
            checkpoint_dirs = []
            for trial_dir in run_dir.iterdir():
                if trial_dir.is_dir() and trial_dir.name.startswith("PPO_"):
                    checkpoints = sorted(trial_dir.glob("checkpoint_*"))
                    checkpoint_dirs.extend(checkpoints)
            
            if not checkpoint_dirs:
                print(f"   ⚠️  No checkpoints found")
                continue
            
            print(f"   Found {len(checkpoint_dirs)} checkpoints")
            
            # Select checkpoints based on method
            if select_method == "latest":
                selected = checkpoint_dirs[-max_per_run:]
            elif select_method == "evenly_spaced":
                if len(checkpoint_dirs) <= max_per_run:
                    selected = checkpoint_dirs
                else:
                    indices = np.linspace(0, len(checkpoint_dirs)-1, max_per_run, dtype=int)
                    selected = [checkpoint_dirs[i] for i in indices]
            elif select_method == "all":
                selected = checkpoint_dirs[:max_per_run]
            else:
                raise ValueError(f"Unknown select_method: {select_method}")
            
            # Add selected checkpoints to pool
            for checkpoint_path in selected:
                try:
                    # Extract iteration number from checkpoint name
                    iteration = int(checkpoint_path.name.split("_")[-1])
                    
                    self.add_checkpoint(
                        checkpoint_path,
                        training_iteration=iteration,
                        copy_to_pool=True
                    )
                    collected_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to add {checkpoint_path.name}: {e}")
        
        print(f"\n✅ Collection complete!")
        print(f"   Collected {collected_count} checkpoints")
        print(f"   Current pool size: {len(self.metadata['checkpoints'])}/{self.max_pool_size}")
        
        return collected_count
    
    def clear_pool(self):
        """Remove all checkpoints from the pool."""
        for checkpoint in self.metadata["checkpoints"]:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
        
        self.metadata["checkpoints"] = []
        self.metadata["total_evicted"] += len(self.metadata["checkpoints"])
        self._save_metadata()
        
        print("🗑️  Cleared all checkpoints from pool")


def main():
    """Demo script for checkpoint pool manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint Pool Manager")
    parser.add_argument("--collect", action="store_true", help="Collect checkpoints from past runs")
    parser.add_argument("--stats", action="store_true", help="Show pool statistics")
    parser.add_argument("--clear", action="store_true", help="Clear the pool")
    parser.add_argument("--max-pool-size", type=int, default=20, help="Maximum pool size")
    parser.add_argument("--max-per-run", type=int, default=3, help="Max checkpoints per run")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = CheckpointPoolManager(max_pool_size=args.max_pool_size)
    
    if args.collect:
        print("\n" + "="*80)
        print("COLLECTING CHECKPOINTS FROM PAST TRAINING RUNS")
        print("="*80)
        manager.collect_from_training_runs(
            base_dir=".",
            pattern="PPO_2025-*",
            max_per_run=args.max_per_run,
            select_method="evenly_spaced"
        )
    
    if args.stats:
        print("\n" + "="*80)
        print("CHECKPOINT POOL STATISTICS")
        print("="*80)
        stats = manager.get_pool_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    if args.clear:
        print("\n" + "="*80)
        print("CLEARING CHECKPOINT POOL")
        print("="*80)
        response = input("Are you sure you want to clear the pool? (yes/no): ")
        if response.lower() == "yes":
            manager.clear_pool()
        else:
            print("Cancelled")


if __name__ == "__main__":
    main()

