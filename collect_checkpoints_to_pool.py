#!/usr/bin/env python3
"""
Collect Checkpoints to Pool

This script collects checkpoints from all past training runs (PPO_2025-* folders)
and adds them to the checkpoint pool for rolling training.

Usage:
    python collect_checkpoints_to_pool.py --max-pool-size 20 --max-per-run 3
"""

import argparse
from checkpoint_pool_manager import CheckpointPoolManager


def main():
    parser = argparse.ArgumentParser(
        description="Collect checkpoints from past training runs into the pool"
    )
    parser.add_argument(
        "--max-pool-size",
        type=int,
        default=20,
        help="Maximum number of checkpoints to keep in pool (default: 20)"
    )
    parser.add_argument(
        "--max-per-run",
        type=int,
        default=3,
        help="Maximum checkpoints to collect per training run (default: 3)"
    )
    parser.add_argument(
        "--select-method",
        type=str,
        default="evenly_spaced",
        choices=["latest", "evenly_spaced", "all"],
        help="Method to select checkpoints from each run (default: evenly_spaced)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="PPO_2025-*",
        help="Glob pattern to match training run directories (default: PPO_2025-*)"
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear existing pool before collecting"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COLLECTING CHECKPOINTS TO POOL")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Max pool size: {args.max_pool_size}")
    print(f"  Max per run: {args.max_per_run}")
    print(f"  Select method: {args.select_method}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Clear first: {args.clear_first}")
    print("=" * 80)
    
    # Initialize checkpoint pool manager
    manager = CheckpointPoolManager(max_pool_size=args.max_pool_size)
    
    # Clear pool if requested
    if args.clear_first:
        print("\n🗑️  Clearing existing pool...")
        manager.clear_pool()
    
    # Collect checkpoints from past runs
    print(f"\n🔍 Collecting checkpoints from past training runs...")
    num_collected = manager.collect_from_training_runs(
        base_dir=".",
        pattern=args.pattern,
        max_per_run=args.max_per_run,
        select_method=args.select_method
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    stats = manager.get_pool_statistics()
    print(f"Pool Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Checkpoint pool is ready for rolling training!")
    print(f"   Pool directory: {manager.pool_dir}")
    print(f"   Use this pool in training by configuring the environment")
    print("=" * 80)


if __name__ == "__main__":
    main()

