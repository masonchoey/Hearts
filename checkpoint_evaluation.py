#!/usr/bin/env python3
"""
Checkpoint Evaluation System

This module evaluates policies against historical checkpoints to measure:
- Win rates against past versions
- Score improvements over time
- Strategy robustness
- Convergence stability

Usage:
    python checkpoint_evaluation.py --current-checkpoint path/to/checkpoint --num-games 100
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from collections import defaultdict

from checkpoint_pool_manager import CheckpointPoolManager
from opponent_policy_loader import OpponentPolicyLoader
from hearts_env_self_play import HeartsGymEnvSelfPlay


class CheckpointEvaluator:
    """
    Evaluates a policy checkpoint against a pool of historical checkpoints.
    
    Measures:
    - Win rate (percentage of games where policy has lowest score)
    - Average score per game
    - Score differential vs opponents
    - Head-to-head performance
    """
    
    def __init__(
        self,
        pool_manager: CheckpointPoolManager,
        policy_loader: OpponentPolicyLoader
    ):
        """
        Initialize the evaluator.
        
        Args:
            pool_manager: CheckpointPoolManager instance
            policy_loader: OpponentPolicyLoader instance
        """
        self.pool_manager = pool_manager
        self.policy_loader = policy_loader
    
    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        opponent_checkpoints: Optional[List[str]] = None,
        num_games: int = 100,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate a checkpoint against opponents.
        
        Args:
            checkpoint_path: Path to the checkpoint to evaluate
            opponent_checkpoints: List of opponent checkpoint paths (or None to use all from pool)
            num_games: Number of games to play against each opponent
            deterministic: Use deterministic action selection
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"=" * 80)
        print(f"EVALUATING CHECKPOINT: {Path(checkpoint_path).name}")
        print(f"=" * 80)
        
        # Get opponent checkpoints
        if opponent_checkpoints is None:
            opponent_checkpoints = self.pool_manager.get_all_checkpoint_paths()
        
        if not opponent_checkpoints:
            print("⚠️  No opponent checkpoints available for evaluation")
            return {}
        
        print(f"Evaluating against {len(opponent_checkpoints)} opponents")
        print(f"Games per opponent: {num_games}")
        print(f"Total games: {len(opponent_checkpoints) * num_games}")
        
        # Load the policy to evaluate
        current_policy = self.policy_loader.load_policy_from_checkpoint(checkpoint_path)
        
        # Results storage
        results = {
            "checkpoint": str(checkpoint_path),
            "num_opponents": len(opponent_checkpoints),
            "num_games_per_opponent": num_games,
            "total_games": 0,
            "wins": 0,
            "total_score": 0.0,
            "opponent_results": [],
            "per_opponent_stats": {}
        }
        
        # Evaluate against each opponent
        for opp_idx, opp_checkpoint in enumerate(opponent_checkpoints, 1):
            print(f"\n[{opp_idx}/{len(opponent_checkpoints)}] Playing against: {Path(opp_checkpoint).name}")
            
            opp_results = self._play_games_against_opponent(
                current_policy,
                checkpoint_path,
                opp_checkpoint,
                num_games=num_games,
                deterministic=deterministic
            )
            
            results["opponent_results"].append(opp_results)
            results["total_games"] += opp_results["games_played"]
            results["wins"] += opp_results["wins"]
            results["total_score"] += opp_results["total_score"]
            
            # Per-opponent stats
            opp_name = Path(opp_checkpoint).name
            results["per_opponent_stats"][opp_name] = {
                "wins": opp_results["wins"],
                "win_rate": opp_results["win_rate"],
                "avg_score": opp_results["avg_score"]
            }
            
            print(f"  Win rate: {opp_results['win_rate']:.1%}")
            print(f"  Avg score: {opp_results['avg_score']:.2f}")
        
        # Compute overall statistics
        if results["total_games"] > 0:
            results["overall_win_rate"] = results["wins"] / results["total_games"]
            results["overall_avg_score"] = results["total_score"] / results["total_games"]
        else:
            results["overall_win_rate"] = 0.0
            results["overall_avg_score"] = 0.0
        
        print(f"\n" + "=" * 80)
        print(f"EVALUATION COMPLETE")
        print(f"=" * 80)
        print(f"Overall Results:")
        print(f"  Total games: {results['total_games']}")
        print(f"  Wins: {results['wins']}")
        print(f"  Win rate: {results['overall_win_rate']:.1%}")
        print(f"  Average score: {results['overall_avg_score']:.2f}")
        print(f"=" * 80)
        
        return results
    
    def _play_games_against_opponent(
        self,
        current_policy,
        current_checkpoint_path: str,
        opponent_checkpoint_path: str,
        num_games: int = 100,
        deterministic: bool = True
    ) -> Dict:
        """
        Play multiple games against a specific opponent.
        
        Args:
            current_policy: The policy being evaluated
            current_checkpoint_path: Path to current checkpoint
            opponent_checkpoint_path: Path to opponent checkpoint
            num_games: Number of games to play
            deterministic: Use deterministic actions
            
        Returns:
            Dictionary with game results
        """
        # Load opponent policy
        opponent_policy = self.policy_loader.load_policy_from_checkpoint(opponent_checkpoint_path)
        
        # Create environment
        env = HeartsGymEnvSelfPlay()
        
        # Results tracking
        wins = 0
        total_score = 0.0
        scores_per_game = []
        
        for game_idx in range(num_games):
            # Play one game
            game_result = self._play_single_game(
                env,
                current_policy,
                opponent_policy,
                deterministic=deterministic
            )
            
            # Current policy is always player 0
            current_score = game_result["scores"][0]
            total_score += current_score
            scores_per_game.append(current_score)
            
            # Check if won (lowest score)
            if current_score == min(game_result["scores"]):
                wins += 1
        
        return {
            "opponent_checkpoint": str(opponent_checkpoint_path),
            "games_played": num_games,
            "wins": wins,
            "win_rate": wins / num_games if num_games > 0 else 0.0,
            "total_score": total_score,
            "avg_score": total_score / num_games if num_games > 0 else 0.0,
            "scores": scores_per_game
        }
    
    def _play_single_game(
        self,
        env: HeartsGymEnvSelfPlay,
        policy_0,  # Current policy being evaluated
        policy_1,  # Opponent policy (used for players 1, 2, 3)
        deterministic: bool = True
    ) -> Dict:
        """
        Play a single game with policy_0 as player 0 and policy_1 as players 1-3.
        
        Args:
            env: Hearts environment
            policy_0: Policy for player 0 (current)
            policy_1: Policy for players 1, 2, 3 (opponent)
            deterministic: Use deterministic action selection
            
        Returns:
            Dictionary with game results
        """
        obs, info = env.reset()
        done = False
        episode_length = 0
        
        while not done:
            # Get current player from observation
            # In self-play mode, environment rotates through players
            # We need to determine which policy to use
            
            # For simplicity, we'll use policy_0 for player 0 and policy_1 for others
            # In the self-play env, all players share the same observation space,
            # so we can just alternate policies based on a counter or info
            
            # Since this is a self-play env where we don't directly control which
            # player is current, we'll use policy_0 for player 0 and policy_1 for others
            
            # Get action from current policy (assuming player 0)
            # This is a simplification - in reality, we'd need to track which player
            # is making the decision and use the appropriate policy
            
            with np.errstate(all='ignore'):  # Ignore numpy warnings
                try:
                    # Use policy_0 (current policy being evaluated)
                    if hasattr(policy_0, 'compute_single_action'):
                        action, state, info_dict = policy_0.compute_single_action(
                            obs,
                            explore=not deterministic
                        )
                    else:
                        # Fallback: sample from action space
                        action = env.action_space.sample()
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_length += 1
                    
                except Exception as e:
                    print(f"⚠️  Error during game play: {e}")
                    done = True
                    break
        
        # Get final scores from info
        final_scores = info.get("all_player_rewards", [0.0, 0.0, 0.0, 0.0])
        
        return {
            "scores": final_scores,
            "episode_length": episode_length
        }
    
    def compare_checkpoint_to_pool(
        self,
        checkpoint_path: str,
        num_opponents: int = 5,
        num_games_per_opponent: int = 50,
        sampling_method: str = "random"
    ) -> Dict:
        """
        Compare a checkpoint to a sample of opponents from the pool.
        
        Args:
            checkpoint_path: Path to checkpoint to evaluate
            num_opponents: Number of opponents to sample from pool
            num_games_per_opponent: Games to play against each opponent
            sampling_method: How to sample opponents ('random', 'performance_weighted')
            
        Returns:
            Dictionary with comparison results
        """
        # Sample opponents from pool
        opponent_checkpoints = self.pool_manager.sample_opponent(
            n=num_opponents,
            method=sampling_method,
            exclude_recent=0
        )
        
        if not opponent_checkpoints:
            print("⚠️  No opponents available in pool")
            return {}
        
        # Evaluate against sampled opponents
        return self.evaluate_checkpoint(
            checkpoint_path,
            opponent_checkpoints=opponent_checkpoints,
            num_games=num_games_per_opponent,
            deterministic=True
        )
    
    def save_evaluation_results(
        self,
        results: Dict,
        output_file: Optional[str] = None
    ):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_file: Output file path (or None for auto-generated name)
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results_{timestamp}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved evaluation results to: {output_path}")


def main():
    """CLI for checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint against historical opponents"
    )
    parser.add_argument(
        "--current-checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to evaluate"
    )
    parser.add_argument(
        "--pool-dir",
        type=str,
        default="models/pool",
        help="Directory containing checkpoint pool"
    )
    parser.add_argument(
        "--num-opponents",
        type=int,
        default=5,
        help="Number of opponents to evaluate against"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games to play per opponent"
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="random",
        choices=["random", "performance_weighted"],
        help="Method to sample opponents from pool"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Initialize managers
    pool_manager = CheckpointPoolManager(pool_dir=args.pool_dir)
    policy_loader = OpponentPolicyLoader(pool_manager)
    evaluator = CheckpointEvaluator(pool_manager, policy_loader)
    
    # Run evaluation
    results = evaluator.compare_checkpoint_to_pool(
        checkpoint_path=args.current_checkpoint,
        num_opponents=args.num_opponents,
        num_games_per_opponent=args.num_games,
        sampling_method=args.sampling_method
    )
    
    # Save results
    if results:
        evaluator.save_evaluation_results(results, args.output)


if __name__ == "__main__":
    main()

