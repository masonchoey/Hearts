#!/usr/bin/env python3
"""
Training monitoring script to track improvements over baseline performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv


class TrainingMonitor:
    """Monitor training progress and compare against baseline."""
    
    def __init__(self, baseline_reward=-1.35):
        self.baseline_reward = baseline_reward
        self.baseline_invalid_rate = 0.966  # 96.6% from your results
        self.baseline_vf_explained = 0.43  # Current VF explained variance from checkpoint_000500
        
    def load_training_data(self, results_dir):
        """Load training data from Ray results directory."""
        results_dirs = []
        
        # Find all PPO result directories
        for item in Path(results_dir).iterdir():
            if item.is_dir() and "PPO" in item.name:
                results_dirs.append(item)
                
        if not results_dirs:
            print(f"No PPO training directories found in {results_dir}")
            return None
            
        # Load data from the most recent run
        latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
        progress_file = latest_dir / "progress.csv"
        
        if not progress_file.exists():
            print(f"No progress.csv found in {latest_dir}")
            return None
            
        df = pd.read_csv(progress_file)
        return df, latest_dir
        
    def analyze_improvements(self, df):
        """Analyze improvements over baseline."""
        print("🔍 IMPROVEMENT ANALYSIS")
        print("=" * 60)
        
        # Reward improvements
        latest_reward = df["env_runners/episode_reward_mean"].iloc[-1]
        reward_improvement = ((latest_reward - self.baseline_reward) / abs(self.baseline_reward)) * 100
        
        print(f"📊 REWARD ANALYSIS:")
        print(f"  Baseline Average Reward: {self.baseline_reward:.0f}")
        print(f"  Current Average Reward:  {latest_reward:.0f}")
        print(f"  Improvement: {reward_improvement:.1f}%")
        
        # Value function learning
        latest_vf_explained = df["info/learner/default_policy/learner_stats/vf_explained_var"].iloc[-1]
        print(f"\n🧠 VALUE FUNCTION LEARNING:")
        print(f"  Baseline VF Explained Variance: {self.baseline_vf_explained:.5f}")
        print(f"  Current VF Explained Variance:  {latest_vf_explained:.5f}")
        vf_improvement = (latest_vf_explained / self.baseline_vf_explained) if self.baseline_vf_explained > 0 else float('inf')
        print(f"  Improvement: {vf_improvement:.2f}x better")
        
        # Policy loss and entropy
        latest_policy_loss = df["info/learner/default_policy/learner_stats/policy_loss"].iloc[-1]
        latest_entropy = df["info/learner/default_policy/learner_stats/entropy"].iloc[-1]
        
        print(f"\n🎯 POLICY METRICS:")
        print(f"  Policy Loss: {latest_policy_loss:.4f}")
        print(f"  Entropy: {latest_entropy:.4f}")
        
        # Performance consistency
        reward_std = df["env_runners/episode_reward_mean"].tail(50).std()
        print(f"\n📈 STABILITY METRICS:")
        print(f"  Reward Std Dev (last 50 iterations): {reward_std:.2f}")
        
        return {
            "reward_improvement": reward_improvement,
            "latest_reward": latest_reward,
            "vf_improvement": vf_improvement,
            "latest_vf_explained": latest_vf_explained,
            "reward_stability": reward_std
        }
        
    def plot_training_progress(self, df, save_path=None):
        """Create comprehensive training progress plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Hearts RL Training Progress Analysis", fontsize=16)
        
        # 1. Reward progression
        axes[0, 0].plot(df["training_iteration"], df["env_runners/episode_reward_mean"], 
                       label="Training", linewidth=2)
        if "evaluation/env_runners/episode_reward_mean" in df.columns:
            axes[0, 0].plot(df["training_iteration"], df["evaluation/env_runners/episode_reward_mean"], 
                           label="Evaluation", linewidth=2, alpha=0.8)
        # Baseline removed for better visibility of training data
        axes[0, 0].set_xlabel("Training Iteration")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].set_title("Episode Reward Progress")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Value function explained variance
        if "info/learner/default_policy/learner_stats/vf_explained_var" in df.columns:
            axes[0, 1].plot(df["training_iteration"], 
                           df["info/learner/default_policy/learner_stats/vf_explained_var"],
                           color='green', linewidth=2, label='VF Explained Variance')
            # Baseline removed for better visibility of training data
            axes[0, 1].set_xlabel("Training Iteration")
            axes[0, 1].set_ylabel("VF Explained Variance")
            axes[0, 1].set_title("Value Function Learning")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            # Remove log scale since baseline comparison is gone
            # axes[0, 1].set_yscale('log')
        
        # 3. Policy loss
        if "info/learner/default_policy/learner_stats/policy_loss" in df.columns:
            axes[0, 2].plot(df["training_iteration"], 
                           df["info/learner/default_policy/learner_stats/policy_loss"],
                           color='orange', linewidth=2)
            axes[0, 2].set_xlabel("Training Iteration")
            axes[0, 2].set_ylabel("Policy Loss")
            axes[0, 2].set_title("Policy Loss")
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Entropy
        if "info/learner/default_policy/learner_stats/entropy" in df.columns:
            axes[1, 0].plot(df["training_iteration"], 
                           df["info/learner/default_policy/learner_stats/entropy"],
                           color='purple', linewidth=2)
            axes[1, 0].set_xlabel("Training Iteration")
            axes[1, 0].set_ylabel("Policy Entropy")
            axes[1, 0].set_title("Exploration (Entropy)")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Learning rate
        if "info/learner/default_policy/learner_stats/cur_lr" in df.columns:
            axes[1, 1].plot(df["training_iteration"], 
                           df["info/learner/default_policy/learner_stats/cur_lr"],
                           color='brown', linewidth=2)
            axes[1, 1].set_xlabel("Training Iteration")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].set_title("Learning Rate Schedule")
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Episode length
        if "env_runners/episode_len_mean" in df.columns:
            axes[1, 2].plot(df["training_iteration"], df["env_runners/episode_len_mean"],
                           color='teal', linewidth=2)
            axes[1, 2].set_xlabel("Training Iteration")
            axes[1, 2].set_ylabel("Episode Length")
            axes[1, 2].set_title("Average Episode Length")
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training progress plots saved to {save_path}")
        
        plt.show()
        
    def create_comparison_report(self, df, results_dir):
        """Create a comprehensive comparison report."""
        improvements = self.analyze_improvements(df)
        
        report = f"""
# Hearts RL Training Improvement Report

## Executive Summary
This report compares the current training results against the baseline performance.

## Key Improvements

### 🎯 Reward Performance
- **Baseline Reward**: {self.baseline_reward:.0f}
- **Current Reward**: {improvements['latest_reward']:.0f}
- **Improvement**: {improvements['reward_improvement']:.1f}%

### 🧠 Value Function Learning  
- **Baseline VF Explained Variance**: {self.baseline_vf_explained:.5f}
- **Current VF Explained Variance**: {improvements['latest_vf_explained']:.5f}
- **Improvement Factor**: {improvements['vf_improvement']:.2f}x

### 📊 Training Stability
- **Reward Standard Deviation**: {improvements['reward_stability']:.2f}
- **Training Consistency**: {'Good' if improvements['reward_stability'] < 100 else 'Needs Improvement'}

## Recommendations for Further Improvement

### If Reward Improvement < 20%:
1. **Increase reward shaping** - Add more intermediate rewards
2. **Improve action masking** - Ensure invalid actions are properly masked
3. **Tune hyperparameters** - Adjust learning rate and batch sizes

### If VF Explained Variance shows decline:
1. **Increase vf_loss_coeff** to strengthen value learning
2. **Add state normalization** for better value function approximation
3. **Consider deeper networks** for more complex value estimation

### If Training is Unstable (high std dev):
1. **Reduce learning rate** for more stable updates
2. **Increase batch size** for more stable gradients
3. **Add gradient clipping** to prevent exploding gradients

## Next Steps
1. Continue training if improvements are positive
2. Implement additional reward shaping if needed
3. Consider opponent curriculum (stronger opponents gradually)
4. Add self-play for more robust strategy learning

Generated from: {results_dir}
"""
        
        report_path = f"training_improvement_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"📄 Comprehensive report saved to: {report_path}")
        return report_path


def monitor_latest_training():
    """Monitor the latest training run."""
    # Load environment variables from .env file
    load_dotenv()
    
    monitor = TrainingMonitor()
    
    # Default to the latest checkpoint path
    default_checkpoint_path = "/Users/masonchoey/Documents/GitHub/OpenSpiel-Hearts/PPO_2025-08-28_23-03-01/PPO_hearts_env_self_play_d3320_00000_0_2025-08-28_23-03-03/checkpoint_000500"
    
    # Check if CHECKPOINT_PATH is specified in environment variables, otherwise use default
    checkpoint_path = os.getenv("CHECKPOINT_PATH", default_checkpoint_path)
    
    if checkpoint_path:
        if checkpoint_path == default_checkpoint_path:
            print(f"🎯 Using default checkpoint path: {checkpoint_path}")
        else:
            print(f"🎯 Using CHECKPOINT_PATH from environment: {checkpoint_path}")
        
        # Convert to absolute path
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
            print("\n🔧 CHECKPOINT_PATH TROUBLESHOOTING:")
            print("1. Verify the path exists and is accessible")
            print("2. Check if training has generated the expected checkpoint")
            print("3. Use absolute paths for best results")
            return
        
        # Determine the results directory (parent of checkpoint or checkpoint itself if it contains progress.csv)
        results_dir = checkpoint_path
        progress_file = os.path.join(checkpoint_path, "progress.csv")
        
        # If progress.csv is not directly in checkpoint_path, look in parent directory structure
        if not os.path.exists(progress_file):
            # Try going up directory levels to find the training run directory
            current_path = checkpoint_path
            while current_path != os.path.dirname(current_path):  # Stop at root
                parent = os.path.dirname(current_path)
                potential_progress = os.path.join(parent, "progress.csv")
                if os.path.exists(potential_progress):
                    results_dir = os.path.dirname(parent)  # Parent of the directory containing progress.csv
                    break
                current_path = parent
            else:
                print(f"❌ No progress.csv found in or above: {checkpoint_path}")
                print("💡 Expected structure: training_run_dir/progress.csv")
                return
        else:
            # progress.csv is directly in checkpoint_path, so use its parent as results_dir
            results_dir = os.path.dirname(checkpoint_path)
        
        print(f"📁 Using results directory: {results_dir}")
        
        # Load training data from the specified path
        data = monitor.load_training_data(results_dir)
        if data is None:
            return
            
        df, latest_dir = data
        print(f"📁 Monitoring: {latest_dir}")

    else:
        print("❌ No checkpoint path available")
        return
    
    # Analyze and report
    improvements = monitor.analyze_improvements(df)
    
    # Create plots
    plot_path = f"training_progress_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    monitor.plot_training_progress(df, plot_path)
    
    # Create report
    monitor.create_comparison_report(df, latest_dir)
    
    # Summary
    print("\n" + "="*60)
    print("📋 MONITORING SUMMARY")
    print("="*60)
    
    if improvements['reward_improvement'] > 10:
        print("✅ GOOD: Significant reward improvement detected!")
    elif improvements['reward_improvement'] > 0:
        print("⚠️  MODERATE: Some reward improvement, consider tuning")
    else:
        print("❌ CONCERN: Reward declining, needs investigation")
        
    if improvements['vf_improvement'] > 1.2:
        print("✅ GOOD: Value function learning improved!")
    elif improvements['vf_improvement'] > 0.9:
        print("⚠️  MODERATE: Value function stable")
    else:
        print("❌ CONCERN: Value function declining, needs attention")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor and analyze RL training runs.")
    parser.add_argument("--live", action="store_true", help="Enable live monitoring mode (refresh plots and metrics during training)")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds for live mode")
    parser.add_argument("--save-plots", action="store_true", help="Save plot image on each refresh in live mode")
    parser.add_argument("--search", type=str, nargs="*", default=None, help="Directories to search for PPO runs (defaults to common locations)")

    args = parser.parse_args()

    def _find_latest_run(search_dirs=None):
        possible_dirs = search_dirs or [
            ".",
            "hearts_phase1_basic",
            "hearts_phase2_enhanced",
            "ray_results",
            os.path.expanduser("~/ray_results"),
        ]

        found_ppo_dirs = []
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                try:
                    items = [d for d in os.listdir(dir_path) if "PPO" in d and os.path.isdir(os.path.join(dir_path, d))]
                    for ppo_dir in items:
                        found_ppo_dirs.append((dir_path, ppo_dir, os.path.join(dir_path, ppo_dir)))
                except PermissionError:
                    continue

        if not found_ppo_dirs:
            return None, None, None

        latest = max(found_ppo_dirs, key=lambda x: os.path.getmtime(x[2]))
        return latest

    def live_monitor(interval_s=5.0, save_plots=False, search_dirs=None):
        latest = _find_latest_run(search_dirs)
        if latest == (None, None, None):
            print("❌ No PPO training results found. Start training first, then re-run with --live.")
            return

        parent_dir, run_name, run_path = latest
        progress_csv = os.path.join(run_path, "progress.csv")

        print(f"📁 Live monitoring: {run_path}")
        print(f"⏱️  Refresh interval: {interval_s:.1f}s")

        # Prepare plot
        plt.ion()
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Hearts RL Training (Live)", fontsize=16)

        lines = {}

        def ensure_line(ax, key, x, y, **kwargs):
            if key in lines:
                line = lines[key]
                line.set_xdata(x)
                line.set_ydata(y)
            else:
                (line,) = ax.plot(x, y, **kwargs)
                lines[key] = line

        # Load environment variables from .env file
        load_dotenv()
        
        monitor = TrainingMonitor()

        last_size = 0
        try:
            while True:
                # If a newer run appears, switch to it
                latest_now = _find_latest_run(search_dirs)
                if latest_now != (None, None, None) and latest_now[2] != run_path:
                    parent_dir, run_name, run_path = latest_now
                    progress_csv = os.path.join(run_path, "progress.csv")
                    print(f"🔄 Switching to newer run: {run_path}")
                    last_size = 0

                if not os.path.exists(progress_csv):
                    print("⏳ Waiting for progress.csv to be created...")
                    time.sleep(interval_s)
                    continue

                try:
                    # Read with low_memory=False to avoid dtype warnings on partial writes
                    df = pd.read_csv(progress_csv, low_memory=False)
                except Exception as e:
                    # File may be in the middle of a write; wait and retry
                    time.sleep(0.5)
                    continue

                if df.empty:
                    time.sleep(interval_s)
                    continue

                # Update plots
                ax00 = axes[0, 0]
                ax01 = axes[0, 1]
                ax02 = axes[0, 2]
                ax10 = axes[1, 0]
                ax11 = axes[1, 1]
                ax12 = axes[1, 2]

                x = df["training_iteration"] if "training_iteration" in df.columns else np.arange(len(df))

                if "env_runners/episode_reward_mean" in df.columns:
                    ensure_line(ax00, "reward_train", x, df["env_runners/episode_reward_mean"], label="Training", linewidth=2)
                if "evaluation/env_runners/episode_reward_mean" in df.columns:
                    ensure_line(ax00, "reward_eval", x, df["evaluation/env_runners/episode_reward_mean"], label="Evaluation", linewidth=2, alpha=0.8)
                ax00.set_title("Episode Reward")
                ax00.set_xlabel("Iteration")
                ax00.set_ylabel("Reward")
                ax00.legend()
                ax00.grid(True, alpha=0.3)

                if "info/learner/default_policy/learner_stats/vf_explained_var" in df.columns:
                    ensure_line(ax01, "vf_ev", x, df["info/learner/default_policy/learner_stats/vf_explained_var"], color="green", linewidth=2)
                    ax01.set_title("VF Explained Variance")
                    ax01.grid(True, alpha=0.3)

                if "info/learner/default_policy/learner_stats/policy_loss" in df.columns:
                    ensure_line(ax02, "policy_loss", x, df["info/learner/default_policy/learner_stats/policy_loss"], color="orange", linewidth=2)
                    ax02.set_title("Policy Loss")
                    ax02.grid(True, alpha=0.3)

                if "info/learner/default_policy/learner_stats/entropy" in df.columns:
                    ensure_line(ax10, "entropy", x, df["info/learner/default_policy/learner_stats/entropy"], color="purple", linewidth=2)
                    ax10.set_title("Entropy")
                    ax10.grid(True, alpha=0.3)

                if "info/learner/default_policy/learner_stats/cur_lr" in df.columns:
                    ensure_line(ax11, "lr", x, df["info/learner/default_policy/learner_stats/cur_lr"], color="brown", linewidth=2)
                    ax11.set_title("Learning Rate")
                    ax11.grid(True, alpha=0.3)

                if "env_runners/episode_len_mean" in df.columns:
                    ensure_line(ax12, "ep_len", x, df["env_runners/episode_len_mean"], color="teal", linewidth=2)
                    ax12.set_title("Episode Length")
                    ax12.grid(True, alpha=0.3)

                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()

                if save_plots:
                    out_path = f"live_training_{int(time.time())}.png"
                    fig.savefig(out_path, dpi=120, bbox_inches="tight")

                # Print a concise live summary line
                last = df.iloc[-1]
                reward = last.get("env_runners/episode_reward_mean", np.nan)
                vf_ev = last.get("info/learner/default_policy/learner_stats/vf_explained_var", np.nan)
                entropy = last.get("info/learner/default_policy/learner_stats/entropy", np.nan)
                itr = int(last.get("training_iteration", len(df)))
                print(f"iter={itr} | reward={reward:.3f} | vf_ev={vf_ev:.4f} | entropy={entropy:.3f}")

                time.sleep(interval_s)
        except KeyboardInterrupt:
            print("\n🛑 Live monitoring stopped by user.")

    if args.live:
        live_monitor(interval_s=args.interval, save_plots=args.save_plots, search_dirs=args.search)
    else:
        monitor_latest_training()