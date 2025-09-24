#!/usr/bin/env python3
"""
Example script demonstrating how to resume PPO Self-Play training from saved checkpoints.

This script shows different ways to use the resume functionality that has been added
to main_self_play.py.
"""

import os
import subprocess
import sys

def main():
    print("PPO Self-Play Training Resume Examples")
    print("=" * 50)
    
    # Check if we have existing results
    results_dir = "PPO_2025-08-28_23-03-01"
    if os.path.exists(results_dir):
        print(f"✅ Found existing training results in: {results_dir}")
        
        # Show available resume options
        print("\n📋 Available Resume Options:")
        print()
        
        print("1. Resume from latest checkpoint automatically:")
        print(f"   python main_self_play.py --resume-from-latest {results_dir}")
        print()
        
        print("2. Resume from specific checkpoint:")
        print(f"   python main_self_play.py --resume {results_dir}/PPO_hearts_env_self_play_*/checkpoint_000049")
        print()
        
        print("3. Resume with custom iteration count:")
        print(f"   python main_self_play.py --resume-from-latest {results_dir} --iterations 300")
        print()
        
        print("4. Start new training (default behavior):")
        print("   python main_self_play.py")
        print()
        
        # Interactive example
        print("🚀 Would you like to run a short resume example? (y/n): ", end="")
        choice = input().lower().strip()
        
        if choice == 'y':
            print("\n⏳ Running 5 iterations from latest checkpoint...")
            try:
                result = subprocess.run([
                    sys.executable, "main_self_play.py", 
                    "--resume-from-latest", results_dir, 
                    "--iterations", "5"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Resume example completed successfully!")
                    print("Check the output above for training progress.")
                else:
                    print("❌ Resume example failed:")
                    print(result.stderr)
            except Exception as e:
                print(f"❌ Error running example: {e}")
        else:
            print("👍 You can run the commands manually when ready.")
    
    else:
        print(f"❌ No existing training results found in: {results_dir}")
        print("💡 Run training first to create checkpoints:")
        print("   python main_self_play.py --iterations 50")
    
    print("\n" + "=" * 50)
    print("📖 Command Line Arguments:")
    print("  --resume <checkpoint_path>         Resume from specific checkpoint")
    print("  --resume-from-latest <results_dir> Resume from latest checkpoint in directory")
    print("  --iterations <number>              Number of training iterations (default: 250)")
    print("  --help                            Show all available options")


if __name__ == "__main__":
    main()
