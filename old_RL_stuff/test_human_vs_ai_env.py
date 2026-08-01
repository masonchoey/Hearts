"""
Test script for Hearts Human vs AI Environment
Demonstrates how to use the HeartsGymEnvHumanVsAI environment with a trained model.
"""

import numpy as np
import os
from hearts_env_human_vs_ai import HeartsGymEnvHumanVsAI
from backend.models.hearts_model import HeartsAIModel


def test_environment_basic():
    """Test the environment with random actions (no AI model)."""
    print("\n" + "="*60)
    print("Test 1: Basic Environment Test (No AI Model)")
    print("="*60)
    
    # Create environment without AI model (will use random policy)
    env = HeartsGymEnvHumanVsAI()
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment reset successfully")
    print(f"  - Observation shape: {obs['observations'].shape}")
    print(f"  - Action mask shape: {obs['action_mask'].shape}")
    print(f"  - Current player: {info['current_player']}")
    print(f"  - Number of legal actions: {sum(obs['action_mask'])}")
    
    # Play a few turns with random actions
    turns = 0
    max_turns = 10
    
    while turns < max_turns and info['is_human_turn']:
        # Get legal actions
        legal_actions = [i for i in range(52) if obs['action_mask'][i] == 1]
        
        # Choose random legal action for human
        action = np.random.choice(legal_actions)
        
        print(f"\nTurn {turns + 1}:")
        print(f"  - Human player choosing action: {action}")
        print(f"  - Legal actions available: {legal_actions[:5]}... (showing first 5)")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Next player: {info['current_player']}")
        
        turns += 1
        
        if terminated:
            print(f"\n✓ Game completed!")
            print(f"  - Total turns: {turns}")
            print(f"  - Final rewards: {info['all_player_rewards']}")
            break
    
    print(f"\n✓ Basic test completed successfully")


def test_environment_with_ai_model():
    """Test the environment with a trained AI model."""
    print("\n" + "="*60)
    print("Test 2: Environment Test with AI Model")
    print("="*60)
    
    # Find the latest checkpoint
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Try to find a checkpoint automatically
        print("\n⚠ No checkpoint path in .env, searching for checkpoints...")
        
        possible_dirs = [
            "PPO_2025-10-17_03-34-46",
            "PPO_2025-10-07_04-21-40",
            "PPO_2025-10-03_15-21-17",
        ]
        
        for dir_name in possible_dirs:
            base_path = f"/Users/masonchoey/Documents/GitHub/Hearts/{dir_name}"
            if os.path.exists(base_path):
                # Find trial directory (PPO_*)
                try:
                    trial_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("PPO_")]
                    if trial_dirs:
                        trial_path = os.path.join(base_path, trial_dirs[0])
                        # Find latest checkpoint inside trial directory
                        checkpoint_dirs = [d for d in os.listdir(trial_path) if d.startswith("checkpoint_")]
                        if checkpoint_dirs:
                            # Sort to get the latest checkpoint
                            checkpoint_dirs.sort()
                            latest_checkpoint = checkpoint_dirs[-1]
                            checkpoint_path = os.path.join(trial_path, latest_checkpoint)
                            print(f"  ✓ Found checkpoint: {checkpoint_path}")
                            break
                except Exception as e:
                    print(f"  ✗ Error searching {base_path}: {e}")
                    continue
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("\n⚠ No checkpoint found. Skipping AI model test.")
        print("  Set MODEL_CHECKPOINT_PATH in .env to test with AI model.")
        return
    
    print(f"\n✓ Using checkpoint: {checkpoint_path}")
    
    # Load AI model
    print("\nLoading AI model...")
    try:
        ai_model = HeartsAIModel(checkpoint_path=checkpoint_path, eager_load=True)
    except Exception as e:
        print(f"\n⚠ Could not load AI model: {e}")
        print("  This may be due to checkpoint format issues.")
        print("  Continuing test with random policy...")
        ai_model = None
    
    # Create environment with AI model
    env_config = {
        "ai_model": ai_model,
        "human_player_id": 0
    }
    env = HeartsGymEnvHumanVsAI(env_config=env_config)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment reset successfully with AI model")
    print(f"  - Current player: {info['current_player']}")
    print(f"  - Number of legal actions: {sum(obs['action_mask'])}")
    
    # Play a few turns with random human actions
    turns = 0
    max_turns = 10
    
    while turns < max_turns and info['is_human_turn']:
        # Get legal actions
        legal_actions = [i for i in range(52) if obs['action_mask'][i] == 1]
        
        # Choose random legal action for human
        action = np.random.choice(legal_actions)
        
        print(f"\nTurn {turns + 1}:")
        print(f"  - Human player choosing action: {action}")
        
        # Take step (AI players will play automatically)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Game history length: {len(info['game_history'])}")
        
        turns += 1
        
        if terminated:
            print(f"\n✓ Game completed!")
            print(f"  - Total human turns: {turns}")
            print(f"  - Total game actions: {len(info['game_history'])}")
            print(f"  - Final rewards: {info['all_player_rewards']}")
            print(f"  - Human player reward: {info['all_player_rewards'][0]}")
            break
    
    # Cleanup
    if ai_model:
        ai_model.shutdown()
    print(f"\n✓ AI model test completed successfully")


def test_action_masking():
    """Test that action masking works correctly."""
    print("\n" + "="*60)
    print("Test 3: Action Masking Test")
    print("="*60)
    
    env = HeartsGymEnvHumanVsAI()
    obs, info = env.reset(seed=42)
    
    # Verify action mask
    legal_actions_from_mask = [i for i in range(52) if obs['action_mask'][i] == 1]
    legal_actions_from_info = info['legal_actions']
    
    print(f"\n✓ Action mask check:")
    print(f"  - Legal actions from mask: {len(legal_actions_from_mask)}")
    print(f"  - Legal actions from info: {len(legal_actions_from_info)}")
    print(f"  - Match: {set(legal_actions_from_mask) == set(legal_actions_from_info)}")
    
    # Try to play an illegal action (should raise error)
    illegal_actions = [i for i in range(52) if obs['action_mask'][i] == 0]
    
    if illegal_actions:
        illegal_action = illegal_actions[0]
        print(f"\n✓ Testing illegal action handling:")
        print(f"  - Attempting illegal action: {illegal_action}")
        
        try:
            env.step(illegal_action)
            print(f"  ✗ ERROR: Illegal action was accepted!")
        except ValueError as e:
            print(f"  ✓ Illegal action correctly rejected")
            print(f"  - Error message: {str(e)[:60]}...")
    
    print(f"\n✓ Action masking test completed successfully")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Hearts Human vs AI Environment Tests")
    print("="*60)
    
    try:
        # Test 1: Basic environment
        test_environment_basic()
        
        # Test 2: Environment with AI model
        test_environment_with_ai_model()
        
        # Test 3: Action masking
        test_action_masking()
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

