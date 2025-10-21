"""
Test Backend Integration with Gymnasium Environment
Verifies that the backend works with the new HeartsGymWrapper
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from backend.game.state_manager import GameStateManager
from backend.schemas.types import Card


def test_game_state_manager():
    """Test that GameStateManager works with HeartsGymWrapper"""
    print("\n" + "="*60)
    print("Test: GameStateManager Integration")
    print("="*60)
    
    # Initialize game manager (will load AI model if checkpoint available)
    print("\n1. Initializing GameStateManager...")
    manager = GameStateManager(eager_load=False)  # Use lazy loading for faster test
    print("   ✓ GameStateManager initialized")
    
    # Create a new game
    print("\n2. Creating new game...")
    game_id = "test-game-123"
    game_state = manager.create_game(game_id)
    print(f"   ✓ Game created with ID: {game_id}")
    print(f"   - Current player: {game_state.current_player}")
    print(f"   - Human player hand: {len(game_state.players[0].hand)} cards")
    print(f"   - Legal actions: {len(game_state.legal_actions)}")
    print(f"   - Passing phase: {game_state.is_passing_phase}")
    
    # Check that hands are dealt
    assert len(game_state.players[0].hand) > 0, "Human player should have cards"
    assert game_state.current_player == 0, "Should be human player's turn after AI auto-play"
    
    # Play a card
    print("\n3. Playing a card...")
    human_hand = game_state.players[0].hand
    card_to_play = human_hand[0]  # Play first card in hand
    print(f"   - Human playing: {card_to_play}")
    
    try:
        updated_state = manager.play_card(game_id, 0, card_to_play)
        print(f"   ✓ Card played successfully")
        print(f"   - Current player after move: {updated_state.current_player}")
        print(f"   - Game over: {updated_state.game_over}")
        print(f"   - Current trick length: {len(updated_state.current_trick)}")
    except Exception as e:
        print(f"   ✗ Error playing card: {e}")
        raise
    
    # Get game state
    print("\n4. Getting game state...")
    retrieved_state = manager.get_game(game_id)
    assert retrieved_state is not None, "Should retrieve game state"
    print(f"   ✓ Game state retrieved")
    print(f"   - Current player: {retrieved_state.current_player}")
    
    # Reset game
    print("\n5. Resetting game...")
    reset_state = manager.reset_game(game_id)
    print(f"   ✓ Game reset")
    print(f"   - Current player: {reset_state.current_player}")
    print(f"   - Human player hand: {len(reset_state.players[0].hand)} cards")
    
    # Delete game
    print("\n6. Deleting game...")
    success = manager.delete_game(game_id)
    assert success, "Should delete game successfully"
    print(f"   ✓ Game deleted")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")


def test_gym_wrapper_direct():
    """Test HeartsGymWrapper directly"""
    print("\n" + "="*60)
    print("Test: Direct HeartsGymWrapper Test")
    print("="*60)
    
    from backend.game.hearts_gym_wrapper import HeartsGymWrapper
    
    # Get checkpoint path
    checkpoint_path = os.getenv("CHECKPOINT_PATH") or os.getenv("MODEL_CHECKPOINT_PATH")
    
    print(f"\n1. Creating HeartsGymWrapper...")
    print(f"   - Checkpoint: {checkpoint_path or 'None (will use random policy)'}")
    
    wrapper = HeartsGymWrapper(
        checkpoint_path=checkpoint_path,
        human_player_id=0,
        eager_load=False
    )
    print(f"   ✓ Wrapper created")
    
    # Reset
    print(f"\n2. Resetting environment...")
    state = wrapper.reset()
    print(f"   ✓ Environment reset")
    print(f"   - Current player: {state['current_player']}")
    print(f"   - Legal actions: {len(state['legal_actions'])}")
    print(f"   - Is human turn: {state['is_human_turn']}")
    
    # Play a few turns
    print(f"\n3. Playing a few turns...")
    for turn in range(3):
        if not wrapper.is_game_active():
            break
        
        if wrapper.is_human_turn():
            legal_actions = wrapper.get_legal_actions()
            action = legal_actions[0]  # Play first legal action
            print(f"   Turn {turn + 1}: Human plays action {action}")
            
            result = wrapper.step(action)
            print(f"   - Game over: {result['terminated']}")
            
            if result['terminated']:
                print(f"   - Final rewards: {result['all_rewards']}")
                break
    
    # Cleanup
    print(f"\n4. Cleaning up...")
    wrapper.shutdown()
    print(f"   ✓ Wrapper shut down")
    
    print("\n" + "="*60)
    print("✓ Direct wrapper test passed!")
    print("="*60 + "\n")


def main():
    """Run all tests"""
    try:
        # Test 1: GameStateManager integration
        test_game_state_manager()
        
        # Test 2: Direct wrapper test
        test_gym_wrapper_direct()
        
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

