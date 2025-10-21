"""
Game State Manager
Manages multiple game sessions using Gymnasium environment with automatic AI control
"""
from typing import Dict, Optional
from ..schemas.types import GameState, Player, Card
from .hearts_gym_wrapper import HeartsGymWrapper
from .hearts_logic import HeartsGame  # Keep for utility functions
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()


class GameStateManager:
    """
    Manages game sessions using HeartsGymWrapper for automatic AI control
    """
    
    def __init__(self, eager_load: bool = True):
        """
        Initialize GameStateManager with Gymnasium environment
        
        Args:
            eager_load: If True, load AI model immediately for low latency
        """
        self.gym_wrappers: Dict[str, HeartsGymWrapper] = {}
        self.game_states: Dict[str, GameState] = {}
        
        # Store checkpoint path for creating new game wrappers
        self.checkpoint_path = os.getenv("CHECKPOINT_PATH") or os.getenv("MODEL_CHECKPOINT_PATH")
        self.eager_load = eager_load
        print(f"GameStateManager initialized (eager_load={eager_load})")
        print(f"Checkpoint path: {self.checkpoint_path}")
        
    def create_game(self, game_id: str) -> GameState:
        """
        Create a new game session using Gymnasium wrapper
        """
        # Create HeartsGymWrapper which handles AI automatically
        wrapper = HeartsGymWrapper(
            checkpoint_path=self.checkpoint_path,
            human_player_id=0,
            eager_load=self.eager_load
        )
        self.gym_wrappers[game_id] = wrapper
        
        # Reset the environment (AI turns are handled automatically)
        gym_state = wrapper.reset()
        
        # Create initial game state for frontend
        players = [
            Player(id=0, name="You", is_ai=False, hand=[], score=0, round_score=0),
            Player(id=1, name="AI 1", is_ai=True, hand=[], score=0, round_score=0),
            Player(id=2, name="AI 2", is_ai=True, hand=[], score=0, round_score=0),
            Player(id=3, name="AI 3", is_ai=True, hand=[], score=0, round_score=0),
        ]
        
        # Get hands from the wrapper's internal game
        # We need to access the internal game for hand extraction
        internal_game = HeartsGame()
        internal_game._env = wrapper.env._base_env
        internal_game._timestep = wrapper.env._last_timestep
        
        for player in players:
            player.hand = internal_game.get_player_hand(player.id)
        
        # Extract game info from gym state
        observation = gym_state["observation"].tolist()
        legal_actions = gym_state["legal_actions"]
        current_player = gym_state["current_player"]
        
        # Check passing phase using internal game
        is_passing_phase = internal_game.is_passing_phase(0)
        pass_direction = internal_game.get_pass_direction(0)
        hearts_broken = internal_game.hearts_broken
        
        game_state = GameState(
            players=players,
            current_trick=[],
            current_player=current_player,
            round_number=1,
            tricks_played=0,
            game_over=False,
            hearts_broken=hearts_broken,
            winner=None,
            observation=observation,
            legal_actions=legal_actions,
            is_passing_phase=is_passing_phase,
            pass_direction=pass_direction
        )
        
        self.game_states[game_id] = game_state
        return game_state
    
    def get_game(self, game_id: str) -> Optional[GameState]:
        """Get game state by ID"""
        return self.game_states.get(game_id)
    
    def play_card(self, game_id: str, player_id: int, card: Card) -> GameState:
        """
        Process a player's move using Gymnasium wrapper (AI turns handled automatically)
        """
        wrapper = self.gym_wrappers.get(game_id)
        game_state = self.game_states.get(game_id)
        
        if not wrapper or not game_state:
            raise ValueError("Game not found")
        
        if game_state.current_player != player_id:
            raise ValueError(f"Not your turn (current player: {game_state.current_player}, you: {player_id})")
        
        # Convert card to action
        internal_game = HeartsGame()
        internal_game._env = wrapper.env._base_env
        internal_game._timestep = wrapper.env._last_timestep
        action = internal_game.card_to_action(card)
        
        print(f"CARD PLAYED: {card} (action: {action})")
        
        # Step the gym environment (AI players will move automatically)
        try:
            result = wrapper.step(action)
        except ValueError as e:
            # Extract legal actions for better error message
            legal_actions = wrapper.get_legal_actions()
            legal_cards = [internal_game.action_to_card(a) for a in legal_actions]
            raise ValueError(f"Invalid move: {card}. Legal cards: {[str(c) for c in legal_cards]}")
        
        # Update game state from result
        game_state.current_trick.append((player_id, card))
        
        # Update state based on gym result
        if result['terminated']:
            game_state.game_over = True
            game_state.is_passing_phase = False
            all_rewards = result['all_rewards']
            # Convert rewards to scores (OpenSpiel returns negative values)
            for i, player in enumerate(game_state.players):
                player.score = int(26-all_rewards[i])  # Negate to get positive penalty scores
            winner_id = min(range(4), key=lambda i: game_state.players[i].score)
            game_state.winner = winner_id
            # Clear all hands
            for player in game_state.players:
                player.hand = []
            game_state.current_trick = []
        else:
            # Update from gym state
            game_state.current_player = result['current_player']
            game_state.observation = result['observation'].tolist()
            game_state.legal_actions = result['legal_actions']
            
            # Update internal game reference
            internal_game._timestep = wrapper.env._last_timestep
            game_state.hearts_broken = internal_game.hearts_broken
            game_state.is_passing_phase = internal_game.is_passing_phase(0)
            game_state.pass_direction = internal_game.get_pass_direction(0)
            
            # Update all players' hands
            for player in game_state.players:
                player.hand = internal_game.get_player_hand(player.id)
        
        self.game_states[game_id] = game_state
        return game_state
    
    def reset_game(self, game_id: str) -> GameState:
        """Reset a game to initial state"""
        if game_id in self.gym_wrappers:
            # Shutdown old wrapper
            self.gym_wrappers[game_id].shutdown()
            del self.gym_wrappers[game_id]
            del self.game_states[game_id]
        return self.create_game(game_id)
    
    def delete_game(self, game_id: str) -> bool:
        """Delete a game session"""
        if game_id in self.gym_wrappers:
            # Shutdown wrapper before deleting
            self.gym_wrappers[game_id].shutdown()
            del self.gym_wrappers[game_id]
            del self.game_states[game_id]
            return True
        return False


