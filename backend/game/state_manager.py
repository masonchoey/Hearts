"""
Game State Manager
Manages multiple game sessions and coordinates between OpenSpiel and frontend
"""
from typing import Dict, Optional
from ..schemas.types import GameState, Player, Card
from .hearts_logic import HeartsGame
from ..models.hearts_model import HeartsAIModel
from dotenv import load_dotenv
import os

load_dotenv()


class GameStateManager:
    """
    Manages game sessions and coordinates between frontend and OpenSpiel
    """
    
    def __init__(self):
        self.games: Dict[str, HeartsGame] = {}
        self.game_states: Dict[str, GameState] = {}
        
        # Load AI model from checkpoint
        checkpoint_path = os.getenv("CHECKPOINT_PATH", "PPO_2025-10-07_04-21-40/PPO_hearts_env_self_play_1f830_00000_0_2025-10-07_04-21-40/checkpoint_000009")
        self.ai_model = HeartsAIModel(checkpoint_path=None)
        # self.ai_model = HeartsAIModel(checkpoint_path=checkpoint_path)
        
    def create_game(self, game_id: str) -> GameState:
        """
        Create a new game session
        """
        # Initialize OpenSpiel game
        game = HeartsGame()
        self.games[game_id] = game
        
        # Create initial game state
        players = [
            Player(id=0, name="You", is_ai=False, hand=[], score=0, round_score=0),
            Player(id=1, name="AI 1", is_ai=True, hand=[], score=0, round_score=0),
            Player(id=2, name="AI 2", is_ai=True, hand=[], score=0, round_score=0),
            Player(id=3, name="AI 3", is_ai=True, hand=[], score=0, round_score=0),
        ]
        
        # The OpenSpiel RL Environment automatically handles card dealing during reset
        # No need to manually process chance nodes
        
        current_player = game.current_player()
        
        # Get hands for all players
        for player in players:
            player.hand = game.get_player_hand(player.id)
        
        # Get observation and legal actions for current player
        observation = game.get_observation(current_player).tolist()
        legal_actions = game.get_legal_actions()
        
        # Check if we're in passing phase and get pass direction
        is_passing_phase = game.is_passing_phase(0)  # Check for human player (player 0)
        pass_direction = game.get_pass_direction(0)  # Get pass direction for human player
        
        game_state = GameState(
            players=players,
            current_trick=[],
            current_player=current_player,
            round_number=1,
            tricks_played=0,
            game_over=False,
            hearts_broken=game.hearts_broken,
            winner=None,
            observation=observation,
            legal_actions=legal_actions,
            is_passing_phase=is_passing_phase,
            pass_direction=pass_direction
        )
        
        self.game_states[game_id] = game_state
        
        # If AI goes first, process AI turns immediately
        if current_player != 0 and not game.is_terminal():
            game_state = self.process_ai_turns(game_id)
        
        return game_state
    
    def get_game(self, game_id: str) -> Optional[GameState]:
        """Get game state by ID"""
        return self.game_states.get(game_id)
    
    def play_card(self, game_id: str, player_id: int, card: Card) -> GameState:
        """
        Process a player's move
        """
        game = self.games.get(game_id)
        game_state = self.game_states.get(game_id)
        
        if not game or not game_state:
            raise ValueError("Game not found")
        
        if game_state.current_player != player_id:
            raise ValueError(f"Not your turn (current player: {game_state.current_player}, you: {player_id})")
        
        # Validate move
        print(f"CARD PLAYED: {card}")
        if not game.validate_move(player_id, card):
            print(f"Current player hand: {game.get_player_hand(player_id)}")
            legal_actions = game.get_legal_actions()
            print(f"Legal actions: {legal_actions}")
            legal_cards = [game.action_to_card(a) for a in legal_actions]
            print(f"Legal cards: {[str(c) for c in legal_cards]}")
            raise ValueError(f"Invalid move: {card} not in legal actions. Legal cards: {[str(c) for c in legal_cards]}")
        
        # Apply action to OpenSpiel
        action = game.card_to_action(card)
        game.apply_action(action)
        
        # Update game state
        game_state.current_trick.append((player_id, card))
        
        # Check if trick is complete
        if len(game_state.current_trick) == 4:
            # Trick completed, determine winner and update scores
            self._complete_trick(game_state, game)
        
        # Update current player and legal actions
        if not game.is_terminal():
            game_state.current_player = game.current_player()
            game_state.observation = game.get_observation(game_state.current_player).tolist()
            game_state.legal_actions = game.get_legal_actions()
            game_state.hearts_broken = game.hearts_broken
            game_state.is_passing_phase = game.is_passing_phase(0)  # Update passing phase status
            game_state.pass_direction = game.get_pass_direction(0)  # Update pass direction
        else:
            game_state.game_over = True
            scores = game.get_scores()
            for i, player in enumerate(game_state.players):
                player.score = scores[i]
            # Determine winner (lowest score)
            winner_id = min(range(4), key=lambda i: game_state.players[i].score)
            game_state.winner = winner_id
        
        # Update all players' hands
        if not game.is_terminal():
            if player_id == 0:
                game_state.players[0].hand = game.get_player_hand(0)
        
        self.game_states[game_id] = game_state
        return game_state
    
    def pass_cards(self, game_id: str, player_id: int, cards: list[Card]) -> GameState:
        """
        Process passing of 3 cards during the passing phase
        """
        game = self.games.get(game_id)
        game_state = self.game_states.get(game_id)
        
        if not game or not game_state:
            raise ValueError("Game not found")
        
        if not game_state.is_passing_phase:
            raise ValueError("Not in passing phase")
        
        if game_state.current_player != player_id:
            raise ValueError(f"Not your turn (current player: {game_state.current_player}, you: {player_id})")
        
        if len(cards) != 3:
            raise ValueError("Must pass exactly 3 cards")
        
        # Validate that all cards are in the player's hand
        player_hand = game.get_player_hand(player_id)
        for card in cards:
            if card not in player_hand:
                raise ValueError(f"Card {card} not in player's hand")
        
        # Apply each card as a pass action
        for card in cards:
            action = game.card_to_action(card)
            game.apply_action(action)
        
        # Update game state
        if not game.is_terminal():
            game_state.current_player = game.current_player()
            game_state.observation = game.get_observation(game_state.current_player).tolist()
            game_state.legal_actions = game.get_legal_actions()
            game_state.hearts_broken = game.hearts_broken
            game_state.is_passing_phase = game.is_passing_phase(0)  # Update passing phase status
            game_state.pass_direction = game.get_pass_direction(0)  # Update pass direction
        else:
            game_state.game_over = True
            scores = game.get_scores()
            for i, player in enumerate(game_state.players):
                player.score = scores[i]
            winner_id = min(range(4), key=lambda i: game_state.players[i].score)
            game_state.winner = winner_id
        
        # Update all players' hands
        if not game.is_terminal():
            for player in game_state.players:
                player.hand = game.get_player_hand(player.id)
        
        self.game_states[game_id] = game_state
        return game_state
    
    def process_ai_turns(self, game_id: str) -> GameState:
        """
        Process AI player turns until it's the human player's turn or trick is complete
        """
        game = self.games.get(game_id)
        game_state = self.game_states.get(game_id)
        
        if not game or not game_state:
            raise ValueError("Game not found")
        
        # Process AI turns
        while not game.is_terminal() and game_state.players[game_state.current_player].is_ai:
            player_id = game_state.current_player
            
            # Get AI action
            observation = game.get_observation(player_id)
            legal_actions = game.get_legal_actions()
            action = self.ai_model.get_action(observation, legal_actions)
            
            # Convert action to card
            card = game.action_to_card(action)
            
            # Apply action
            game.apply_action(action)
            game_state.current_trick.append((player_id, card))
            
            # Check if trick is complete
            if len(game_state.current_trick) == 4:
                self._complete_trick(game_state, game)
            
            # Update state
            if not game.is_terminal():
                game_state.current_player = game.current_player()
                game_state.observation = game.get_observation(game_state.current_player).tolist()
                game_state.legal_actions = game.get_legal_actions()
                game_state.hearts_broken = game.hearts_broken
                game_state.is_passing_phase = game.is_passing_phase(0)  # Update passing phase status
                game_state.pass_direction = game.get_pass_direction(0)  # Update pass direction
            else:
                game_state.game_over = True
                scores = game.get_scores()
                for i, player in enumerate(game_state.players):
                    player.score = scores[i]
                winner_id = min(range(4), key=lambda i: game_state.players[i].score)
                game_state.winner = winner_id
                break
        
        # Update all players' hands
        if not game.is_terminal():
            for player in game_state.players:
                player.hand = game.get_player_hand(player.id)
        
        self.game_states[game_id] = game_state
        return game_state
    
    def _complete_trick(self, game_state: GameState, game: HeartsGame):
        """Complete a trick and update scores"""
        # Clear current trick
        game_state.current_trick = []
        game_state.tricks_played += 1
        
        # Update scores (OpenSpiel handles this internally)
        # We'll fetch the updated scores when the game is terminal
    
    def reset_game(self, game_id: str) -> GameState:
        """Reset a game to initial state"""
        if game_id in self.games:
            del self.games[game_id]
            del self.game_states[game_id]
        return self.create_game(game_id)
    
    def delete_game(self, game_id: str) -> bool:
        """Delete a game session"""
        if game_id in self.games:
            del self.games[game_id]
            del self.game_states[game_id]
            return True
        return False


