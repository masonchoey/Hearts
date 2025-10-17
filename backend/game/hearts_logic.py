"""
Hearts Game Logic
Wraps OpenSpiel Hearts environment for game rules and state management
Uses OpenSpiel RL Environment wrapper (same as training code) to prevent segfaults
"""
import pyspiel
import numpy as np
from typing import List, Tuple, Optional
from ..schemas.types import Card, Player
from open_spiel.python.rl_environment import Environment as OSPSingle


class HeartsGame:
    """
    Hearts game logic using OpenSpiel RL Environment
    Uses the same wrapper as the training code to ensure stability
    """
    
    # Card mappings for OpenSpiel
    # OpenSpiel uses: action = rank * 4 + suit (NOT suit * 13 + rank)
    # Suit order: Clubs=0, Diamonds=1, Hearts=2, Spades=3
    SUIT_MAP = {"C": 0, "D": 1, "H": 2, "S": 3}
    RANK_MAP = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
    REVERSE_SUIT_MAP = {v: k for k, v in SUIT_MAP.items()}
    REVERSE_RANK_MAP = {v: k for k, v in RANK_MAP.items()}
    
    def __init__(self):
        """Initialize Hearts game with OpenSpiel RL Environment wrapper"""
        # Use the same RL Environment wrapper as training code
        self._env = OSPSingle(pyspiel.load_game("hearts"), players=4)
        self._timestep = None
        self.hearts_broken = False
        self.reset()
        
    def reset(self):
        """Reset game to initial state"""
        self._timestep = self._env.reset()
        self.hearts_broken = False
        
    def get_observation(self, player_id: int) -> np.ndarray:
        """
        Get observation vector for a player using RL Environment API.
        Returns info_state (5088 values) which contains:
        - Pass direction (4 values)
        - Dealt hand (52 values)
        - Passed cards (52 values)
        - Received cards (52 values)
        - Current hand (52 values at indices 160-212)
        - Points (144 values)
        - Trick history (4732 values)
        
        This is the same observation format used during training.
        """
        if self._timestep is None:
            return np.zeros(5088, dtype=np.float32)
        
        # Use info_state from the timestep (same as training code)
        info_state = self._timestep.observations["info_state"][player_id]
        return np.array(info_state, dtype=np.float32)
    
    def get_legal_actions(self) -> List[int]:
        """Get list of legal action indices for current player"""
        if self._timestep is None or self._timestep.last():
            return []
        
        current_player = self._timestep.observations["current_player"]
        return self._timestep.observations["legal_actions"][current_player]
    
    def current_player(self) -> int:
        """Get current player (0-3, or -1 for chance/terminal)"""
        if self._timestep is None:
            return -1
        return self._timestep.observations["current_player"]
    
    def is_terminal(self) -> bool:
        """Check if game is over"""
        if self._timestep is None:
            return False
        return self._timestep.last()
    
    def get_returns(self) -> List[float]:
        """Get final scores (negative for Hearts)"""
        if self._timestep is None or not self._timestep.last():
            return [0.0] * 4
        return self._timestep.rewards if self._timestep.rewards else [0.0] * 4
    
    def apply_action(self, action: int):
        """Apply an action to the game state using RL Environment API"""
        if self._timestep is None or self._timestep.last():
            raise ValueError("Cannot apply action to terminal or uninitialized state")
        
        # Step the environment with the action (wrapped in list for OSPSingle API)
        self._timestep = self._env.step([action])
        
        # Update hearts_broken based on the action (if it's a heart)
        # Hearts have suit=2, so action % 4 == 2
        if not self.is_passing_phase(0):
            if action % 4 == 2:
                self.hearts_broken = True
        
    def card_to_action(self, card: Card) -> int:
        """
        Convert a Card object to an OpenSpiel action index
        OpenSpiel uses action = rank * 4 + suit
        """
        suit_idx = self.SUIT_MAP[card.suit]
        rank_idx = self.RANK_MAP[card.rank]
        return rank_idx * 4 + suit_idx
    
    def action_to_card(self, action: int) -> Card:
        """
        Convert an OpenSpiel action index to a Card object
        OpenSpiel uses action = rank * 4 + suit, so:
        - suit = action % 4
        - rank = action // 4
        """
        suit_idx = action % 4
        rank_idx = action // 4
        suit = self.REVERSE_SUIT_MAP[suit_idx]
        rank = self.REVERSE_RANK_MAP[rank_idx]
        return Card(suit=suit, rank=rank)
    
    def get_player_hand(self, player_id: int) -> list[Card]:
        """
        Extract player's hand from the observation vector
        Returns list of Card objects sorted by suit (C, D, H, S) then rank (2-A)
        
        The observation vector contains:
        - Dealt hand at indices 4-56 (52 values)
        - Current hand at indices 160-212 (52 values)
        
        Each index represents a card (1 if present, 0 if not), where:
        card_index = suit * 13 + rank
        """
        if self._timestep is None:
            return []
        
        hand = []
        
        try:
            # Get observation vector for this player
            observation = self.get_observation(player_id)
            
            # Use current hand (indices 160-212)
            # This represents the cards currently in the player's hand
            hand_slice = observation[160:212]
            
            # Convert the one-hot encoding to Card objects
            # hand_slice has 52 values, one for each possible card
            # Index i corresponds to card with action index i
            for card_action in range(52):
                if hand_slice[card_action] > 0:  # Card is in hand
                    card = self.action_to_card(card_action)
                    hand.append(card)
            
            # Sort by suit first (C=0, D=1, H=2, S=3), then by rank (2=0 to A=12)
            hand.sort(key=lambda c: (self.SUIT_MAP[c.suit], self.RANK_MAP[c.rank]))
            
            return hand
            
        except Exception as e:
            # Failed to parse hand - return empty list
            print(f"Error parsing hand for player {player_id}: {e}")
            return []
    
    def validate_move(self, player_id: int, card: Card) -> bool:
        """
        Check if a move is legal
        """
        if self.current_player() != player_id:
            return False
        
        action = self.card_to_action(card)
        return action in self.get_legal_actions()
    
    def get_trick_info(self) -> Tuple[List[Tuple[int, Card]], Optional[int], int]:
        """
        Get information about current trick
        Returns: (cards_played, winner, points)
        Note: This is simplified as OpenSpiel doesn't directly expose trick info
        """
        return [], None, 0
    
    def get_scores(self) -> List[int]:
        """
        Get current scores for all players
        In Hearts, scores are cumulative penalty points
        """
        if self.is_terminal():
            returns = self.get_returns()
            # Convert returns (negative) to positive scores
            # OpenSpiel returns are normalized, need to scale appropriately
            return [26-int(r) for r in returns]
        return [0, 0, 0, 0]
    
    def is_passing_phase(self, player_id: int) -> bool:
        """
        Check if we're currently in the passing phase for a given player.
        
        The passing phase occurs at the start of the game and when:
        1. Pass direction is 1, 2, or 3 (Left, Across, or Right)
        2. Player has 13 cards in their hand (initial deal)
        3. Player hasn't passed any cards yet (passed cards section is empty)
        4. Player hasn't received any cards yet (received cards section is empty)
        5. History of tricks is empty
        """
        if self._timestep is None:
            return False
        
        try:
            observation = self.get_observation(player_id)
            
            # Check pass direction (first 4 values)
            pass_direction = observation[0:4]
            
            # Only proceed if pass direction is 1, 2, or 3 (not 0 = No Pass)
            if pass_direction[0] == 1:
                return False
            
            # Check if player has passed any cards yet (indices 56-108)
            passed_cards = observation[56:108]
            cards_passed = int(np.sum(passed_cards))

            #Check if player has received any cards yet (indices 108-160)
            received_cards = observation[108:160]
            cards_received = int(np.sum(received_cards))

            # Check if history of tricks is empty (indices 356-5087)
            trick_history = observation[356:5087]
            history_of_tricks = int(np.sum(trick_history))
            
            # Passing phase: < 3 cards passed or < 3 cards received
            return (cards_passed < 3 or cards_received < 3) or history_of_tricks == 0
            
        except Exception as e:
            print(f"Error checking passing phase for player {player_id}: {e}")
            return False
    
    def get_pass_direction(self, player_id: int) -> str:
        """
        Get the pass direction as a human-readable string.
        
        Returns:
            str: "No Pass", "Left", "Across", or "Right"
        """
        if self._timestep is None:
            return "No Pass"
        
        try:
            observation = self.get_observation(player_id)
            
            # Check pass direction (first 4 values)
            pass_direction = observation[0:4]
            pass_direction_idx = np.argmax(pass_direction) if np.sum(pass_direction) > 0 else 0
            
            direction_map = {
                0: "No Pass",
                1: "Left", 
                2: "Across",
                3: "Right"
            }
            
            return direction_map.get(pass_direction_idx, "No Pass")
            
        except Exception as e:
            print(f"Error getting pass direction for player {player_id}: {e}")
            return "No Pass"


