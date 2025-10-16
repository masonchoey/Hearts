"""
Hearts Game Logic
Wraps OpenSpiel Hearts environment for game rules and state management
Uses OpenSpiel RL Environment wrapper (same as training code) to prevent segfaults
"""
import pyspiel
import numpy as np
from typing import List, Tuple, Optional
from schemas.types import Card, Player
from open_spiel.python.rl_environment import Environment as OSPSingle


class HeartsGame:
    """
    Hearts game logic using OpenSpiel RL Environment
    Uses the same wrapper as the training code to ensure stability
    """
    
    # Card mappings for OpenSpiel
    SUIT_MAP = {"C": 0, "D": 1, "S": 2, "H": 3}
    RANK_MAP = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
    REVERSE_SUIT_MAP = {v: k for k, v in SUIT_MAP.items()}
    REVERSE_RANK_MAP = {v: k for k, v in RANK_MAP.items()}
    
    def __init__(self):
        """Initialize Hearts game with OpenSpiel RL Environment wrapper"""
        # Use the same RL Environment wrapper as training code
        self._env = OSPSingle(pyspiel.load_game("hearts"), players=4)
        self._timestep = None
        self.hearts_broken = False
        # Access the underlying OpenSpiel state for hand extraction
        self._state = None
        self.reset()
        
    def reset(self):
        """Reset game to initial state"""
        self._timestep = self._env.reset()
        self._state = self._env.get_state  # This is the OpenSpiel state object
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
        self._state = self._env.get_state  # Update state reference
        
        # Update hearts_broken based on the action (if it's a heart)
        # Heart cards are actions 39-51 (suit 3 * 13 + rank)
        if 39 <= action <= 51:
            self.hearts_broken = True
        
    def card_to_action(self, card: Card) -> int:
        """
        Convert a Card object to an OpenSpiel action index
        OpenSpiel uses action = suit * 13 + rank
        """
        suit_idx = self.SUIT_MAP[card.suit]
        rank_idx = self.RANK_MAP[card.rank]
        return suit_idx * 13 + rank_idx
    
    def action_to_card(self, action: int) -> Card:
        """
        Convert an OpenSpiel action index to a Card object
        """
        suit_idx = action // 13
        rank_idx = action % 13
        suit = self.REVERSE_SUIT_MAP[suit_idx]
        rank = self.REVERSE_RANK_MAP[rank_idx]
        return Card(suit=suit, rank=rank)
    
    def get_player_hand(self, player_id: int) -> List[Card]:
        """
        Extract player's hand from the underlying OpenSpiel state
        Returns list of Card objects
        
        Parses the information_state_string which contains the hand in a readable format.
        """
        if self._state is None:
            return []
        
        hand = []
        state = self._state
        
        try:
            # Get the information state string for this player
            # Format example:
            # Pass Direction: Left
            # 
            # Hand: 
            # S KQT92
            # H 2
            # D AKJ
            # C Q974
            info_state_str = state.information_state_string(player_id)
            
            # Parse the hand section
            # Find the "Hand:" marker and extract cards from the following lines
            lines = info_state_str.split('\n')
            in_hand_section = False
            
            for line in lines:
                line = line.strip()
                
                if line == "Hand:":
                    in_hand_section = True
                    continue
                
                if not in_hand_section:
                    continue
                
                # Empty line or start of next section ends the hand
                if line == "" or (line and not line[0] in ['S', 'H', 'D', 'C']):
                    break
                
                # Parse card line (e.g., "S KQT92" or "H 2" or "S none")
                if line and line[0] in ['S', 'H', 'D', 'C']:
                    suit = line[0]
                    cards_part = line[2:].strip()  # Get the cards part
                    
                    # Handle "none" case (no cards in this suit)
                    if cards_part.lower() == 'none':
                        continue
                    
                    # Remove spaces and parse each rank character
                    cards_str = cards_part.replace(' ', '')
                    for rank_char in cards_str:
                        hand.append(Card(suit=suit, rank=rank_char))
            
            return hand
            
        except Exception as e:
            # Failed to parse hand - return empty list
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
            return [-int(r * 26) for r in returns]
        return [0, 0, 0, 0]


