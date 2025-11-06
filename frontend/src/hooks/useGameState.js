/**
 * Game State Management using Zustand
 */
import { create } from 'zustand';
import { startGame, getState, playMove, resetGame, processAITurns, passCards } from '../api/backend';

export const useGameStore = create((set, get) => ({
  // State
  gameId: null,
  gameState: null,
  isLoading: false,
  error: null,
  selectedCard: null,
  selectedCards: [], // For passing phase - array of up to 3 cards
  animationDelay: 350, // Default 250ms (in milliseconds)
  showGameOverModal: false, // Control visibility of game over modal
  animatedTrick: [], // Temporary state for card animations - array of [playerId, card] tuples
  hasAnimatedInitialTrick: false, // Track if we've animated the initial trick on game start
  passingAnimations: [], // Array of { fromPlayerId, toPlayerId, cards, startPos, endPos } for passing animations

  // Actions
  startNewGame: async () => {
    set({ isLoading: true, error: null });
    try {
      const data = await startGame();
      set({
        gameId: data.game_id,
        gameState: data.state,
        isLoading: false,
        selectedCards: [],
        hasAnimatedInitialTrick: false, // Reset animation flag
        animatedTrick: [], // Clear any previous animations
      });
      
      // Animation will be handled by GameBoard when it mounts
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

  // LEGACY: refreshState - Not used in current implementation
  // The gym environment automatically handles state updates
  refreshState: async () => {
    const { gameId } = get();
    if (!gameId) return;

    try {
      const data = await getState(gameId);
      set({ gameState: data.state });
    } catch (error) {
      set({ error: error.message });
    }
  },

  playCard: async (playerId, card) => {
    const { gameId, isLoading, gameState, animationDelay } = get();
    if (!gameId) return null;
    
    // Prevent double-clicks
    if (isLoading) {
      console.warn('Already processing a move, ignoring...');
      return null;
    }

    console.log('Playing card:', { playerId, card });
    set({ isLoading: true, error: null });
    
    // Save the current trick before the move
    const oldTrick = gameState?.current_trick || [];
    
    try {
      const data = await playMove(gameId, playerId, card);
      console.log('Move successful, new state:', data.state);
      
      // Get the new trick from the response
      // cardsToAnimate is the cards that were played before the current_trick but after the player's last move.
      const cardsToAnimate = data.state.move_sequence || [];
      const currentTrick = data.state.current_trick || [];
      // Calculate which cards were just played (difference between old and new trick)
      console.log('Cards to animate:', cardsToAnimate);
      
      console.log(`Animating ${cardsToAnimate.length} new cards`);
      
      // If there are new cards to animate, show them one by one
      if (cardsToAnimate.length > 0 && animationDelay > 0) {
        //Animate in two different steps: First, animate the cards in move_sequence but not in current_trick, then animate the cards in current_trick
        for (let i = 0; i < cardsToAnimate.length; i++) {
          const [playerId, card] = cardsToAnimate[i];
          const currentState = get().gameState;
          
          // Immediately remove the card from the player's hand
          const updatedPlayers = currentState.players.map(player => {
            if (player.id === playerId) {
              return {
                ...player,
                hand: player.hand.filter(c => c.suit !== card.suit || c.rank !== card.rank)
              };
            }
            return player;
          });
          
          set({ 
            animatedTrick: [...get().animatedTrick, [playerId, card]],
            gameState: {
              ...currentState,
              players: updatedPlayers
            }
          });
          await new Promise(resolve => setTimeout(resolve, animationDelay));
        }

        // Small delay to show the complete trick before updating state
        await new Promise(resolve => setTimeout(resolve, Math.max(500, Math.min(animationDelay * 2.5, 1500))));
        //Clear the previous animated trick
        set({ animatedTrick: [] });

        //Animate the cards in current_trick
        for (let i = 0; i < currentTrick.length; i++) {
          const [playerId, card] = currentTrick[i];
          const currentState = get().gameState;
          
          // Immediately remove the card from the player's hand
          const updatedPlayers = currentState.players.map(player => {
            if (player.id === playerId) {
              return {
                ...player,
                hand: player.hand.filter(c => c.suit !== card.suit || c.rank !== card.rank)
              };
            }
            return player;
          });
          
          set({ 
            animatedTrick: [...get().animatedTrick, [playerId, card]],
            gameState: {
              ...currentState,
              players: updatedPlayers
            }
          });
          await new Promise(resolve => setTimeout(resolve, animationDelay));
        }
      
      }
      
      // Update the actual game state and clear animation
      const updatedState = {
        gameState: data.state, 
        isLoading: false,
        selectedCard: null,
        // animatedTrick: [], // Clear animation after state update
      };
      
      set(updatedState);
      
      // Check if game is over and schedule modal display
      if (data.state.game_over) {
        get().scheduleGameOverModal();
      }
      
      return updatedState;
    } catch (error) {
      console.error('Play card error:', error.response?.data || error);
      const errorMsg = error.response?.data?.detail || error.message;
      const errorState = { error: errorMsg, isLoading: false};
      set(errorState);
      return errorState;
    }
  },

  // LEGACY: processAIMovesWithDelay - Not used in current implementation
  // The gym environment automatically processes all AI moves when playCard() is called
  processAIMovesWithDelay: async () => {
    // NOTE: The gym environment automatically processes all AI moves
    // when playCard() is called, so AI moves don't need to be processed separately.
    // This function is kept for backward compatibility but only handles
    // trick clearing delays for better UX.
    
    const { gameState, animationDelay } = get();
    if (!gameState || gameState.game_over) return;
    
    // Clear completed tricks after a delay for better visual feedback
    if (gameState.current_trick?.length === 4) {
      const delay = Math.max(500, Math.min(animationDelay * 7, 1500));
      await new Promise(resolve => setTimeout(resolve, delay));
      set({ gameState: { ...gameState, current_trick: [] } });
    }
  },

  setAnimationDelay: (delay) => {
    set({ animationDelay: delay });
  },

  resetCurrentGame: async () => {
    const { gameId } = get();
    if (!gameId) return;

    set({ isLoading: true, error: null });
    try {
      const data = await resetGame(gameId);
      set({
        gameState: data.state,
        isLoading: false,
        selectedCard: null,
        selectedCards: [],
        showGameOverModal: false,
        animatedTrick: [],
        hasAnimatedInitialTrick: false, // Reset animation flag for new game
      });
      
      // Animation will be handled by GameBoard when it mounts
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

  // LEGACY: processAITurns - Not used in current implementation
  // The gym environment automatically processes all AI moves when playCard() is called
  processAITurns: async () => {
    const { gameId, isLoading } = get();
    if (!gameId || isLoading) return;

    set({ isLoading: true, error: null });
    try {
      const data = await processAITurns(gameId);
      set({
        gameState: data.state,
        isLoading: false,
      });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

  passCards: async (playerId, cards) => {
    const { gameId, isLoading, animationDelay, gameState } = get();
    if (!gameId) return;
    
    // Prevent double-clicks
    if (isLoading) {
      console.warn('Already processing a move, ignoring...');
      return;
    }

    console.log('Passing cards:', { playerId, cards });
    set({ isLoading: true, error: null });
    
    // Calculate pass direction and destination player
    const passDirection = gameState?.pass_direction;
    let toPlayerId = null;
    
    if (passDirection === 'Left') {
      toPlayerId = (playerId + 1) % 4;
    } else if (passDirection === 'Right') {
      toPlayerId = (playerId + 3) % 4;
    } else if (passDirection === 'Across') {
      toPlayerId = (playerId + 2) % 4;
    }
    
    // Remove cards from hand immediately for visual feedback
    const currentGameState = get().gameState;
    const updatedPlayers = currentGameState.players.map(player => {
      if (player.id === playerId) {
        return {
          ...player,
          hand: player.hand.filter(c => 
            !cards.some(passedCard => passedCard.suit === c.suit && passedCard.rank === c.rank)
          )
        };
      }
      return player;
    });
    
    // Update state to remove cards from hand
    set({
      gameState: {
        ...currentGameState,
        players: updatedPlayers
      }
    });
    
    // Create animations for ALL players (human + AI)
    // Human player passes actual cards, AI players pass card backs
    const allAnimations = [];
    
    if (toPlayerId !== null) {
      // Human player animation with actual cards
      allAnimations.push({
        fromPlayerId: playerId,
        toPlayerId: toPlayerId,
        cards: cards,
        passDirection: passDirection,
        isHuman: true
      });
      
      // AI player animations with card backs
      for (let aiPlayerId = 0; aiPlayerId < 4; aiPlayerId++) {
        if (aiPlayerId === playerId) continue; // Skip human player
        
        // Calculate AI's destination based on pass direction
        let aiToPlayerId = null;
        if (passDirection === 'Left') {
          aiToPlayerId = (aiPlayerId + 1) % 4;
        } else if (passDirection === 'Right') {
          aiToPlayerId = (aiPlayerId + 3) % 4;
        } else if (passDirection === 'Across') {
          aiToPlayerId = (aiPlayerId + 2) % 4;
        }
        
        if (aiToPlayerId !== null) {
          // For AI, we'll pass placeholder cards (will be rendered as card backs)
          allAnimations.push({
            fromPlayerId: aiPlayerId,
            toPlayerId: aiToPlayerId,
            cards: [{suit: 'back', rank: '?'}, {suit: 'back', rank: '?'}, {suit: 'back', rank: '?'}],
            passDirection: passDirection,
            isHuman: false
          });
        }
      }
    }
    
    // Trigger animation for all players
    if (allAnimations.length > 0 && animationDelay > 0) {
      set({ passingAnimations: allAnimations });
      
      // Animation duration scales with slider (min 800ms, max 5 seconds)
      // Add extra 100ms to ensure cards fully arrive before clearing
      const animationDuration = Math.max(800, Math.min(animationDelay * 7, 5000)) + 100;
      await new Promise(resolve => setTimeout(resolve, animationDuration));
      
      // Clear animation
      set({ passingAnimations: [] });
    }
    
    try {
      const data = await passCards(gameId, playerId, cards);
      console.log('Pass successful, new state:', data.state);
      
      // The gym environment automatically processes all AI moves (including AI passing)
      // So the returned state already has it back to the human's turn
      set({
        gameState: data.state,
        isLoading: false,
        selectedCards: [],
      });
    } catch (error) {
      console.error('Pass cards error:', error.response?.data || error);
      const errorMsg = error.response?.data?.detail || error.message;
      set({ error: errorMsg, isLoading: false, passingAnimations: [] });
    }
  },

  selectCard: (card) => {
    const { gameState, selectedCards } = get();
    
    // If we're in passing phase, handle multiple selection
    if (gameState?.is_passing_phase) {
      const isSelected = selectedCards.some(c => c.suit === card.suit && c.rank === card.rank);
      
      if (isSelected) {
        // Remove card from selection
        set({ selectedCards: selectedCards.filter(c => !(c.suit === card.suit && c.rank === card.rank)) });
      } else if (selectedCards.length < 3) {
        // Add card to selection (max 3)
        set({ selectedCards: [...selectedCards, card] });
      }
    } else {
      // Normal single card selection for playing
      if (get().selectedCard?.suit === card.suit && get().selectedCard?.rank === card.rank) {
        set({ selectedCard: null });
      } else {
        set({ selectedCard: card });
      }
    }
  },

  clearError: () => {
    set({ error: null });
  },

  scheduleGameOverModal: () => {
    // Wait 2 seconds before showing the game over modal
    setTimeout(() => {
      set({ showGameOverModal: true });
    }, 2000);
  },

  hideGameOverModal: () => {
    set({ showGameOverModal: false });
  },

  showGameOverModalNow: () => {
    set({ showGameOverModal: true });
  }
}));


