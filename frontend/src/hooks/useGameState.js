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
  animationDelay: 200, // Default 1 second (in milliseconds)
  showGameOverModal: false, // Control visibility of game over modal

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
      });
      
      // No need to process AI moves - the gym environment
      // automatically advances to the human's turn
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

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
    const { gameId, isLoading } = get();
    if (!gameId) return;
    
    // Prevent double-clicks
    if (isLoading) {
      console.warn('Already processing a move, ignoring...');
      return;
    }

    console.log('Playing card:', { playerId, card });
    set({ isLoading: true, error: null });
    try {
      const data = await playMove(gameId, playerId, card);
      console.log('Move successful, new state:', data.state);
      
      // The gym environment automatically processes all AI moves
      // So the returned state already has it back to the human's turn
      set({
        gameState: data.state,
        isLoading: false,
        selectedCard: null,
      });
      
      // Check if game is over and schedule modal display
      if (data.state.game_over) {
        get().scheduleGameOverModal();
      }
    } catch (error) {
      console.error('Play card error:', error.response?.data || error);
      const errorMsg = error.response?.data?.detail || error.message;
      set({ error: errorMsg, isLoading: false });
    }
  },

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
      });
      
      // No need to process AI moves - the gym environment
      // automatically advances to the human's turn
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

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
    const { gameId, isLoading } = get();
    if (!gameId) return;
    
    // Prevent double-clicks
    if (isLoading) {
      console.warn('Already processing a move, ignoring...');
      return;
    }

    console.log('Passing cards:', { playerId, cards });
    set({ isLoading: true, error: null });
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
      set({ error: errorMsg, isLoading: false });
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
  },
}));


