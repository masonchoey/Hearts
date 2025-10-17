/**
 * Game State Management using Zustand
 */
import { create } from 'zustand';
import { startGame, getState, playMove, resetGame, processAITurns, processSingleAIMove, passCards } from '../api/backend';

export const useGameStore = create((set, get) => ({
  // State
  gameId: null,
  gameState: null,
  isLoading: false,
  error: null,
  selectedCard: null,
  selectedCards: [], // For passing phase - array of up to 3 cards
  animationDelay: 200, // Default 1 second (in milliseconds)

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
      
      // If AI goes first, trigger AI moves
      get().processAIMovesWithDelay();
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
      set({
        gameState: data.state,
        isLoading: false,
        selectedCard: null,
      });
      
      // Trigger AI moves processing with animation delays
      get().processAIMovesWithDelay();
    } catch (error) {
      console.error('Play card error:', error.response?.data || error);
      const errorMsg = error.response?.data?.detail || error.message;
      set({ error: errorMsg, isLoading: false });
    }
  },

  processAIMovesWithDelay: async () => {
    const { gameId, gameState, isLoading, animationDelay } = get();
    if (!gameId || !gameState || isLoading) return;
    
    // Check if it's an AI's turn
    const currentPlayer = gameState.players[gameState.current_player];
    if (!currentPlayer?.is_ai || gameState.game_over) {
      //clear the completed trick
      if (gameState.current_trick?.length === 4) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        set({ gameState: { ...gameState, current_trick: [] } });
      }
      return;
    }

    // If there's a completed trick (4 cards), add extra delay before clearing it
    if (gameState.current_trick?.length === 4) {
      await new Promise(resolve => setTimeout(resolve, 1500));
    }

    // Wait for animation delay (user-configurable)
    await new Promise(resolve => setTimeout(resolve, animationDelay));

    // Process one AI move
    try {
      const data = await processSingleAIMove(gameId);
      set({ gameState: data.state });
      
      // Recursively process next AI move if needed
      get().processAIMovesWithDelay();
    } catch (error) {
      console.error('AI move error:', error);
      set({ error: error.message });
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
      });
      
      // If AI goes first, trigger AI moves
      get().processAIMovesWithDelay();
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
      set({
        gameState: data.state,
        isLoading: false,
        selectedCards: [],
      });
      
      // Trigger AI moves processing with animation delays
      get().processAIMovesWithDelay();
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
}));


