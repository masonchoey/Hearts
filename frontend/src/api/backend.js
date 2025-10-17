/**
 * API Communication Layer
 * Handles all communication with FastAPI backend
 */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Start a new game
 * @returns {Promise} Game ID and initial state
 */
export const startGame = async () => {
  const response = await api.post('/start');
  return response.data;
};

/**
 * Get current game state
 * @param {string} gameId - Game session ID
 * @returns {Promise} Current game state
 */
export const getState = async (gameId) => {
  const response = await api.get(`/state/${gameId}`);
  return response.data;
};

/**
 * Play a card
 * @param {string} gameId - Game session ID
 * @param {number} playerId - Player ID (0-3)
 * @param {Object} card - Card object {suit, rank}
 * @returns {Promise} Updated game state
 */
export const playMove = async (gameId, playerId, card) => {
  const response = await api.post(`/play/${gameId}`, {
    player_id: playerId,
    card: card,
  });
  return response.data;
};

/**
 * Reset game to initial state
 * @param {string} gameId - Game session ID
 * @returns {Promise} Reset game state
 */
export const resetGame = async (gameId) => {
  const response = await api.post(`/reset/${gameId}`);
  return response.data;
};

/**
 * Process AI turns
 * @param {string} gameId - Game session ID
 * @returns {Promise} Updated game state
 */
export const processAITurns = async (gameId) => {
  const response = await api.post(`/ai-turns/${gameId}`);
  return response.data;
};

/**
 * Pass 3 cards during passing phase
 * @param {string} gameId - Game session ID
 * @param {number} playerId - Player ID (0-3)
 * @param {Array} cards - Array of 3 card objects [{suit, rank}, ...]
 * @returns {Promise} Updated game state
 */
export const passCards = async (gameId, playerId, cards) => {
  const response = await api.post(`/pass/${gameId}`, {
    player_id: playerId,
    cards: cards,
  });
  return response.data; 
};

/**
 * Delete a game session
 * @param {string} gameId - Game session ID
 * @returns {Promise} Success message
 */
export const deleteGame = async (gameId) => {
  const response = await api.delete(`/game/${gameId}`);
  return response.data;
};

export default api;


