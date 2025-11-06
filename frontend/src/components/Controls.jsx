/**
 * Game Controls Component
 * Provides buttons for game actions
 */
import { useGameStore } from '../hooks/useGameState';
import './Controls.css';

const Controls = ({ onShowScoreboard }) => {
  const { resetCurrentGame, isLoading, gameState } = useGameStore();

  return (
    <div className="controls">
      <button
        className="control-button scoreboard"
        onClick={onShowScoreboard}
      >
        📊 Scores
      </button>
      
      <button
        className="control-button reset"
        onClick={resetCurrentGame}
        disabled={isLoading}
      >
        🔄 New Round
      </button>
      
      {gameState && (
        <div className="game-info">
          <span>Round: {gameState.round_number}</span>
          <span className={gameState.hearts_broken ? 'hearts-broken' : ''}>
            {gameState.hearts_broken ? '♥ Broken' : '♥ Not Broken'}
          </span>
        </div>
      )}
    </div>
  );
};

export default Controls;


