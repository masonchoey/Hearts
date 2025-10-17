/**
 * Main Game Board Component
 * Arranges 4 players around a table
 */
import { useState } from 'react';
import { useGameStore } from '../hooks/useGameState';
import PlayerHand from './PlayerHand';
import TableCenter from './TableCenter';
import Card from './Card';
import PassDirectionWidget from './PassDirectionWidget';
import './GameBoard.css';

const GameBoard = () => {
  const { gameState, error, clearError } = useGameStore();
  const [debugMode, setDebugMode] = useState(false);

  if (!gameState) {
    return <div className="loading">Loading game...</div>;
  }

  const players = gameState.players || [];
  const humanPlayer = players[0];
  const topPlayer = players[2];
  const leftPlayer = players[3];
  const rightPlayer = players[1];

  // Helper function to render AI hands
  const renderAIHand = (player, orientation = 'horizontal') => {
    console.log('renderAIHand', player, orientation, debugMode);
    if (debugMode && player?.hand) {
      return (
        <div className={`ai-hand horizontal debug-mode`}>
          {player.hand.map((card, index) => (
            <div key={`${card.suit}-${card.rank}-${index}`} className="ai-card-wrapper">
              <Card card={card} className={`card-${orientation === 'horizontal' ? 'horizontal' : ''}`}/>
            </div>
          ))}
        </div>
      );
    } else {
      // Show card backs - backend now sends all players' hands
      const cardCount = player?.hand?.length || 13;
      return (
        <div className={`ai-hand ${orientation === 'vertical' ? 'vertical' : ''}`}>
          {Array(cardCount).fill(0).map((_, i) => (
            <div key={i} className={orientation === 'vertical' ? 'card-back-horizontal' : 'card-back'} />
          ))}
        </div>
      );
    }
  };

  return (
    <div className="game-board">
      {/* Pass Direction Widget */}
      <PassDirectionWidget />
      
      {/* Debug Toggle */}
      <div className="debug-toggle">
        <label>
          <input
            type="checkbox"
            checked={debugMode}
            onChange={(e) => setDebugMode(e.target.checked)}
          />
          <span>Debug: Show AI Cards</span>
        </label>
      </div>

      {error && (
        <div className="error-banner" onClick={clearError}>
          <strong>Error:</strong> {error}
          <button onClick={clearError} style={{ marginLeft: '10px' }}>✕</button>
        </div>
      )}

      {/* Top Player (AI) */}
      <div className="player-area player-top">
        <div className="player-info">
          <span className="player-name">{topPlayer?.name}</span>
          <span className="player-score">Score: {topPlayer?.score || 0}</span>
        </div>
        {renderAIHand(topPlayer, 'horizontal')}
      </div>

      {/* Left Player (AI) */}
      <div className="player-area player-left">
        <div className="player-info">
          <span className="player-name">{leftPlayer?.name}</span>
          <span className="player-score">Score: {leftPlayer?.score || 0}</span>
        </div>
        {renderAIHand(leftPlayer, 'vertical')}
      </div>

      {/* Right Player (AI) */}
      <div className="player-area player-right">
        <div className="player-info">
          <span className="player-name">{rightPlayer?.name}</span>
          <span className="player-score">Score: {rightPlayer?.score || 0}</span>
        </div>
        {renderAIHand(rightPlayer, 'vertical')}
      </div>

      {/* Center Table */}
      <TableCenter trick={gameState.current_trick || []} />

      {/* Bottom Player (Human) */}
      <div className="player-area player-bottom">
        <div className="player-info">
          <span className="player-name you">{humanPlayer?.name}</span>
          <span className="player-score">Score: {humanPlayer?.score || 0}</span>
        </div>
        <PlayerHand 
          cards={humanPlayer?.hand || []} 
          playerId={0}
          isCurrentPlayer={gameState.current_player === 0}
        />
      </div>

      {/* Game Over Modal */}
      {gameState.game_over && (
        <div className="game-over-modal">
          <div className="modal-content">
            <h2>Game Over!</h2>
            <div className="final-scores">
              {players.map((player) => (
                <div 
                  key={player.id} 
                  className={`score-row ${player.id === gameState.winner ? 'winner' : ''}`}
                >
                  <span>{player.name}</span>
                  <span>{player.score} points</span>
                </div>
              ))}
            </div>
            <button 
              className="play-again-button"
              onClick={() => useGameStore.getState().resetCurrentGame()}
            >
              Play Again
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default GameBoard;


