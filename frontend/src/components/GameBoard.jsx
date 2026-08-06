/**
 * Main Game Board Component
 * Arranges 4 players around a table
 */
import { useState, useEffect, useRef } from 'react';
import { useGameStore } from '../hooks/useGameState';
import PlayerHand from './PlayerHand';
import TableCenter from './TableCenter';
import Card from './Card';
import PassDirectionWidget from './PassDirectionWidget';
import PassingAnimation from './PassingAnimation';
import { slotForIndex, orientationForSlot } from './seatLayout';
import './GameBoard.css';

const GameBoard = () => {
  const { gameId, gameState, error, clearError, animationDelay, setAnimationDelay, showGameOverModal, hideGameOverModal, showGameOverModalNow, animatedTrick, hasAnimatedInitialTrick, passingAnimations } = useGameStore();
  const [debugMode, setDebugMode] = useState(false);
  const isAnimatingRef = useRef(false);
  const gameBoardRef = useRef(null);

  // Refs for player positions (one per possible table slot)
  const playerRefs = {
    bottom: useRef(null),
    top: useRef(null),
    left: useRef(null),
    right: useRef(null),
    'top-left': useRef(null),
    'top-right': useRef(null),
  };

  // Animate initial cards when board first mounts
  useEffect(() => {
    let cancelled = false;
    
    const animateInitialTrick = async () => {
      const trick = gameState?.current_trick || [];
      
      // Only animate if:
      // 1. Game is loaded
      // 2. Haven't animated yet
      // 3. Not currently animating (check ref)
      // 4. There are cards to animate
      if (!gameId || hasAnimatedInitialTrick || isAnimatingRef.current || trick.length === 0) {
        return;
      }

      console.log(`Animating initial trick with ${trick.length} cards`);
      
      // Mark that we're animating
      isAnimatingRef.current = true;
      
      // Clear any previous animations
      useGameStore.setState({ animatedTrick: [] });
      
      // Animate each card one by one
      for (let i = 0; i < trick.length && !cancelled; i++) {
        const [playerId, card] = trick[i];
        console.log(`Animating card ${i}: Player ${playerId}, ${card.rank}${card.suit}`);
        
        // Immediately remove the card from the player's hand
        useGameStore.setState(state => {
          const currentGameState = state.gameState;
          const updatedPlayers = currentGameState.players.map(player => {
            if (player.id === playerId) {
              return {
                ...player,
                hand: player.hand.filter(c => c.suit !== card.suit || c.rank !== card.rank)
              };
            }
            return player;
          });
          
          return { 
            animatedTrick: [...state.animatedTrick, [playerId, card]],
            gameState: {
              ...currentGameState,
              players: updatedPlayers
            }
          };
        });
        await new Promise(resolve => setTimeout(resolve, animationDelay));
      }
      
      // Mark as animated AFTER all cards are shown
      if (!cancelled) {
        useGameStore.setState({ hasAnimatedInitialTrick: true });
      }
      
      isAnimatingRef.current = false;
    };

    animateInitialTrick();
    
    return () => { 
      cancelled = true;
      isAnimatingRef.current = false;
    };
  }, [gameId, gameState?.current_trick, animationDelay, hasAnimatedInitialTrick]);

  if (!gameState) {
    return <div className="loading">Loading game...</div>;
  }

  const players = gameState.players || [];
  const playerCount = gameState.player_count || players.length || 4;
  const humanPlayer = players[0];
  // Opponents are rotated indices 1..N-1, each placed in a table slot.
  const opponents = players.slice(1).map((player, i) => {
    const rotatedIndex = i + 1;
    const slot = slotForIndex(playerCount, rotatedIndex);
    return { player, slot, orientation: orientationForSlot(slot) };
  });

  // Helper function to render AI hands
  const renderAIHand = (player, orientation = 'horizontal') => {
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
      const cardCount = player?.hand?.length;
      return (
        <div className={`ai-hand ${orientation === 'vertical' ? 'vertical' : ''}`}>
          {Array(cardCount).fill(0).map((_, i) => (
            <div key={i} className={orientation === 'vertical' ? 'card-back-horizontal' : 'card-back'} />
          ))}
        </div>
      );
    }
  };

  // Multiplayer per-seat status badge (disconnected). Undefined in single-player.
  const statusBadge = (player) => {
    if (player?.disconnected) {
      return <span className="player-badge player-badge--offline">🔌 Disconnected</span>;
    }
    return null;
  };

  return (
    <div className="game-board" ref={gameBoardRef}>
      {/* Pass Direction Widget */}
      <PassDirectionWidget />
      
      {/* Bottom Left Controls Panel */}
      <div className="game-controls-bottom-left">
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

        {/* Animation Speed Slider */}
        <div className="animation-speed-control">
          <label>
            <span>Card Animation Delay: {(animationDelay / 1000).toFixed(2)}s</span>
            <input
              type="range"
              min="10"
              max="1000"
              step="10"
              value={animationDelay}
              onChange={(e) => setAnimationDelay(Number(e.target.value))}
              className="speed-slider"
            />
          </label>
        </div>
      </div>

      {/* Top Right Controls Panel */}
      <div className="game-controls-top-right">
        {/* Show Game Over Button - appears when game is over but modal is hidden */}
        {gameState.game_over && !showGameOverModal && (
          <button 
            className="show-game-over-button"
            onClick={showGameOverModalNow}
            aria-label="Show game over results"
          >
            Show Results
          </button>
        )}
      </div>

      {error && (
        <div className="error-banner" onClick={clearError}>
          <strong>Error:</strong> {error}
          <button onClick={clearError} style={{ marginLeft: '10px' }}>✕</button>
        </div>
      )}

      {/* Opponent players, placed around the table by seat slot */}
      {opponents.map(({ player, slot, orientation }) => (
        <div
          key={player?.id ?? slot}
          className={`player-area player-${slot} ${player?.disconnected ? 'player-area--offline' : ''}`}
          ref={playerRefs[slot]}
        >
          <div className="player-info">
            <span className="player-name">{player?.name}</span>
            <span className="player-score">Score: {player?.score || 0}</span>
            {statusBadge(player)}
          </div>
          {renderAIHand(player, orientation)}
        </div>
      ))}

      {/* Center Table */}
      <TableCenter/>

      {/* Bottom Player (Human) */}
      <div className="player-area player-bottom" ref={playerRefs.bottom}>
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
      
      {/* Passing Animation Overlay */}
      {passingAnimations.length > 0 && (
        <PassingAnimation 
          animations={passingAnimations}
          playerRefs={playerRefs}
          containerRef={gameBoardRef}
          animationDelay={animationDelay}
        />
      )}

      {/* Game Over Modal */}
      {gameState.game_over && showGameOverModal && (
        <div className="game-over-modal">
          <div className="modal-content">
            <button 
              className="modal-close-button"
              onClick={hideGameOverModal}
              aria-label="Close game over modal"
            >
              ✕
            </button>
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


