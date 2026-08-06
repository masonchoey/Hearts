/**
 * Player Hand Component
 * Displays clickable cards for the human player
 */
import { useEffect, useLayoutEffect } from 'react';
import { useGameStore } from '../hooks/useGameState';
import Card from './Card';
import './PlayerHand.css';

const PlayerHand = ({ cards, playerId, isCurrentPlayer }) => {
  const { gameId, playCard, selectedCard, selectedCards, selectCard, isLoading, gameState, passCards, enableDealAnimation, receivedCards, receivedFromPos, animationDelay } = useGameStore();

  // When cards are received from a pass, fly them into their hand slots from
  // the seat that passed them — mirroring how the player's cards leave, and
  // timed to match the other passing packets so it all moves together.
  useLayoutEffect(() => {
    if (!receivedCards?.length || !receivedFromPos) return;
    const src = document.querySelector(`.player-${receivedFromPos}`);
    if (!src) return;
    const s = src.getBoundingClientRect();
    const srcX = s.left + s.width / 2;
    const srcY = s.top + s.height / 2;
    const flyDur = Math.max(800, Math.min(animationDelay * 5, 5000)) / 1000; // seconds, matches the packets
    const els = document.querySelectorAll('.player-hand .card-wrapper.received .card');
    els.forEach((el) => {
      const r = el.getBoundingClientRect();
      const dx = srcX - (r.left + r.width / 2);
      const dy = srcY - (r.top + r.height / 2);
      // The card lives inside a rotated wrapper (the fan angle), so a plain
      // translate would travel along the card's tilted axis, not the true
      // screen path. Counter-rotate the offset by -fanAngle so it flies
      // straight from the opponent's hand to its slot.
      const wrapper = el.closest('.card-wrapper');
      const rot = parseFloat(wrapper?.style.getPropertyValue('--card-rot')) || 0;
      const rad = (-rot * Math.PI) / 180;
      const tx = dx * Math.cos(rad) - dy * Math.sin(rad);
      const ty = dx * Math.sin(rad) + dy * Math.cos(rad);
      el.style.transition = 'none';
      el.style.transform = `translate(${tx}px, ${ty}px) scale(0.9)`;
      // eslint-disable-next-line no-unused-expressions
      el.offsetWidth; // commit start position
      el.style.transition = `transform ${flyDur}s cubic-bezier(0.22, 0.61, 0.36, 1)`;
      el.style.transform = '';
    });
  }, [receivedCards, receivedFromPos, animationDelay]);

  const isMultiplayer = gameId?.startsWith('mp_');
  const inPassingPhase = gameState?.is_passing_phase;
  const myPassSubmitted = gameState?.my_pass_submitted;
  // Multiplayer passing: all players can queue their pass independently of turn order
  const canInteract = isMultiplayer && inPassingPhase
    ? !myPassSubmitted
    : isCurrentPlayer;

  const handleCardClick = (card) => {
    if (!canInteract || isLoading) return;
    // During play (not passing), clicking an already-selected card plays it.
    if (!gameState?.is_passing_phase && isCardSelected(card)) {
      handlePlayCard();
      return;
    }
    selectCard(card);
  };

  const handlePlayCard = () => {
    if (!selectedCard || !canInteract || isLoading) {
      console.warn('Cannot play card:', { selectedCard, canInteract, isLoading, currentPlayer: gameState?.current_player });
      return;
    }
    // Capture the on-screen position of the selected card so the table
    // animation can launch it out of the player's hand.
    const selEl = document.querySelector('.player-hand .card-wrapper.selected .card');
    if (selEl) {
      const r = selEl.getBoundingClientRect();
      useGameStore.setState({
        playedFromRect: { left: r.left, top: r.top, width: r.width, height: r.height },
      });
    }
    const updatedState = playCard(playerId, selectedCard);
  };

  const handlePassCards = () => {
    if (selectedCards.length !== 3 || !canInteract || isLoading) {
      console.warn('Cannot pass cards:', { selectedCards, canInteract, isLoading, currentPlayer: gameState?.current_player });
      return;
    }
    // Capture each selected card's on-screen slot (in the same order as the
    // cards being passed) so the pass animation launches each card out of its
    // real position in the hand.
    const rects = selectedCards.map(sc => {
      const el = document.querySelector(
        `.player-hand .card-wrapper[data-card="${sc.suit}-${sc.rank}"] .card`
      );
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    }).filter(Boolean);
    useGameStore.setState({ passedFromRects: rects });
    passCards(playerId, selectedCards);
  };

  const isCardReceived = (card) =>
    receivedCards?.some(c => c.suit === card.suit && c.rank === card.rank);

  const isCardSelected = (card) => {
    if (gameState?.is_passing_phase) {
      return selectedCards.some(c => c.suit === card.suit && c.rank === card.rank);
    } else {
      return selectedCard?.suit === card.suit && selectedCard?.rank === card.rank;
    }
  };

  const isCardPlayed = (card) => {
    // Check if this card is in the current trick
    // current_trick format: [[playerId, card], ...]
    return gameState?.current_trick?.some(([playerId, trickCard]) => 
      trickCard.suit === card.suit && trickCard.rank === card.rank
    );
  };

  // Handle keyboard events
  useEffect(() => {
    const handleKeyPress = (event) => {
      // Only handle spacebar when it's the player's turn and not loading
      if (event.code === 'Space' && canInteract && !isLoading) {
        event.preventDefault(); // Prevent page scroll
        
        // handlePlayCard();
        if (gameState?.is_passing_phase) {
          // In passing phase, spacebar passes the selected cards
          handlePassCards();
        } else {
          // In normal play, spacebar plays the selected card
          handlePlayCard();
        }
      }
    };

    // Add event listener
    document.addEventListener('keydown', handleKeyPress);

    // Cleanup
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [canInteract, isLoading, gameState?.is_passing_phase, selectedCard, selectedCards]);

  // Fan geometry: each card is rotated and lifted along an arc so the hand
  // curves like real cards held in the hand (highest in the middle).
  const n = cards.length;
  const mid = (n - 1) / 2;
  const anglePerCard = n > 1 ? Math.min(5.5, 58 / (n - 1)) : 0; // degrees between adjacent cards
  const liftCurve = 1.5; // arc depth (px per unit²) — higher = middle rises more above the edges
  const step = 50; // horizontal distance between adjacent cards

  const fanStyle = (index) => {
    const dist = index - mid;
    const rot = dist * anglePerCard;
    // Anchor the outermost cards at the baseline (lift 0) and raise the middle
    // upward, so a deeper arc peaks in the center without dropping the edges.
    const lift = (Math.pow(dist, 2) - Math.pow(mid, 2)) * liftCurve;
    return {
      '--card-index': index,
      '--offset-x': `${dist * step}px`,
      '--card-rot': `${rot}deg`,
      '--card-lift': `${lift}px`,
      zIndex: index,
    };
  };

  return (
    <div className="player-hand-container">
      <div className={`player-hand ${enableDealAnimation ? 'deal-in' : ''}`}>
        {cards.map((card, index) => (
          <div
            key={`${card.suit}-${card.rank}`}
            data-card={`${card.suit}-${card.rank}`}
            className={`card-wrapper ${isCardSelected(card) ? 'selected' : ''} ${isCardPlayed(card) ? 'played' : ''} ${isCardReceived(card) ? 'received' : ''} ${!canInteract ? 'disabled' : ''}`}
            onClick={() => handleCardClick(card)}
            style={fanStyle(index)}
          >
            <Card card={card} />
          </div>
        ))}
      </div>
      
      {canInteract && gameState?.is_passing_phase && (
        <div className="passing-phase-controls">
          <div className={`selected-cards-info ${selectedCards.length === 3 ? 'ready' : ''}`}>
            {selectedCards.length === 3 ? '✓ Ready to pass' : `Selected: ${selectedCards.length}/3 cards`}
          </div>
          <button
            className="pass-cards-button"
            onClick={handlePassCards}
            disabled={isLoading || selectedCards.length !== 3}
          >
            {isLoading ? 'Passing...' : 'Pass Cards'}
          </button>
        </div>
      )}
      
      <div className="player-controls">
        {isMultiplayer && inPassingPhase && myPassSubmitted && (
          <div className="waiting-message">
            Pass submitted — waiting for others ({gameState.passes_submitted?.length ?? 0}/4)
          </div>
        )}

        {!canInteract && !(isMultiplayer && inPassingPhase && myPassSubmitted) && (
          <div className="waiting-message">
            Waiting for other players...
          </div>
        )}
        
        {canInteract && !gameState?.is_passing_phase && (
          <button
            className="play-card-button"
            onClick={handlePlayCard}
            disabled={isLoading || !selectedCard}
          >
            {isLoading ? 'Playing...' : 'Play Card'}
          </button>
        )}
      </div>
    </div>
  );
};

export default PlayerHand;


