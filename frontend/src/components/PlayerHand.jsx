/**
 * Player Hand Component
 * Displays clickable cards for the human player
 */
import { useEffect } from 'react';
import { useGameStore } from '../hooks/useGameState';
import Card from './Card';
import './PlayerHand.css';

const PlayerHand = ({ cards, playerId, isCurrentPlayer }) => {
  const { gameId, playCard, selectedCard, selectedCards, selectCard, isLoading, gameState, passCards } = useGameStore();

  const isMultiplayer = gameId?.startsWith('mp_');
  const inPassingPhase = gameState?.is_passing_phase;
  const myPassSubmitted = gameState?.my_pass_submitted;
  // Multiplayer passing: all players can queue their pass independently of turn order
  const canInteract = isMultiplayer && inPassingPhase
    ? !myPassSubmitted
    : isCurrentPlayer;

  const handleCardClick = (card) => {
    if (!canInteract || isLoading) return;
    selectCard(card);
  };

  const handlePlayCard = () => {
    if (!selectedCard || !canInteract || isLoading) {
      console.warn('Cannot play card:', { selectedCard, canInteract, isLoading, currentPlayer: gameState?.current_player });
      return;
    }
    console.log('Attempting to play card:', { playerId, card: selectedCard, currentPlayer: gameState?.current_player });
    //check to see the difference between the updated state, and the current state, and then animate the cards
    console.log('Current State:', gameState);
    const updatedState = playCard(playerId, selectedCard);
    // const difference = updatedState.filter(card => !gameState.current_trick.includes(card));
    console.log('Updated State:', updatedState);
    // animateCards
  };

  const handlePassCards = () => {
    if (selectedCards.length !== 3 || !canInteract || isLoading) {
      console.warn('Cannot pass cards:', { selectedCards, canInteract, isLoading, currentPlayer: gameState?.current_player });
      return;
    }
    console.log('Attempting to pass cards:', { playerId, cards: selectedCards, currentPlayer: gameState?.current_player });
    passCards(playerId, selectedCards);
  };

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

  return (
    <div className="player-hand-container">
      <div className="player-hand">
        {cards.map((card, index) => (
          <div
            key={`${card.suit}-${card.rank}`}
            className={`card-wrapper ${isCardSelected(card) ? 'selected' : ''} ${isCardPlayed(card) ? 'played' : ''} ${!canInteract ? 'disabled' : ''}`}
            onClick={() => handleCardClick(card)}
            style={{ '--card-index': index }}
          >
            <Card card={card} />
          </div>
        ))}
      </div>
      
      {canInteract && gameState?.is_passing_phase && (
        <div className="passing-phase-controls">
          <div className="selected-cards-info">
            Selected: {selectedCards.length}/3 cards
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


