/**
 * Player Hand Component
 * Displays clickable cards for the human player
 */
import { useGameStore } from '../hooks/useGameState';
import Card from './Card';
import './PlayerHand.css';

const PlayerHand = ({ cards, playerId, isCurrentPlayer }) => {
  const { playCard, selectedCard, selectedCards, selectCard, isLoading, gameState, passCards } = useGameStore();

  const handleCardClick = (card) => {
    if (!isCurrentPlayer || isLoading) return;
    selectCard(card);
  };

  const handlePlayCard = () => {
    if (!selectedCard || !isCurrentPlayer || isLoading) {
      console.warn('Cannot play card:', { selectedCard, isCurrentPlayer, isLoading, currentPlayer: gameState?.current_player });
      return;
    }
    console.log('Attempting to play card:', { playerId, card: selectedCard, currentPlayer: gameState?.current_player });
    playCard(playerId, selectedCard);
  };

  const handlePassCards = () => {
    if (selectedCards.length !== 3 || !isCurrentPlayer || isLoading) {
      console.warn('Cannot pass cards:', { selectedCards, isCurrentPlayer, isLoading, currentPlayer: gameState?.current_player });
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

  return (
    <div className="player-hand-container">
      <div className="player-hand">
        {cards.map((card, index) => (
          <div
            key={`${card.suit}-${card.rank}`}
            className={`card-wrapper ${isCardSelected(card) ? 'selected' : ''} ${!isCurrentPlayer ? 'disabled' : ''}`}
            onClick={() => handleCardClick(card)}
            style={{ '--card-index': index }}
          >
            <Card card={card} />
          </div>
        ))}
      </div>
      
      {isCurrentPlayer && gameState?.is_passing_phase && (
        <div className="passing-phase-controls">
          <div className="selected-cards-info">
            Selected: {selectedCards.length}/3 cards
          </div>
          {selectedCards.length === 3 && (
            <button
              className="pass-cards-button"
              onClick={handlePassCards}
              disabled={isLoading}
            >
              {isLoading ? 'Passing...' : 'Pass Cards'}
            </button>
          )}
        </div>
      )}
      
      {isCurrentPlayer && !gameState?.is_passing_phase && selectedCard && (
        <button
          className="play-card-button"
          onClick={handlePlayCard}
          disabled={isLoading}
        >
          {isLoading ? 'Playing...' : 'Play Card'}
        </button>
      )}
      
      {!isCurrentPlayer && (
        <div className="waiting-message">
          Waiting for other players...
        </div>
      )}
    </div>
  );
};

export default PlayerHand;


