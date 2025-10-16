/**
 * Player Hand Component
 * Displays clickable cards for the human player
 */
import { useGameStore } from '../hooks/useGameState';
import Card from './Card';
import './PlayerHand.css';

const PlayerHand = ({ cards, playerId, isCurrentPlayer }) => {
  const { playCard, selectedCard, selectCard, isLoading, gameState } = useGameStore();

  const handleCardClick = (card) => {
    if (!isCurrentPlayer || isLoading) return;

    if (selectedCard?.suit === card.suit && selectedCard?.rank === card.rank) {
      // If same card clicked, deselect
      selectCard(null);
    } else {
      // Select card
      selectCard(card);
    }
  };

  const handlePlayCard = () => {
    if (!selectedCard || !isCurrentPlayer || isLoading) {
      console.warn('Cannot play card:', { selectedCard, isCurrentPlayer, isLoading, currentPlayer: gameState?.current_player });
      return;
    }
    console.log('Attempting to play card:', { playerId, card: selectedCard, currentPlayer: gameState?.current_player });
    playCard(playerId, selectedCard);
  };

  const isCardSelected = (card) => {
    return selectedCard?.suit === card.suit && selectedCard?.rank === card.rank;
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
      
      {isCurrentPlayer && selectedCard && (
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


