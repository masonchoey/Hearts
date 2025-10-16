/**
 * Card Component
 * Renders a single playing card
 */
import './Card.css';

const Card = ({ card }) => {
  const getSuitSymbol = (suit) => {
    const symbols = {
      'H': '♥',
      'D': '♦',
      'C': '♣',
      'S': '♠'
    };
    return symbols[suit] || suit;
  };

  const getSuitColor = (suit) => {
    return (suit === 'H' || suit === 'D') ? 'red' : 'black';
  };

  const getRankDisplay = (rank) => {
    const displays = {
      'T': '10',
      'J': 'J',
      'Q': 'Q',
      'K': 'K',
      'A': 'A'
    };
    return displays[rank] || rank;
  };

  const suitSymbol = getSuitSymbol(card.suit);
  const suitColor = getSuitColor(card.suit);
  const rankDisplay = getRankDisplay(card.rank);

  return (
    <div className={`card ${suitColor}`}>
      <div className="card-corner top-left">
        <div className="card-rank">{rankDisplay}</div>
        <div className="card-suit">{suitSymbol}</div>
      </div>
      <div className="card-center">
        <span className="card-suit-large">{suitSymbol}</span>
      </div>
      <div className="card-corner bottom-right">
        <div className="card-rank">{rankDisplay}</div>
        <div className="card-suit">{suitSymbol}</div>
      </div>
    </div>
  );
};

export default Card;


