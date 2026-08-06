/**
 * Card Component
 * Renders a single, realistic playing card with proper pip layouts
 */
import './Card.css';

const SUIT_SYMBOLS = { H: '♥', D: '♦', C: '♣', S: '♠' };
const RANK_DISPLAY = { T: '10', J: 'J', Q: 'Q', K: 'K', A: 'A' };

/**
 * Standard pip layouts. Each entry is [xFraction, yFraction] within the pip
 * area. x: 0 = left column, 0.5 = center, 1 = right column. y: 0 = top,
 * 1 = bottom. Pips in the lower half are rotated 180deg (as on real cards).
 */
const PIP_LAYOUTS = {
  '2': [[0.5, 0.14], [0.5, 0.86]],
  '3': [[0.5, 0.14], [0.5, 0.5], [0.5, 0.86]],
  '4': [[0.16, 0.14], [0.84, 0.14], [0.16, 0.86], [0.84, 0.86]],
  '5': [[0.16, 0.14], [0.84, 0.14], [0.5, 0.5], [0.16, 0.86], [0.84, 0.86]],
  '6': [[0.16, 0.14], [0.84, 0.14], [0.16, 0.5], [0.84, 0.5], [0.16, 0.86], [0.84, 0.86]],
  '7': [[0.16, 0.14], [0.84, 0.14], [0.5, 0.32], [0.16, 0.5], [0.84, 0.5], [0.16, 0.86], [0.84, 0.86]],
  '8': [[0.16, 0.14], [0.84, 0.14], [0.5, 0.32], [0.16, 0.5], [0.84, 0.5], [0.5, 0.68], [0.16, 0.86], [0.84, 0.86]],
  '9': [[0.16, 0.14], [0.84, 0.14], [0.16, 0.38], [0.84, 0.38], [0.5, 0.5], [0.16, 0.62], [0.84, 0.62], [0.16, 0.86], [0.84, 0.86]],
  '10': [[0.16, 0.14], [0.84, 0.14], [0.5, 0.26], [0.16, 0.38], [0.84, 0.38], [0.16, 0.62], [0.84, 0.62], [0.5, 0.74], [0.16, 0.86], [0.84, 0.86]],
};

const Card = ({ card }) => {
  const suitSymbol = SUIT_SYMBOLS[card.suit] || card.suit;
  const suitColor = (card.suit === 'H' || card.suit === 'D') ? 'red' : 'black';
  const rankDisplay = RANK_DISPLAY[card.rank] || card.rank;

  const isFace = card.rank === 'J' || card.rank === 'Q' || card.rank === 'K';
  const isAce = card.rank === 'A';
  const pips = PIP_LAYOUTS[rankDisplay];

  const renderCenter = () => {
    if (isAce) {
      return (
        <div className="card-center">
          <span className="card-suit-ace">{suitSymbol}</span>
        </div>
      );
    }
    if (isFace) {
      return (
        <div className="card-center card-face">
          <div className="card-face-panel">
            <span className="card-face-letter">{rankDisplay}</span>
          </div>
        </div>
      );
    }
    if (pips) {
      return (
        <div className="card-center card-pips">
          {pips.map(([x, y], i) => (
            <span
              key={i}
              className={`card-pip ${y > 0.5 ? 'flipped' : ''}`}
              style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
            >
              {suitSymbol}
            </span>
          ))}
        </div>
      );
    }
    return (
      <div className="card-center">
        <span className="card-suit-large">{suitSymbol}</span>
      </div>
    );
  };

  const twoChar = rankDisplay.length > 1; // "10" — needs a narrower index

  return (
    <div className={`card ${suitColor} ${twoChar ? 'two-char' : ''}`}>
      <div className="card-corner top-left">
        <div className="card-rank">{rankDisplay}</div>
        <div className="card-suit">{suitSymbol}</div>
      </div>
      {renderCenter()}
      <div className="card-corner bottom-right">
        <div className="card-rank">{rankDisplay}</div>
        <div className="card-suit">{suitSymbol}</div>
      </div>
    </div>
  );
};

export default Card;
