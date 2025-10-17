/**
 * Table Center Component
 * Displays cards played in the current trick
 */
import Card from './Card';
import './TableCenter.css';

const TableCenter = ({ trick, isPassingPhase }) => {
  // Map player positions for display
  const getCardPosition = (playerId) => {
    const positions = {
      0: 'bottom',
      3: 'right',
      2: 'top',
      1: 'left'
    };
    return positions[playerId] || 'bottom';
  };

  return (
    <div className="table-center">
      <div className="table-surface">
        {trick && trick.length > 0 && !isPassingPhase ? (
          <div className="trick-cards">
            {trick.map(([playerId, card], index) => (
              <div
                key={`${playerId}-${card.suit}-${card.rank}`}
                className={`trick-card position-${getCardPosition(playerId)}`}
                style={{ '--card-index': index }}
              >
                <Card card={card} />
              </div>
            ))}
          </div>
        ) : (
          <div className="table-placeholder">
            <span>♥</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default TableCenter;


