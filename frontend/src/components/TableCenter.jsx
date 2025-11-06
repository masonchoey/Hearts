/**
 * Table Center Component
 * Displays cards played in the current trick
 */
import { useGameStore } from '../hooks/useGameState';
import Card from './Card';
import './TableCenter.css';

const TableCenter = () => {
  const { animatedTrick, animationDelay } = useGameStore();
  
  // Use animatedTrick during animation, otherwise show the actual trick
  // animatedTrick will be [] when not animating, so we'll use trick in that case
  const displayTrick = animatedTrick.length > 0 ? animatedTrick : [];
  
  // Calculate glide duration based on animation delay
  // Scale it appropriately: min 0.3s, max 2s
  const glideDuration = Math.max(300, Math.min(animationDelay * 2, 2000));
  
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
        {displayTrick && displayTrick.length > 0 ? (
          <div className="trick-cards">
            {displayTrick.map(([playerId, card], index) => (
              <div
                key={`${playerId}-${card.suit}-${card.rank}-${index}`}
                className={`trick-card position-${getCardPosition(playerId)}`}
                style={{ 
                  '--card-index': index,
                  '--glide-duration': `${glideDuration}ms`
                }}
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


