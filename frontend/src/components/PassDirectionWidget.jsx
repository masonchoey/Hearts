/**
 * Pass Direction Widget
 * Shows the current pass direction in the corner of the game board
 */
import { useGameStore } from '../hooks/useGameState';
import './PassDirectionWidget.css';

const PassDirectionWidget = () => {
  const { gameState } = useGameStore();
  
  if (!gameState?.pass_direction) {
    return null;
  }

  const getDirectionIcon = (direction) => {
    switch (direction) {
      case 'No Pass':
        return '🚫';
      case 'Left':
        return '⬅️';
      case 'Across':
        return '↕️';
      case 'Right':
        return '➡️';
      default:
        return '❓';
    }
  };

  const getDirectionColor = (direction) => {
    switch (direction) {
      case 'No Pass':
        return '#95a5a6';
      case 'Left':
        return '#3498db';
      case 'Across':
        return '#9b59b6';
      case 'Right':
        return '#e67e22';
      default:
        return '#95a5a6';
    }
  };

  return (
    <div 
      className="pass-direction-widget"
      style={{ '--direction-color': getDirectionColor(gameState.pass_direction) }}
    >
      <div className="pass-direction-icon">
        {getDirectionIcon(gameState.pass_direction)}
      </div>
      <div className="pass-direction-text">
        Pass: {gameState.pass_direction}
      </div>
    </div>
  );
};

export default PassDirectionWidget;
