/**
 * Scoreboard Component
 * Displays current game scores in a modal
 */
import { useGameStore } from '../hooks/useGameState';
import './Scoreboard.css';

const Scoreboard = ({ onClose }) => {
  const { gameState } = useGameStore();

  if (!gameState) return null;

  const players = gameState.players || [];
  const rules = gameState.rules || {};
  const activeRules = [
    rules.jd_bonus && 'J♦ = −10',
    rules.ten_club_doubler && '10♣ doubles',
  ].filter(Boolean);

  return (
    <div className="scoreboard-modal" onClick={onClose}>
      <div className="scoreboard-content" onClick={(e) => e.stopPropagation()}>
        <h2>Scoreboard</h2>
        {activeRules.length > 0 && (
          <p className="scoreboard-rules">Rules: {activeRules.join(' · ')}</p>
        )}

        <div className="scoreboard-table">
          <div className="scoreboard-header">
            <span>Player</span>
            <span>Current Score</span>
            <span>Round Points</span>
          </div>
          
          {players
            .slice()
            .sort((a, b) => a.score - b.score)
            .map((player, index) => (
              <div 
                key={player.id} 
                className={`scoreboard-row ${index === 0 ? 'leader' : ''}`}
              >
                <span className="player-name-col">
                  {index === 0 && '👑 '}
                  {player.name}
                </span>
                <span className="score-col">{player.score}</span>
                <span className="round-col">{player.round_score || 0}</span>
              </div>
            ))}
        </div>

        <button className="close-button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default Scoreboard;


