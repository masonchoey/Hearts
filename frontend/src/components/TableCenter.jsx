/**
 * Table Center Component
 * Displays cards played in the current trick.
 *
 * Each played card animates out of the player it belongs to:
 *  - the human's card flies from its exact spot in the hand (already face-up)
 *  - a CPU's card flies from a random card in that CPU's fanned hand and flips
 *    from its back to its face as it travels to the table.
 */
import { useLayoutEffect, useRef } from 'react';
import { useGameStore } from '../hooks/useGameState';
import Card from './Card';
import { slotForIndex } from './seatLayout';
import './TableCenter.css';

const POSITIONS = { 0: 'bottom', 1: 'left', 2: 'top', 3: 'right' };
const REST_ROT = { bottom: -3, top: 5, left: 7, right: -7 };

// Resting transform (centering + tilt) for a settled trick card at a position.
const restTransform = (position, restRot) => {
  const center = (position === 'left' || position === 'right') ? 'translateY(-50%)' : 'translateX(-50%)';
  return `${center} rotate(${restRot}deg)`;
};

// Pick a random card element from a CPU's fanned hand to launch the play from.
const randomHandRect = (position) => {
  const nodes = document.querySelectorAll(`.player-${position} .ai-card-wrapper`);
  if (!nodes.length) return null;
  const el = nodes[Math.floor(Math.random() * nodes.length)];
  return el.getBoundingClientRect();
};

const TrickCard = ({ playerId, card, index, glideDuration }) => {
  const ref = useRef(null);
  const flipRef = useRef(null);
  const position = POSITIONS[playerId] || 'bottom';
  const isHuman = position === 'bottom';
  const jitter = ((index % 3) - 1) * 2.5;
  const restRot = REST_ROT[position] + jitter;

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Determine where this card should launch from.
    let fromRect = null;
    if (isHuman) {
      fromRect = useGameStore.getState().playedFromRect;
    } else {
      fromRect = randomHandRect(position);
    }

    // No source found → fall back to the CSS "throw from side" animation.
    if (!fromRect) return;

    // Suppress the CSS throw; we drive this card by hand with a FLIP.
    el.style.animation = 'none';
    const rest = restTransform(position, restRot);
    el.style.transition = 'none';
    el.style.transform = rest;
    const target = el.getBoundingClientRect();

    const dx = (fromRect.left + fromRect.width / 2) - (target.left + target.width / 2);
    const dy = (fromRect.top + fromRect.height / 2) - (target.top + target.height / 2);
    const startScale = isHuman && target.width ? fromRect.width / target.width : 1;

    // Place it at the source, then transition to rest.
    el.style.transform = `translate(${dx}px, ${dy}px) ${rest} scale(${startScale})`;
    if (!isHuman && flipRef.current) {
      flipRef.current.style.transition = 'none';
      flipRef.current.style.transform = 'rotateY(180deg)'; // start showing the back
    }
    // Force reflow so the start state is committed before we animate.
    void el.offsetWidth;

    const ease = 'cubic-bezier(0.22, 0.61, 0.36, 1)';
    el.style.transition = `transform ${glideDuration}ms ${ease}`;
    el.style.transform = rest;
    if (!isHuman && flipRef.current) {
      flipRef.current.style.transition = `transform ${glideDuration}ms ${ease}`;
      flipRef.current.style.transform = 'rotateY(0deg)'; // flip to the face
    }

    if (isHuman) useGameStore.setState({ playedFromRect: null });
  }, [position, restRot, glideDuration, isHuman]);

  return (
    <div
      ref={ref}
      className={`trick-card position-${position}`}
      style={{
        '--card-index': index,
        '--glide-duration': `${glideDuration}ms`,
        '--rest-rot': `${restRot}deg`,
      }}
    >
      {isHuman ? (
        <Card card={card} />
      ) : (
        <div className="trick-flip" ref={flipRef}>
          <div className="trick-flip-face trick-flip-front">
            <Card card={card} />
          </div>
          <div className="trick-flip-face trick-flip-back">
            <div className="trick-card-back" />
          </div>
        </div>
      )}
    </div>
  );
};

const TableCenter = () => {
  const { animatedTrick, animationDelay, gameState } = useGameStore();

  // Use animatedTrick during animation, otherwise show the actual trick
  // animatedTrick will be [] when not animating, so we'll use trick in that case
  const displayTrick = animatedTrick.length > 0 ? animatedTrick : [];

  // Calculate glide duration based on animation delay
  // Scale it appropriately: min 0.3s, max 2s
  const glideDuration = Math.max(300, Math.min(animationDelay * 2, 2000));

  const playerCount = gameState?.player_count || gameState?.players?.length || 4;

  // Player ids here are already rotated so the viewer is index 0 (bottom).
  const getCardPosition = (playerId) => slotForIndex(playerCount, playerId);

  return (
    <div className="table-center">
      <div className="table-surface">
        {displayTrick && displayTrick.length > 0 ? (
          <div className="trick-cards">
            {displayTrick.map(([playerId, card], index) => (
              <TrickCard
                key={`${playerId}-${card.suit}-${card.rank}-${index}`}
                playerId={playerId}
                card={card}
                index={index}
                glideDuration={glideDuration}
              />
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
