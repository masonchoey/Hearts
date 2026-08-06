/**
 * PassingAnimation Component
 * Animates cards being passed between players
 */
import { useEffect, useState, useRef } from 'react';
import Card from './Card';
import './PassingAnimation.css';

const PassingAnimation = ({ animations, playerRefs, containerRef: gameBoardRef, animationDelay }) => {
  const [cardPositions, setCardPositions] = useState([]);
  const overlayRef = useRef(null);
  const animationKeyRef = useRef(0);
  
  // Calculate animation duration based on slider (min 800ms, max 5 seconds)
  const animationDuration = Math.max(800, Math.min(animationDelay * 5, 5000)) / 1000; // Convert to seconds
  const fadeDelay = animationDuration * 0.85; // Fade only at the very end, so cards arrive at the hand before merging in

  useEffect(() => {
    if (!gameBoardRef?.current || animations.length === 0) {
      setCardPositions([]);
      return;
    }

    // Increment key to force re-animation
    animationKeyRef.current += 1;
    const currentKey = animationKeyRef.current;

    // Calculate positions for all cards being passed
    const calculatePositions = () => {
      const boardRect = gameBoardRef.current.getBoundingClientRect();
      const positions = [];

      animations.forEach((animation) => {
        const { fromPlayerId, toPlayerId, cards, isHuman, fromRects } = animation;

        // Map player IDs to position keys
        const playerPositions = {
          0: 'bottom',
          1: 'left',
          2: 'top',
          3: 'right'
        };

        const fromKey = playerPositions[fromPlayerId];
        const toKey = playerPositions[toPlayerId];

        const fromRef = playerRefs[fromKey];
        const toRef = playerRefs[toKey];

        if (!fromRef?.current || !toRef?.current) {
          console.warn(`Missing refs for animation: from=${fromKey}, to=${toKey}`);
          return;
        }

        const fromRect = fromRef.current.getBoundingClientRect();
        const toRect = toRef.current.getBoundingClientRect();

        // Calculate center positions relative to the game board
        const fromX = fromRect.left + fromRect.width / 2 - boardRect.left;
        const fromY = fromRect.top + fromRect.height / 2 - boardRect.top;
        const toX = toRect.left + toRect.width / 2 - boardRect.left;
        const toY = toRect.top + toRect.height / 2 - boardRect.top;

        // Create positions for each card, arranged as a tight fanned packet
        // (rotated + slightly offset) so the three cards travel together like a
        // handful being passed rather than three separate sliding cards.
        cards.forEach((card, cardIndex) => {
          const isVertical = fromKey === 'top' || fromKey === 'bottom';
          const spread = cardIndex - (cards.length - 1) / 2; // -1, 0, 1
          const cardOffset = spread * 14; // tighter packet
          const fanRot = spread * 9;       // fan angle within the packet

          // If we captured the real hand-slot position for this card (human's
          // own pass), launch it from there instead of the seat's center.
          const slot = fromRects && fromRects[cardIndex];
          const startX = slot
            ? slot.left + slot.width / 2 - boardRect.left
            : fromX + (isVertical ? cardOffset : 0);
          const startY = slot
            ? slot.top + slot.height / 2 - boardRect.top
            : fromY + (isVertical ? 0 : cardOffset);

          positions.push({
            card,
            fromPlayerId,
            toPlayerId,
            startX,
            startY,
            endX: toX + (isVertical ? cardOffset : 0),
            endY: toY + (isVertical ? 0 : cardOffset),
            fanRot,
            stagger: cardIndex * 0.05, // seconds — cards leave in quick succession
            animating: false,
            isHuman: isHuman,
            key: `${currentKey}-${fromPlayerId}-${toPlayerId}-${cardIndex}`
          });
        });
      });

      return positions;
    };

    // Initial positions
    const initialPositions = calculatePositions();
    if (initialPositions.length === 0) {
      return;
    }
    
    setCardPositions(initialPositions);

    // Trigger animation after a brief delay to ensure DOM is ready
    const startTimeout = setTimeout(() => {
      // Only animate if this is still the current animation
      if (animationKeyRef.current === currentKey) {
        setCardPositions(prev => 
          prev.map(pos => ({ ...pos, animating: true }))
        );
      }
    }, 50);

    return () => {
      clearTimeout(startTimeout);
    };
  }, [animations, playerRefs, gameBoardRef, animationDelay, animationDuration, fadeDelay]);

  if (cardPositions.length === 0) {
    return null;
  }

  return (
    <div 
      ref={overlayRef}
      className="passing-animation-container"
    >
      {cardPositions.map((pos) => {
        const ease = 'cubic-bezier(0.33, 0, 0.28, 1)';
        const delay = pos.stagger || 0;
        return (
          <div
            key={pos.key}
            className={`passing-card-animation ${pos.animating ? 'animating' : ''} ${!pos.isHuman ? 'ai-card' : ''}`}
            style={{
              left: pos.animating ? `${pos.endX}px` : `${pos.startX}px`,
              top: pos.animating ? `${pos.endY}px` : `${pos.startY}px`,
              transform: pos.animating
                ? `translate(-50%, -50%) rotate(${pos.fanRot + 6}deg) scale(0.92)`
                : `translate(-50%, -50%) rotate(${pos.fanRot}deg) scale(1)`,
              transition: pos.animating
                ? `left ${animationDuration}s ${ease} ${delay}s, top ${animationDuration}s ${ease} ${delay}s, transform ${animationDuration}s ${ease} ${delay}s, opacity ${animationDuration * 0.15}s ease ${fadeDelay + delay}s`
                : 'none',
              opacity: pos.animating ? 0 : 1
            }}
          >
            {pos.isHuman ? (
              <div className="passing-card-face">
                <Card card={pos.card} />
              </div>
            ) : (
              <div className="card-back-animation" />
            )}
          </div>
        );
      })}
    </div>
  );
};

export default PassingAnimation;

