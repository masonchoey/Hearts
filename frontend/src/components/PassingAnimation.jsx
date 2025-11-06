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
  const fadeDelay = animationDuration * 0.9; // Fade starts at 90% through animation (near the end)

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
        const { fromPlayerId, toPlayerId, cards, isHuman } = animation;
        
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

        // Create positions for each card (spread them slightly)
        cards.forEach((card, cardIndex) => {
          // Spread cards based on direction
          const isVertical = fromKey === 'top' || fromKey === 'bottom';
          const cardOffset = (cardIndex - (cards.length - 1) / 2) * 25;
          
          positions.push({
            card,
            fromPlayerId,
            toPlayerId,
            startX: fromX + (isVertical ? cardOffset : 0),
            startY: fromY + (isVertical ? 0 : cardOffset),
            endX: toX + (isVertical ? cardOffset : 0),
            endY: toY + (isVertical ? 0 : cardOffset),
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
      {cardPositions.map((pos) => (
        <div
          key={pos.key}
          className={`passing-card-animation ${pos.animating ? 'animating' : ''} ${!pos.isHuman ? 'ai-card' : ''}`}
          style={{
            left: pos.animating ? `${pos.endX}px` : `${pos.startX}px`,
            top: pos.animating ? `${pos.endY}px` : `${pos.startY}px`,
            transform: pos.animating ? 'translate(-50%, -50%) scale(0.6)' : 'translate(-50%, -50%) scale(1)',
            transition: pos.animating 
              ? `left ${animationDuration}s cubic-bezier(0.4, 0, 0.2, 1), top ${animationDuration}s cubic-bezier(0.4, 0, 0.2, 1), transform ${animationDuration}s cubic-bezier(0.4, 0, 0.2, 1), opacity ${animationDuration * 0.2}s ease ${fadeDelay}s` 
              : 'none',
            opacity: pos.animating ? 0 : 1
          }}
        >
          {pos.isHuman ? (
            <Card card={pos.card} />
          ) : (
            <div className="card-back-animation" />
          )}
        </div>
      ))}
    </div>
  );
};

export default PassingAnimation;

