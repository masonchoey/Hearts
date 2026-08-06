/**
 * Seat layout helpers for 3/4/5-player tables.
 *
 * The multiplayer store rotates players so the viewer is always at rotated
 * index 0 (bottom). These maps place the remaining players around the table,
 * ordered clockwise starting from the viewer's left.
 *
 * Slot names double as CSS position suffixes:
 *   GameBoard: `.player-<slot>`   TableCenter: `.trick-card.position-<slot>`
 */
const SLOTS_BY_COUNT = {
  3: ['bottom', 'left', 'right'],
  4: ['bottom', 'left', 'top', 'right'],
  5: ['bottom', 'left', 'top-left', 'top-right', 'right'],
}

// Left/right seats show a vertical fan of card backs; everything else horizontal.
const VERTICAL_SLOTS = new Set(['left', 'right'])

export function getSlots(playerCount) {
  return SLOTS_BY_COUNT[playerCount] || SLOTS_BY_COUNT[4]
}

/** Slot name for a rotated seat index (0 = viewer/bottom). */
export function slotForIndex(playerCount, rotatedIndex) {
  const slots = getSlots(playerCount)
  return slots[rotatedIndex] ?? 'bottom'
}

export function orientationForSlot(slot) {
  return VERTICAL_SLOTS.has(slot) ? 'vertical' : 'horizontal'
}
