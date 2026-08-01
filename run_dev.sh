#!/bin/bash
# Run the Hearts backend and frontend together for local development.
# Starts both, streams their logs to this terminal, and shuts both down
# cleanly on Ctrl+C.
#
#   ./run_dev.sh
#
# Backend:  http://localhost:8000
# Frontend: http://localhost:3000/OpenSpiel-Hearts/   (note the base path)
#
# Once both are up, four browser tabs open automatically — one per dev user
# (alice / bob / carol / dave) — so you can play a full 4-player game. Pass
# --no-open (or set OPEN_BROWSER=0) to skip auto-opening.

cd "$(dirname "$0")"
ROOT="$PWD"

# Prefer the project virtualenv's Python if it exists.
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

# ── Config ───────────────────────────────────────────────────────────────────
FRONTEND_BASE="http://localhost:3000/OpenSpiel-Hearts/"   # note the vite base path
API_BASE="http://localhost:8000"
BACKEND_HEALTH="$API_BASE/health"
DEV_USERS=(alice bob carol dave)                          # one browser tab each; [0] hosts

OPEN_BROWSER="${OPEN_BROWSER:-1}"
AUTO_JOIN="${AUTO_JOIN:-1}"   # pre-create a room (host = first dev user) and seat everyone
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_BROWSER=0 ;;
    --no-join) AUTO_JOIN=0 ;;
  esac
done

# Open a URL in the default browser (macOS / Linux / Git-Bash fallbacks).
open_url() {
  if command -v open >/dev/null 2>&1; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1
  elif command -v start >/dev/null 2>&1; then start "" "$1"
  else echo "  (no browser opener found — open manually: $1)"; fi
}

BACKEND_PID=""
FRONTEND_PID=""
OPENER_PID=""

cleanup() {
  echo ""
  echo "🛑 Shutting down..."
  # Kill each server and any child processes it spawned (e.g. npm -> vite).
  for pid in "$OPENER_PID" "$FRONTEND_PID" "$BACKEND_PID"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      pkill -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "🎮 Starting Hearts (dev)"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000/OpenSpiel-Hearts/  (add ?dev=alice to sign in)"
echo "============================================================"

# ── Backend ──────────────────────────────────────────────────────────────────
# Run in-process (no pipe) so $! is the real python PID and signals reach it.
export PYTHONPATH="$ROOT/backend:${PYTHONPATH:-}"
"$PYTHON" -m backend.main &
BACKEND_PID=$!

# ── Frontend ─────────────────────────────────────────────────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "[frontend] Installing dependencies (first run)..."
  (cd "$ROOT/frontend" && npm install)
fi
( cd "$ROOT/frontend" && npm run dev ) &
FRONTEND_PID=$!

echo "✓ Both processes started (PIDs: backend=$BACKEND_PID frontend=$FRONTEND_PID)."
echo "  Press Ctrl+C to stop both."

# ── Auto-open dev tabs ───────────────────────────────────────────────────────
# Wait (in the background, so Ctrl+C still works) until both servers respond,
# then open one tab per dev user.
if [ "$OPEN_BROWSER" = "1" ]; then
  (
    # Wait for the backend to respond.
    for _ in $(seq 1 60); do
      curl -sf -o /dev/null "$BACKEND_HEALTH" 2>/dev/null && break
      sleep 1
    done

    # Pre-seed a room so all four tabs land in the same game with the first dev
    # user as host. Uses the dev-auth REST API (dev:<name> bearer tokens).
    room_qs=""
    if [ "$AUTO_JOIN" = "1" ]; then
      host="${DEV_USERS[0]}"
      resp=$(curl -sf -X POST "$API_BASE/mp/rooms" \
        -H "Authorization: Bearer dev:${host}" 2>/dev/null)
      room_id=$(printf '%s' "$resp" | "$PYTHON" -c \
        'import sys,json; print(json.load(sys.stdin).get("room_id",""))' 2>/dev/null)
      code=$(printf '%s' "$resp" | "$PYTHON" -c \
        'import sys,json; print(json.load(sys.stdin).get("invite_code",""))' 2>/dev/null)
      if [ -n "$room_id" ]; then
        for u in "${DEV_USERS[@]:1}"; do
          curl -sf -o /dev/null -X POST "$API_BASE/mp/rooms/join/${code}" \
            -H "Authorization: Bearer dev:${u}" 2>/dev/null || true
        done
        room_qs="&room=${room_id}"
        echo "🃏 Room ${code} ready (host: ${host}) — all four tabs will auto-join."
      else
        echo "⚠️  Could not pre-create a room; opening tabs without auto-join."
      fi
    fi

    # Wait for the frontend to respond.
    for _ in $(seq 1 30); do
      curl -sf -o /dev/null "$FRONTEND_BASE" 2>/dev/null && break
      sleep 1
    done

    echo "🌐 Opening ${#DEV_USERS[@]} dev tabs (${DEV_USERS[*]})..."
    for u in "${DEV_USERS[@]}"; do
      open_url "${FRONTEND_BASE}?dev=${u}${room_qs}"
      sleep 0.7   # stagger so the browser opens them in order
    done
    [ -n "$room_qs" ] && echo "   → In the ${DEV_USERS[0]} tab, click “Start Game” once all four show connected."
  ) &
  OPENER_PID=$!
else
  echo "  (auto-open disabled — visit ${FRONTEND_BASE}?dev=alice )"
fi

# Exit as soon as either process dies, so a crash doesn't leave a half-running
# stack. (Portable poll loop — macOS ships bash 3.2 without `wait -n`.)
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
  sleep 1
done

echo "⚠️  One process exited — stopping the other."
