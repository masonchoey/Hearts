"""
Multiplayer routes: room management + WebSocket real-time play
"""
import random
import string
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..db.database import get_db, AsyncSessionLocal
from ..db.models import User, MultiplayerRoom, RoomPlayer
from ..auth.dependencies import get_current_user, resolve_user_from_token
from ..realtime.connection_manager import manager as ws_manager
from ..game.multiplayer_game import multiplayer_manager
from ..game.native import RuleConfig
from ..schemas.types import Card

router = APIRouter(prefix="/mp", tags=["multiplayer"])


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_invite_code(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _room_rules(room: MultiplayerRoom) -> RuleConfig:
    """Build a RuleConfig from a room's stored player_count + rules_config JSON."""
    try:
        cfg = json.loads(room.rules_config) if room.rules_config else {}
    except (TypeError, ValueError):
        cfg = {}
    return RuleConfig(
        player_count=room.player_count or 4,
        jd_bonus=bool(cfg.get("jd_bonus", False)),
        ten_club_doubler=bool(cfg.get("ten_club_doubler", False)),
    )


# ── Pydantic models ─────────────────────────────────────────────────────────

class RoomPlayerOut(BaseModel):
    seat: int
    user_id: str
    name: str
    picture: Optional[str]


class RoomRulesOut(BaseModel):
    jd_bonus: bool = False
    ten_club_doubler: bool = False


class RoomOut(BaseModel):
    room_id: str
    invite_code: str
    host_id: str
    status: str
    target_score: Optional[int] = None
    player_count: int = 4
    rules: RoomRulesOut = RoomRulesOut()
    players: List[RoomPlayerOut]


class CreateRoomIn(BaseModel):
    # Cumulative score at which the match ends; None = infinite play.
    target_score: Optional[int] = 100
    # Custom game config.
    player_count: int = Field(4, ge=3, le=5)
    jd_bonus: bool = False
    ten_club_doubler: bool = False


def _rules_out(room: MultiplayerRoom) -> RoomRulesOut:
    r = _room_rules(room)
    return RoomRulesOut(jd_bonus=r.jd_bonus, ten_club_doubler=r.ten_club_doubler)


def _room_out(room: MultiplayerRoom, players_out: List[RoomPlayerOut]) -> RoomOut:
    return RoomOut(
        room_id=room.id,
        invite_code=room.invite_code,
        host_id=room.host_id,
        status=room.status,
        target_score=room.target_score,
        player_count=room.player_count or 4,
        rules=_rules_out(room),
        players=players_out,
    )


# ── REST endpoints ───────────────────────────────────────────────────────────

@router.post("/rooms", response_model=RoomOut)
async def create_room(
    payload: Optional[CreateRoomIn] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new multiplayer room and join seat 0 as host.

    The host chooses the player count (3/4/5), which optional scoring rules are
    active, and the target score (or infinite play).
    """
    cfg = payload or CreateRoomIn()
    if cfg.target_score is not None and cfg.target_score < 1:
        raise HTTPException(status_code=400, detail="target_score must be a positive number or null for infinite play")

    # Generate a unique invite code
    for _ in range(10):
        code = _make_invite_code()
        result = await db.execute(select(MultiplayerRoom).where(MultiplayerRoom.invite_code == code))
        if result.scalar_one_or_none() is None:
            break

    rules_json = json.dumps({"jd_bonus": cfg.jd_bonus, "ten_club_doubler": cfg.ten_club_doubler})
    room = MultiplayerRoom(
        invite_code=code,
        host_id=current_user.id,
        status="waiting",
        target_score=cfg.target_score,
        player_count=cfg.player_count,
        rules_config=rules_json,
    )
    db.add(room)
    await db.flush()

    seat = RoomPlayer(room_id=room.id, user_id=current_user.id, seat=0)
    db.add(seat)
    await db.commit()
    await db.refresh(room)

    return _room_out(room, [
        RoomPlayerOut(seat=0, user_id=current_user.id, name=current_user.name, picture=current_user.picture),
    ])


@router.post("/rooms/join/{invite_code}", response_model=RoomOut)
async def join_room(
    invite_code: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Join an existing room via invite code."""
    result = await db.execute(
        select(MultiplayerRoom)
        .options(selectinload(MultiplayerRoom.players).selectinload(RoomPlayer.user))
        .where(MultiplayerRoom.invite_code == invite_code.upper())
    )
    room = result.scalar_one_or_none()
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.status != "waiting":
        raise HTTPException(status_code=409, detail="Game already started")
    capacity = room.player_count or 4
    if len(room.players) >= capacity:
        raise HTTPException(status_code=409, detail="Room is full")

    # Check not already in room
    for p in room.players:
        if p.user_id == current_user.id:
            raise HTTPException(status_code=409, detail="Already in this room")

    # Assign next available seat
    taken_seats = {p.seat for p in room.players}
    seat_num = next(s for s in range(capacity) if s not in taken_seats)

    new_player = RoomPlayer(room_id=room.id, user_id=current_user.id, seat=seat_num)
    db.add(new_player)
    await db.commit()

    # Reload with fresh data
    result = await db.execute(
        select(MultiplayerRoom)
        .options(selectinload(MultiplayerRoom.players).selectinload(RoomPlayer.user))
        .where(MultiplayerRoom.id == room.id)
    )
    room = result.scalar_one()

    players_out = [
        RoomPlayerOut(seat=p.seat, user_id=p.user_id, name=p.user.name, picture=p.user.picture)
        for p in room.players
    ]

    # Notify existing WebSocket connections that someone joined
    await ws_manager.broadcast(room.id, {
        "type": "player_joined",
        "player": {"seat": seat_num, "name": current_user.name, "user_id": current_user.id},
        "player_count": len(room.players),
    })

    return _room_out(room, players_out)


@router.get("/rooms/{room_id}", response_model=RoomOut)
async def get_room(
    room_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(MultiplayerRoom)
        .options(selectinload(MultiplayerRoom.players).selectinload(RoomPlayer.user))
        .where(MultiplayerRoom.id == room_id)
    )
    room = result.scalar_one_or_none()
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")

    players_out = [
        RoomPlayerOut(seat=p.seat, user_id=p.user_id, name=p.user.name, picture=p.user.picture)
        for p in room.players
    ]
    return _room_out(room, players_out)


@router.post("/rooms/{room_id}/start", response_model=RoomOut)
async def start_room(
    room_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Host-only: start the game and broadcast game_started to all WebSocket clients."""
    result = await db.execute(
        select(MultiplayerRoom)
        .options(selectinload(MultiplayerRoom.players).selectinload(RoomPlayer.user))
        .where(MultiplayerRoom.id == room_id)
    )
    room = result.scalar_one_or_none()
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.host_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the host can start the game")
    if room.status != "waiting":
        raise HTTPException(status_code=409, detail="Game already started")
    required = room.player_count or 4
    if len(room.players) != required:
        raise HTTPException(status_code=400, detail=f"Need exactly {required} players to start")

    await _start_game(room, db)

    result = await db.execute(
        select(MultiplayerRoom)
        .options(selectinload(MultiplayerRoom.players).selectinload(RoomPlayer.user))
        .where(MultiplayerRoom.id == room_id)
    )
    room = result.scalar_one()
    players_out = [
        RoomPlayerOut(seat=p.seat, user_id=p.user_id, name=p.user.name, picture=p.user.picture)
        for p in room.players
    ]
    return _room_out(room, players_out)


# ── Internal: connection status ──────────────────────────────────────────────

async def _broadcast_connection_status(room_id: str):
    """Notify all clients which seats currently have an active WebSocket."""
    await ws_manager.broadcast(room_id, {
        "type": "connection_status",
        "connected_seats": ws_manager.connected_seats(room_id),
    })


# ── Internal: start game ─────────────────────────────────────────────────────

async def _start_game(room: MultiplayerRoom, db: AsyncSession):
    """Initialise the in-memory game and notify all WebSocket clients."""
    seat_to_user = {p.seat: p.user_id for p in room.players}
    seat_to_name = {p.seat: p.user.name for p in room.players}
    rules = _room_rules(room)

    game = multiplayer_manager.create_game(
        room.id, seat_to_user, seat_to_name, rules=rules, target_score=room.target_score,
    )

    room.status = "playing"
    await db.commit()

    # Send each player their personalised initial state
    for seat in range(rules.player_count):
        await ws_manager.send_to_seat(room.id, seat, {
            "type": "game_started",
            "state": _state_for(room.id, game, seat),
        })


def _connection_overlay(room_id: str, game) -> dict:
    """Connection-layer fields merged into every game_state (pause UI).

    A seat is 'disconnected' if it has no active socket. The match is 'paused'
    when it's such a seat's turn — no one can move until they return.
    """
    connected = ws_manager.connected_seats(room_id)
    disconnected = [s for s in range(game.n) if s not in connected]
    return {
        "connected_seats": connected,
        "disconnected_seats": disconnected,
        "paused": game.current_player() in disconnected,
    }


def _state_for(room_id: str, game, seat: int) -> dict:
    """Per-seat game state with the connection overlay merged in."""
    state = game.build_state_for_seat(seat)
    state.update(_connection_overlay(room_id, game))
    return state


async def _broadcast_game_state(room_id: str, game):
    """Send personalised game state to every connected seat."""
    for target_seat in range(game.n):
        await ws_manager.send_to_seat(room_id, target_seat, {
            "type": "game_state",
            "state": _state_for(room_id, game, target_seat),
        })


async def _advance_and_broadcast(room_id: str, game):
    """Finalise the deal if it just ended, then broadcast to everyone."""
    game.finalize_round_if_over()
    await _broadcast_game_state(room_id, game)


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/{room_id}")
async def websocket_endpoint(
    room_id: str,
    websocket: WebSocket,
    token: str,
):
    """
    WebSocket for real-time multiplayer game.
    Auth: pass ?token=<jwt> as a query param.
    Messages in:  { type: "play_card", card: { suit, rank } }
                  { type: "pass_cards", cards: [{ suit, rank }, ...] }  (exactly 3)
                  { type: "next_round" } | { type: "end_match" } | { type: "ping" }
    Messages out: { type: "game_state", state: {...} }
                  { type: "error", message: "..." }
                  { type: "player_joined", ... }
                  { type: "game_started", state: {...} }
    """
    # Authenticate via token query param. Accepts either an app JWT (direct
    # Google flow) or a Neon Auth JWT — resolved to the same app user.

    async with AsyncSessionLocal() as auth_db:
        user = await resolve_user_from_token(token, auth_db)
    if user is None:
        await websocket.close(code=4001)
        return
    user_id = user.id

    # Find the player's seat in this room (in-memory game)
    game = multiplayer_manager.get_game(room_id)

    # If game doesn't exist yet, user is still in the waiting lobby — connect anyway
    # so they receive game_started broadcast
    seat = None
    if game:
        seat = game.user_to_seat.get(user_id)
        if seat is None:
            await websocket.close(code=4003)
            return
    else:
        # Game not started yet; look up seat from DB to hold the connection
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(RoomPlayer).where(RoomPlayer.room_id == room_id, RoomPlayer.user_id == user_id)
            )
            rp = result.scalar_one_or_none()
            if rp is None:
                await websocket.close(code=4003)
                return
            seat = rp.seat

    await ws_manager.connect(room_id, seat, websocket)

    # Game may have auto-started before this client connected — and this may be a
    # reconnect that clears a pause.
    game = multiplayer_manager.get_game(room_id)
    if game is not None:
        await _broadcast_connection_status(room_id)
        await ws_manager.send_to_seat(room_id, seat, {
            "type": "game_started",
            "state": _state_for(room_id, game, seat),
        })
        # Refresh everyone — this reconnect may have cleared a pause.
        await _broadcast_game_state(room_id, game)
    else:
        await _broadcast_connection_status(room_id)

    async def _send_error(message: str):
        await ws_manager.send_to_seat(room_id, seat, {"type": "error", "message": message})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error("Invalid JSON")
                continue

            msg_type = msg.get("type")
            game = multiplayer_manager.get_game(room_id)

            if msg_type == "ping":
                await ws_manager.send_to_seat(room_id, seat, {"type": "pong"})
                continue

            if game is None:
                await _send_error("Game not started")
                continue

            if msg_type == "play_card":
                try:
                    card_data = msg.get("card", {})
                    card = Card(suit=card_data["suit"], rank=card_data["rank"])
                    game.apply_move(seat, card)
                except (ValueError, KeyError) as e:
                    await _send_error(str(e))
                    continue
                # Trick travels via current_trick; no move_sequence (no pass leak).
                await _advance_and_broadcast(room_id, game)

            elif msg_type == "pass_cards":
                try:
                    cards_data = msg.get("cards", [])
                    cards = [Card(suit=c["suit"], rank=c["rank"]) for c in cards_data]
                    game.queue_pass(seat, cards)
                    game.process_pending_passes()
                except (ValueError, KeyError) as e:
                    await _send_error(str(e))
                    continue
                await _advance_and_broadcast(room_id, game)

            elif msg_type == "next_round":
                # Cooperative: any connected player may advance to the next deal.
                if game.start_next_round():
                    await _advance_and_broadcast(room_id, game)
                else:
                    await _broadcast_game_state(room_id, game)

            elif msg_type == "end_match":
                game.end_match()
                await _broadcast_game_state(room_id, game)

    except WebSocketDisconnect:
        pass
    except Exception:
        # Abnormal close (e.g. RuntimeError from receive after the socket died).
        pass
    finally:
        # Only clear the seat if THIS socket is still the active one (P1-4).
        removed = ws_manager.disconnect(room_id, seat, websocket)
        if removed:
            game = multiplayer_manager.get_game(room_id)
            await ws_manager.broadcast(room_id, {"type": "player_disconnected", "seat": seat})
            await _broadcast_connection_status(room_id)
            if game is not None:
                await _broadcast_game_state(room_id, game)  # surface the pause to others
