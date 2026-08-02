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


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_invite_code(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


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
    player_count: int = 4
    rules: RoomRulesOut = RoomRulesOut()
    players: List[RoomPlayerOut]


class CreateRoomRequest(BaseModel):
    player_count: int = Field(4, ge=3, le=5)
    jd_bonus: bool = False
    ten_club_doubler: bool = False


def _rules_out(room: MultiplayerRoom) -> RoomRulesOut:
    r = _room_rules(room)
    return RoomRulesOut(jd_bonus=r.jd_bonus, ten_club_doubler=r.ten_club_doubler)


# ── REST endpoints ───────────────────────────────────────────────────────────

@router.post("/rooms", response_model=RoomOut)
async def create_room(
    config: CreateRoomRequest = CreateRoomRequest(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new multiplayer room and join seat 0 as host.

    The host chooses the player count (3/4/5) and which optional scoring rules
    are active; these are stored on the room and applied when the game starts.
    """
    # Generate a unique invite code
    for _ in range(10):
        code = _make_invite_code()
        result = await db.execute(select(MultiplayerRoom).where(MultiplayerRoom.invite_code == code))
        if result.scalar_one_or_none() is None:
            break

    rules_json = json.dumps({
        "jd_bonus": config.jd_bonus,
        "ten_club_doubler": config.ten_club_doubler,
    })
    room = MultiplayerRoom(
        invite_code=code,
        host_id=current_user.id,
        status="waiting",
        player_count=config.player_count,
        rules_config=rules_json,
    )
    db.add(room)
    await db.flush()

    seat = RoomPlayer(room_id=room.id, user_id=current_user.id, seat=0)
    db.add(seat)
    await db.commit()
    await db.refresh(room)

    return RoomOut(
        room_id=room.id,
        invite_code=room.invite_code,
        host_id=room.host_id,
        status=room.status,
        player_count=room.player_count,
        rules=_rules_out(room),
        players=[RoomPlayerOut(seat=0, user_id=current_user.id, name=current_user.name, picture=current_user.picture)],
    )


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

    return RoomOut(
        room_id=room.id,
        invite_code=room.invite_code,
        host_id=room.host_id,
        status=room.status,
        player_count=room.player_count,
        rules=_rules_out(room),
        players=players_out,
    )


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
    return RoomOut(
        room_id=room.id,
        invite_code=room.invite_code,
        host_id=room.host_id,
        status=room.status,
        player_count=room.player_count,
        rules=_rules_out(room),
        players=players_out,
    )


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
    return RoomOut(
        room_id=room.id,
        invite_code=room.invite_code,
        host_id=room.host_id,
        status=room.status,
        player_count=room.player_count,
        rules=_rules_out(room),
        players=players_out,
    )


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

    game = multiplayer_manager.create_game(room.id, seat_to_user, seat_to_name, rules)

    room.status = "playing"
    await db.commit()

    # Send each player their personalised initial state
    for seat in range(rules.player_count):
        state = game.build_state_for_seat(seat)
        await ws_manager.send_to_seat(room.id, seat, {
            "type": "game_started",
            "state": state,
        })


async def _broadcast_game_state(room_id: str, game, move_sequence: Optional[list] = None):
    """Send personalised game state to every connected seat."""
    moves = move_sequence or []
    for target_seat in range(game.n):
        state = game.build_state_with_move(target_seat, moves)
        await ws_manager.send_to_seat(room_id, target_seat, {
            "type": "game_state",
            "state": state,
        })


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
    await _broadcast_connection_status(room_id)

    # Game may have auto-started before this client connected.
    game = multiplayer_manager.get_game(room_id)
    if game is not None:
        state = game.build_state_for_seat(seat)
        await ws_manager.send_to_seat(room_id, seat, {
            "type": "game_started",
            "state": state,
        })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws_manager.send_to_seat(room_id, seat, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "play_card":
                game = multiplayer_manager.get_game(room_id)
                if game is None:
                    await ws_manager.send_to_seat(room_id, seat, {"type": "error", "message": "Game not started"})
                    continue

                try:
                    card_data = msg.get("card", {})
                    card = Card(suit=card_data["suit"], rank=card_data["rank"])
                    game.apply_move(seat, card)
                    moves = [(seat, card)] + game.process_pending_passes()
                    move_sequence = [[s, c.dict()] for s, c in moves]
                except (ValueError, KeyError) as e:
                    await ws_manager.send_to_seat(room_id, seat, {"type": "error", "message": str(e)})
                    continue

                await _broadcast_game_state(room_id, game, move_sequence)

            elif msg_type == "pass_cards":
                game = multiplayer_manager.get_game(room_id)
                if game is None:
                    await ws_manager.send_to_seat(room_id, seat, {"type": "error", "message": "Game not started"})
                    continue

                try:
                    cards_data = msg.get("cards", [])
                    cards = [Card(suit=c["suit"], rank=c["rank"]) for c in cards_data]
                    game.queue_pass(seat, cards)
                    moves = game.process_pending_passes()
                    move_sequence = [[s, c.dict()] for s, c in moves]
                except (ValueError, KeyError) as e:
                    await ws_manager.send_to_seat(room_id, seat, {"type": "error", "message": str(e)})
                    continue

                await _broadcast_game_state(room_id, game, move_sequence)

            elif msg_type == "ping":
                await ws_manager.send_to_seat(room_id, seat, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(room_id, seat)
        await ws_manager.broadcast(room_id, {
            "type": "player_disconnected",
            "seat": seat,
        })
        await _broadcast_connection_status(room_id)
