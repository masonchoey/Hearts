"""
WebSocket connection manager for multiplayer game rooms.
Tracks one active WebSocket per player seat per room and broadcasts updates.

Connection identity: each seat maps to exactly one *active* socket. If a seat
reconnects (or the lobby socket hands off to the game socket), the newer socket
replaces the old one, and a disconnect only clears the seat if the socket that
disconnected is still the active one. This prevents a stale socket's close from
evicting a freshly-connected player (P1-4).
"""
import json
from typing import Dict
from fastapi import WebSocket


class RoomConnection:
    """Holds the active WebSocket per seat for a single room."""

    def __init__(self):
        self.connections: Dict[int, WebSocket] = {}  # seat (0-3) → active WebSocket

    def is_full(self) -> bool:
        return len(self.connections) >= 4

    def connected_seats(self) -> list[int]:
        return sorted(self.connections.keys())


class ConnectionManager:
    def __init__(self):
        self.rooms: Dict[str, RoomConnection] = {}

    def get_or_create_room(self, room_id: str) -> RoomConnection:
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomConnection()
        return self.rooms[room_id]

    async def connect(self, room_id: str, seat: int, websocket: WebSocket) -> None:
        """Accept a socket and make it the active one for this seat.

        Any previous socket for the seat is closed first so it can't linger or
        later evict this one.
        """
        await websocket.accept()
        room = self.get_or_create_room(room_id)
        old = room.connections.get(seat)
        if old is not None and old is not websocket:
            try:
                await old.close(code=4000)  # replaced by a newer connection
            except Exception:
                pass
        room.connections[seat] = websocket

    def disconnect(self, room_id: str, seat: int, websocket: WebSocket | None = None) -> bool:
        """Remove the seat's connection — but only if `websocket` is still the
        active socket for it. Returns True if the active seat was actually removed.

        Passing websocket=None forces removal (legacy behaviour).
        """
        room = self.rooms.get(room_id)
        if room is None:
            return False
        active = room.connections.get(seat)
        if active is None:
            return False
        if websocket is not None and active is not websocket:
            # A stale socket closed; the seat has since been taken by a newer
            # socket. Leave the active one in place.
            return False
        room.connections.pop(seat, None)
        if not room.connections:
            del self.rooms[room_id]
        return True

    async def broadcast(self, room_id: str, message: dict) -> None:
        """Send a message to every connected player in the room."""
        room = self.rooms.get(room_id)
        if room is None:
            return
        dead = []
        for seat, ws in list(room.connections.items()):
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append((seat, ws))
        for seat, ws in dead:
            # Only drop if still the active socket for that seat.
            if room.connections.get(seat) is ws:
                room.connections.pop(seat, None)

    async def send_to_seat(self, room_id: str, seat: int, message: dict) -> None:
        """Send a message to a specific seat in a room."""
        room = self.rooms.get(room_id)
        if room and seat in room.connections:
            ws = room.connections[seat]
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                if room.connections.get(seat) is ws:
                    room.connections.pop(seat, None)

    def connected_seats(self, room_id: str) -> list[int]:
        room = self.rooms.get(room_id)
        return room.connected_seats() if room else []


# Singleton used across the app
manager = ConnectionManager()
