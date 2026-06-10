"""
WebSocket connection manager for multiplayer game rooms.
Tracks one WebSocket per player seat per room and broadcasts game state updates.
"""
import json
import asyncio
from typing import Dict, Optional
from fastapi import WebSocket


class RoomConnection:
    """Holds the 4 WebSocket connections for a single room."""

    def __init__(self):
        # seat (0-3) → WebSocket
        self.connections: Dict[int, WebSocket] = {}

    def is_full(self) -> bool:
        return len(self.connections) >= 4

    def connected_seats(self) -> list[int]:
        return list(self.connections.keys())


class ConnectionManager:
    def __init__(self):
        # room_id → RoomConnection
        self.rooms: Dict[str, RoomConnection] = {}

    def get_or_create_room(self, room_id: str) -> RoomConnection:
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomConnection()
        return self.rooms[room_id]

    async def connect(self, room_id: str, seat: int, websocket: WebSocket):
        await websocket.accept()
        room = self.get_or_create_room(room_id)
        room.connections[seat] = websocket

    def disconnect(self, room_id: str, seat: int):
        if room_id in self.rooms:
            self.rooms[room_id].connections.pop(seat, None)
            if not self.rooms[room_id].connections:
                del self.rooms[room_id]

    async def broadcast(self, room_id: str, message: dict):
        """Send a message to every connected player in the room."""
        if room_id not in self.rooms:
            return
        dead = []
        for seat, ws in self.rooms[room_id].connections.items():
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(seat)
        for seat in dead:
            self.rooms[room_id].connections.pop(seat, None)

    async def send_to_seat(self, room_id: str, seat: int, message: dict):
        """Send a message to a specific seat in a room."""
        room = self.rooms.get(room_id)
        if room and seat in room.connections:
            try:
                await room.connections[seat].send_text(json.dumps(message))
            except Exception:
                room.connections.pop(seat, None)

    def connected_seats(self, room_id: str) -> list[int]:
        room = self.rooms.get(room_id)
        return room.connected_seats() if room else []


# Singleton used across the app
manager = ConnectionManager()
