"""
SQLAlchemy ORM models for users, friends, and multiplayer rooms
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from .database import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    """Application user profile synced from Neon Auth (``neon_auth.user``)."""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    neon_auth_id = Column(String, unique=True, nullable=False, index=True)
    auth_provider = Column(String, default="neon", nullable=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    picture = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sent_requests = relationship(
        "Friendship", foreign_keys="Friendship.requester_id", back_populates="requester"
    )
    received_requests = relationship(
        "Friendship", foreign_keys="Friendship.addressee_id", back_populates="addressee"
    )
    room_memberships = relationship("RoomPlayer", back_populates="user")


class Friendship(Base):
    __tablename__ = "friendships"

    id = Column(String, primary_key=True, default=_uuid)
    requester_id = Column(String, ForeignKey("users.id"), nullable=False)
    addressee_id = Column(String, ForeignKey("users.id"), nullable=False)
    # status: "pending", "accepted", "rejected"
    status = Column(String, default="pending", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    requester = relationship("User", foreign_keys=[requester_id], back_populates="sent_requests")
    addressee = relationship("User", foreign_keys=[addressee_id], back_populates="received_requests")

    __table_args__ = (
        UniqueConstraint("requester_id", "addressee_id", name="unique_friendship"),
    )


class MultiplayerRoom(Base):
    __tablename__ = "multiplayer_rooms"

    id = Column(String, primary_key=True, default=_uuid)
    invite_code = Column(String, unique=True, nullable=False, index=True)
    host_id = Column(String, ForeignKey("users.id"), nullable=False)
    # status: "waiting", "playing", "finished"
    status = Column(String, default="waiting", nullable=False)
    # Custom game config: number of seats (3/4/5) and JSON-encoded scoring rules.
    player_count = Column(Integer, default=4, nullable=False)
    rules_config = Column(String, nullable=True)  # JSON: {"jd_bonus": bool, "ten_club_doubler": bool}
    created_at = Column(DateTime, default=datetime.utcnow)

    players = relationship("RoomPlayer", back_populates="room", order_by="RoomPlayer.seat")


class RoomPlayer(Base):
    __tablename__ = "room_players"

    id = Column(String, primary_key=True, default=_uuid)
    room_id = Column(String, ForeignKey("multiplayer_rooms.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    seat = Column(Integer, nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)

    room = relationship("MultiplayerRoom", back_populates="players")
    user = relationship("User", back_populates="room_memberships")

    __table_args__ = (
        UniqueConstraint("room_id", "seat", name="unique_room_seat"),
        UniqueConstraint("room_id", "user_id", name="unique_room_user"),
    )
