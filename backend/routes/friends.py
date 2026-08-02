"""
Friends system: send requests, accept, list friends, search users
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from typing import List, Optional

from ..db.database import get_db
from ..db.models import User, Friendship
from ..auth.dependencies import get_current_user

router = APIRouter(prefix="/friends", tags=["friends"])


class FriendUserOut(BaseModel):
    id: str
    name: str
    email: str
    picture: Optional[str]


class FriendRequestOut(BaseModel):
    id: str
    from_user: FriendUserOut
    status: str


# ── Search users ────────────────────────────────────────────────────────────

@router.get("/search", response_model=List[FriendUserOut])
async def search_users(
    q: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Search users by name or email (partial match, excluding self)."""
    if len(q) < 2:
        return []
    result = await db.execute(
        select(User).where(
            and_(
                User.id != current_user.id,
                or_(
                    User.name.ilike(f"%{q}%"),
                    User.email.ilike(f"%{q}%"),
                ),
            )
        ).limit(10)
    )
    users = result.scalars().all()
    return [FriendUserOut(id=u.id, name=u.name, email=u.email, picture=u.picture) for u in users]


# ── Send friend request ─────────────────────────────────────────────────────

@router.post("/request/{target_user_id}")
async def send_friend_request(
    target_user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if target_user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot add yourself")

    # Check target exists
    result = await db.execute(select(User).where(User.id == target_user_id))
    target = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Check existing relationship
    result = await db.execute(
        select(Friendship).where(
            or_(
                and_(Friendship.requester_id == current_user.id, Friendship.addressee_id == target_user_id),
                and_(Friendship.requester_id == target_user_id, Friendship.addressee_id == current_user.id),
            )
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Friend request already exists")

    friendship = Friendship(requester_id=current_user.id, addressee_id=target_user_id, status="pending")
    db.add(friendship)
    await db.commit()
    return {"message": "Friend request sent"}


# ── Accept / reject request ─────────────────────────────────────────────────

@router.post("/accept/{request_id}")
async def accept_request(
    request_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Friendship).where(Friendship.id == request_id))
    req = result.scalar_one_or_none()
    if req is None or req.addressee_id != current_user.id:
        raise HTTPException(status_code=404, detail="Request not found")
    req.status = "accepted"
    await db.commit()
    return {"message": "Friend request accepted"}


@router.post("/reject/{request_id}")
async def reject_request(
    request_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Friendship).where(Friendship.id == request_id))
    req = result.scalar_one_or_none()
    if req is None or req.addressee_id != current_user.id:
        raise HTTPException(status_code=404, detail="Request not found")
    req.status = "rejected"
    await db.commit()
    return {"message": "Friend request rejected"}


# ── List friends & pending requests ────────────────────────────────────────

@router.get("/", response_model=List[FriendUserOut])
async def list_friends(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Friendship).where(
            and_(
                Friendship.status == "accepted",
                or_(
                    Friendship.requester_id == current_user.id,
                    Friendship.addressee_id == current_user.id,
                ),
            )
        )
    )
    friendships = result.scalars().all()

    friend_ids = [
        f.addressee_id if f.requester_id == current_user.id else f.requester_id
        for f in friendships
    ]

    if not friend_ids:
        return []

    result = await db.execute(select(User).where(User.id.in_(friend_ids)))
    friends = result.scalars().all()
    return [FriendUserOut(id=u.id, name=u.name, email=u.email, picture=u.picture) for u in friends]


@router.get("/requests", response_model=List[FriendRequestOut])
async def list_pending_requests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Incoming pending friend requests."""
    result = await db.execute(
        select(Friendship).where(
            and_(
                Friendship.addressee_id == current_user.id,
                Friendship.status == "pending",
            )
        )
    )
    requests = result.scalars().all()

    out = []
    for req in requests:
        r2 = await db.execute(select(User).where(User.id == req.requester_id))
        requester = r2.scalar_one_or_none()
        if requester:
            out.append(
                FriendRequestOut(
                    id=req.id,
                    from_user=FriendUserOut(
                        id=requester.id, name=requester.name,
                        email=requester.email, picture=requester.picture,
                    ),
                    status=req.status,
                )
            )
    return out
