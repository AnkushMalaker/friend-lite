"""
WebSocket authentication dependencies for FastAPI.
Provides dependency injection for WebSocket endpoints with proper authentication.
"""

import re
from fastapi import Query, status, WebSocket
from fastapi.exceptions import WebSocketException
from auth import get_jwt_strategy, get_user_db, UserManager
from users import User


async def current_user_ws(
    websocket: WebSocket,                       # ← FastAPI injects the live WS
    token: str | None = Query(None),            # ?token=<JWT>
) -> User:
    """
    Dependency for WebSocket endpoints.
    Priority: 1) ?token=<JWT>  2) cookie 'fastapiusersauth'.
    Raises WebSocketException(1008) if authentication fails.
    """
    strategy = get_jwt_strategy()

    # ── 1) query-string JWT ───────────────────────────────────────────
    if token:
        try:
            user_db = await get_user_db().__anext__()
            user = await strategy.read_token(token, UserManager(user_db))
            if user and user.is_active:
                return user
        except Exception:
            pass  # fall through

    # ── 2) cookie JWT ─────────────────────────────────────────────────
    cookie_header = websocket.headers.get("cookie")
    if cookie_header:
        m = re.search(r"fastapiusersauth=([^;]+)", cookie_header)
        if m:
            try:
                jwt = m.group(1)
                user_db = await get_user_db().__anext__()
                user = await strategy.read_token(jwt, UserManager(user_db))
                if user and user.is_active:
                    return user
            except Exception:
                pass

    # ── auth failed ──────────────────────────────────────────────────
    raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthenticated") 