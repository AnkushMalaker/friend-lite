"""
Client management routes for Friend-Lite API.

Handles active client monitoring and management.
"""

import logging

from fastapi import APIRouter, Depends

from advanced_omi_backend.auth import current_active_user
from advanced_omi_backend.client_manager import (
    ClientManager,
    get_client_manager_dependency,
)
from advanced_omi_backend.controllers import client_controller
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clients", tags=["clients"])


@router.get("/active")
async def get_active_clients(
    current_user: User = Depends(current_active_user),
    client_manager: ClientManager = Depends(get_client_manager_dependency),
):
    """Get information about active clients. Users see only their own clients, admins see all."""
    return await client_controller.get_active_clients(current_user, client_manager)
