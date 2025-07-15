
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from action_items_service import ActionItemsService as action_items_service
from utils.logging import items_logger

router = APIRouter()

@router.get("/api/action_items")
async def get_action_items(user_id: str, limit: int = 100):
    """Retrieves action items from the action items service."""
    print(f"Getting action items for user {user_id}")
    try:
        action_items = action_items_service.get_action_items(user_id=user_id, limit=limit)
        return JSONResponse(content=action_items)
    except Exception as e:
        # items_logger.error(f"Error fetching action items: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching action items"}
        )
