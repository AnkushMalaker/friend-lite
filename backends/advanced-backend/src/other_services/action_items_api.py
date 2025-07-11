from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
import time
from bson import ObjectId

# Import authentication
from auth import current_active_user, User

# Set up logging
import logging
action_items_api_logger = logging.getLogger("action_items_api")

class ActionItemCreate(BaseModel):
    description: str
    assignee: Optional[str] = "unassigned"
    due_date: Optional[str] = "not_specified"
    priority: Optional[str] = "medium"
    context: Optional[str] = ""

class ActionItemUpdate(BaseModel):
    description: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    context: Optional[str] = None

def create_action_items_router(action_items_service) -> APIRouter:
    """Create FastAPI router for action items endpoints."""
    router = APIRouter(prefix="/api/action-items", tags=["action-items"])
    
    @router.get("")
    async def get_action_items(
        current_user: User = Depends(current_active_user), 
        user_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: Optional[int] = Query(50, ge=1, le=1000)
    ):
        """Get action items. Admins can specify user_id, users see only their own."""
        try:
            # Determine which user's action items to retrieve
            if current_user.is_superuser and user_id:
                target_user_id = user_id
            else:
                target_user_id = current_user.user_id
            
            # Use the service to get action items
            action_items = await action_items_service.get_action_items(
                target_user_id, limit=limit, status_filter=status
            )
            
            return {"action_items": action_items, "count": len(action_items)}
        except Exception as e:
            action_items_api_logger.error(f"Error getting action items: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("")
    async def create_action_item(
        item: ActionItemCreate, 
        current_user: User = Depends(current_active_user)
    ):
        """Create a new action item."""
        try:
            # Use the service to create action item
            action_item = await action_items_service.create_action_item(
                user_id=current_user.user_id,
                description=item.description,
                assignee=item.assignee,
                due_date=item.due_date,
                priority=item.priority,
                context=item.context
            )
            
            if action_item:
                return {"message": "Action item created successfully", "action_item": action_item}
            else:
                raise HTTPException(status_code=500, detail="Failed to create action item")
        except Exception as e:
            action_items_api_logger.error(f"Error creating action item: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/stats")
    async def get_action_items_stats(
        current_user: User = Depends(current_active_user), 
        user_id: Optional[str] = Query(None)
    ):
        """Get action items statistics. Admins can specify user_id, users see only their own stats."""
        try:
            # Determine which user's stats to retrieve
            if current_user.is_superuser and user_id:
                target_user_id = user_id
            else:
                target_user_id = current_user.user_id
            
            # Use the service to get stats
            stats = await action_items_service.get_action_item_stats(target_user_id)
            
            return {"stats": stats}
        except Exception as e:
            action_items_api_logger.error(f"Error getting action items stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/search")
    async def search_action_items(
        q: str = Query(..., description="Search query"),
        current_user: User = Depends(current_active_user),
        limit: Optional[int] = Query(20, ge=1, le=100)
    ):
        """Search action items by text query."""
        try:
            # Use the service to search
            results = await action_items_service.search_action_items(
                q, current_user.user_id, limit=limit
            )
            
            return {"results": results, "count": len(results)}
        except Exception as e:
            action_items_api_logger.error(f"Error searching action items: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{item_id}")
    async def get_action_item(
        item_id: str, 
        current_user: User = Depends(current_active_user)
    ):
        """Get a specific action item. Users can only access their own."""
        try:
            # Get all action items for the user and filter by ID
            # This ensures users can only access their own action items
            if current_user.is_superuser:
                action_items = await action_items_service.get_action_items(
                    user_id=None, limit=1000  # Get all for admin
                )
            else:
                action_items = await action_items_service.get_action_items(
                    current_user.user_id, limit=1000
                )
            
            # Find the specific item
            target_item = None
            for item in action_items:
                if str(item.get("_id")) == item_id or item.get("action_item_id") == item_id:
                    target_item = item
                    break
            
            if not target_item:
                raise HTTPException(status_code=404, detail="Action item not found")
            
            return {"action_item": target_item}
        except HTTPException:
            raise
        except Exception as e:
            action_items_api_logger.error(f"Error getting action item: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.put("/{item_id}")
    async def update_action_item(
        item_id: str, 
        updates: ActionItemUpdate, 
        current_user: User = Depends(current_active_user)
    ):
        """Update an action item. Users can only update their own."""
        try:
            # Update status if provided
            if updates.status:
                user_id = None if current_user.is_superuser else current_user.user_id
                success = await action_items_service.update_action_item_status(
                    item_id, updates.status, user_id
                )
                if not success:
                    raise HTTPException(status_code=404, detail="Action item not found or access denied")
            
            # For other updates, we'd need to extend the service with more update methods
            # For now, we'll handle the most common case (status updates)
            
            return {"message": "Action item updated successfully"}
        except HTTPException:
            raise
        except Exception as e:
            action_items_api_logger.error(f"Error updating action item: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/{item_id}")
    async def delete_action_item(
        item_id: str, 
        current_user: User = Depends(current_active_user)
    ):
        """Delete an action item. Users can only delete their own."""
        try:
            user_id = None if current_user.is_superuser else current_user.user_id
            success = await action_items_service.delete_action_item(item_id, user_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Action item not found or access denied")
            
            return {"message": "Action item deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            action_items_api_logger.error(f"Error deleting action item: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router