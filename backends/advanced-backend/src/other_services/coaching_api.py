from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
import logging

# Import authentication
from auth import current_active_user, User

# Set up logging
coaching_api_logger = logging.getLogger("coaching_api")

def create_coaching_router(coaching_service) -> APIRouter:
    """Create FastAPI router for coaching endpoints."""
    router = APIRouter(prefix="/api/coaching", tags=["coaching"])
    
    @router.get("/insights")
    async def get_coaching_insights(
        current_user: User = Depends(current_active_user),
        limit: Optional[int] = Query(10, ge=1, le=100)
    ):
        """Get coaching insights for the current user."""
        try:
            # This is a placeholder - in a real implementation you'd:
            # 1. Query insights from a database
            # 2. Use the coaching service to generate insights
            # 3. Return personalized coaching data
            
            insights = {
                "total_insights": coaching_service.insights_count,
                "recent_insights": [
                    {
                        "type": "communication_pattern",
                        "message": "You've been having more detailed conversations lately!",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "user_id": current_user.user_id
            }
            
            return {"insights": insights}
        except Exception as e:
            coaching_api_logger.error(f"Error getting coaching insights: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/stats")
    async def get_coaching_stats(
        current_user: User = Depends(current_active_user)
    ):
        """Get coaching statistics for the current user."""
        try:
            stats = {
                "total_sessions": coaching_service.insights_count,
                "improvement_areas": ["communication", "listening"],
                "recent_activity": "active"
            }
            
            return {"stats": stats}
        except Exception as e:
            coaching_api_logger.error(f"Error getting coaching stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router