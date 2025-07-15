
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from utils.logging import audio_logger, memory_logger


router = APIRouter()

from memory.memory_service import get_memory_service 

memory_service = get_memory_service()


@router.get("/api/memories")
async def get_memories(user_id: str, limit: int = 100):
    """Retrieves memories from the mem0 store with optional filtering."""
    memory_logger.info(f"Fetching memories for user {user_id}")
    try:
        all_memories = memory_service.get_all_memories(user_id=user_id, limit=limit)
        memory_logger.info(f"Retrieved {len(all_memories)} memories for user {user_id}")
        return JSONResponse(content=all_memories)
    except Exception as e:
        memory_logger.error(f"Error fetching memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching memories"}
        )


@router.get("/api/memories/search")
async def search_memories(user_id: str, query: str, limit: int = 10):
    """Search memories using semantic similarity for better retrieval."""
    try:
        relevant_memories = memory_service.search_memories(
            query=query, user_id=user_id, limit=limit
        )
        return JSONResponse(content=relevant_memories)
    except Exception as e:
        memory_logger.error(f"Error searching memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error searching memories"}
        )


@router.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        memory_service.delete_memory(memory_id=memory_id)
        return JSONResponse(
            content={"message": f"Memory {memory_id} deleted successfully"}
        )
    except Exception as e:
        memory_logger.error(f"Error deleting memory {memory_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error deleting memory"}
        )


