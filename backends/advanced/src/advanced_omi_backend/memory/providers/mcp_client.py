"""Client for communicating with OpenMemory servers.

This module provides a client interface for interacting with the official
OpenMemory servers using REST API endpoints for memory operations.
"""

import logging
import uuid
from typing import List, Dict, Any
import httpx

memory_logger = logging.getLogger("memory_service")


class MCPClient:
    """Client for communicating with OpenMemory servers.
    
    Uses the official OpenMemory REST API:
    - POST /api/v1/memories - Create new memory
    - GET /api/v1/memories - List memories
    - DELETE /api/v1/memories - Delete memories
    
    Attributes:
        server_url: Base URL of the OpenMemory server (default: http://localhost:8765)
        client_name: Client identifier for memory tagging
        user_id: User identifier for memory isolation
        timeout: Request timeout in seconds
        client: HTTP client instance
    """
    
    def __init__(self, server_url: str, client_name: str = "friend_lite", user_id: str = "default", timeout: int = 30):
        """Initialize client for OpenMemory.
        
        Args:
            server_url: Base URL of the OpenMemory server
            client_name: Client identifier (used as app name)
            user_id: User identifier for memory isolation
            timeout: HTTP request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.client_name = client_name
        self.user_id = user_id
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def add_memories(self, text: str) -> List[str]:
        """Add memories to the OpenMemory server.
        
        Uses the REST API to create memories. OpenMemory will handle:
        - Memory extraction from text
        - Deduplication
        - Vector embedding and storage
        
        Args:
            text: Memory text to store
            
        Returns:
            List of created memory IDs
            
        Raises:
            MCPError: If the server request fails
        """
        try:
            # Use REST API endpoint for creating memories
            response = await self.client.post(
                f"{self.server_url}/api/v1/memories/",
                json={
                    "user_id": self.user_id,
                    "text": text,
                    "metadata": {
                        "source": "friend_lite",
                        "client": self.client_name
                    },
                    "infer": True,  # Let OpenMemory extract memories
                    "app": self.client_name  # Use client name as app name
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle None result - OpenMemory returns None when no memory is created
            # (due to deduplication, insufficient content, etc.)
            if result is None:
                memory_logger.info("OpenMemory returned None - no memory created (likely deduplication)")
                return []
            
            # Handle error response
            if isinstance(result, dict) and "error" in result:
                memory_logger.error(f"OpenMemory error: {result['error']}")
                return []
            
            # Extract memory ID from response
            if isinstance(result, dict):
                memory_id = result.get("id") or str(uuid.uuid4())
                return [memory_id]
            elif isinstance(result, list):
                return [str(item.get("id", uuid.uuid4())) for item in result]
            
            # Default success response
            return [str(uuid.uuid4())]
            
        except httpx.HTTPError as e:
            memory_logger.error(f"HTTP error adding memories: {e}")
            raise MCPError(f"HTTP error: {e}")
        except Exception as e:
            memory_logger.error(f"Error adding memories: {e}")
            raise MCPError(f"Failed to add memories: {e}")
    
    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for memories using semantic similarity.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of memory dictionaries with content and metadata
        """
        try:
            # First get the app_id for the default app
            apps_response = await self.client.get(f"{self.server_url}/api/v1/apps/")
            apps_response.raise_for_status()
            apps_data = apps_response.json()
            
            if not apps_data.get("apps") or len(apps_data["apps"]) == 0:
                memory_logger.warning("No apps found in OpenMemory MCP for search")
                return []
            
            # Find the app matching our client name, or use first app as fallback
            app_id = None
            for app in apps_data["apps"]:
                if app["name"] == self.client_name:
                    app_id = app["id"]
                    break
            
            if not app_id:
                memory_logger.warning(f"App '{self.client_name}' not found, using first available app")
                app_id = apps_data["apps"][0]["id"]
            
            # Use app-specific memories endpoint with search
            params = {
                "user_id": self.user_id,
                "search_query": query,
                "page": 1,
                "size": limit
            }
            
            response = await self.client.get(
                f"{self.server_url}/api/v1/apps/{app_id}/memories",
                params=params
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract memories from app-specific response format
            if isinstance(result, dict) and "memories" in result:
                memories = result["memories"]
            elif isinstance(result, list):
                memories = result
            else:
                memories = []
            
            # Format memories for Friend-Lite
            formatted_memories = []
            for memory in memories:
                formatted_memories.append({
                    "id": memory.get("id", str(uuid.uuid4())),
                    "content": memory.get("content", "") or memory.get("text", ""),
                    "metadata": memory.get("metadata_", {}) or memory.get("metadata", {}),
                    "created_at": memory.get("created_at"),
                    "score": memory.get("score", 0.0)  # No score from list API
                })
            
            return formatted_memories[:limit]
            
        except Exception as e:
            memory_logger.error(f"Error searching memories: {e}")
            return []
    
    async def list_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all memories for the current user.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries
        """
        try:
            # First get the app_id for the default app
            apps_response = await self.client.get(f"{self.server_url}/api/v1/apps/")
            apps_response.raise_for_status()
            apps_data = apps_response.json()
            
            if not apps_data.get("apps") or len(apps_data["apps"]) == 0:
                memory_logger.warning("No apps found in OpenMemory MCP")
                return []
            
            # Find the app matching our client name, or use first app as fallback
            app_id = None
            for app in apps_data["apps"]:
                if app["name"] == self.client_name:
                    app_id = app["id"]
                    break
            
            if not app_id:
                memory_logger.warning(f"App '{self.client_name}' not found, using first available app")
                app_id = apps_data["apps"][0]["id"]
            
            # Use app-specific memories endpoint
            params = {
                "user_id": self.user_id,
                "page": 1,
                "size": limit
            }
            
            response = await self.client.get(
                f"{self.server_url}/api/v1/apps/{app_id}/memories",
                params=params
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract memories from app-specific response format
            if isinstance(result, dict) and "memories" in result:
                memories = result["memories"]
            elif isinstance(result, list):
                memories = result
            else:
                memories = []
            
            # Format memories
            formatted_memories = []
            for memory in memories:
                formatted_memories.append({
                    "id": memory.get("id", str(uuid.uuid4())),
                    "content": memory.get("content", "") or memory.get("text", ""),
                    "metadata": memory.get("metadata_", {}) or memory.get("metadata", {}),
                    "created_at": memory.get("created_at")
                })
            
            return formatted_memories
            
        except Exception as e:
            memory_logger.error(f"Error listing memories: {e}")
            return []
    
    async def delete_all_memories(self) -> int:
        """Delete all memories for the current user.
        
        Note: OpenMemory may not support bulk delete via REST API.
        This is typically done through MCP tools for safety.
        
        Returns:
            Number of memories that were deleted
        """
        try:
            # First get all memory IDs
            memories = await self.list_memories(limit=1000)
            if not memories:
                return 0
            
            memory_ids = [m["id"] for m in memories]
            
            # Delete memories using the batch delete endpoint
            response = await self.client.request(
                "DELETE",
                f"{self.server_url}/api/v1/memories/",
                json={
                    "memory_ids": memory_ids,
                    "user_id": self.user_id
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract count from response
            if isinstance(result, dict):
                if "message" in result:
                    # Parse message like "Successfully deleted 5 memories"
                    import re
                    match = re.search(r'(\d+)', result["message"])
                    return int(match.group(1)) if match else len(memory_ids)
                return result.get("deleted_count", len(memory_ids))
            
            return len(memory_ids)
            
        except Exception as e:
            memory_logger.error(f"Error deleting all memories: {e}")
            return 0
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            response = await self.client.request(
                "DELETE",
                f"{self.server_url}/api/v1/memories/",
                json={
                    "memory_ids": [memory_id],
                    "user_id": self.user_id
                }
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            memory_logger.warning(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to the OpenMemory server.
        
        Returns:
            True if server is reachable and responsive, False otherwise
        """
        try:
            # Test basic connectivity with health endpoint
            # OpenMemory may not have /health, try root or API endpoint
            for endpoint in ["/health", "/", "/api/v1/memories"]:
                try:
                    response = await self.client.get(
                        f"{self.server_url}{endpoint}",
                        params={"user_id": self.user_id, "page": 1, "size": 1}
                        if endpoint == "/api/v1/memories" else {}
                    )
                    if response.status_code in [200, 404, 422]:  # 404/422 means endpoint exists but params wrong
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            memory_logger.error(f"OpenMemory server connection test failed: {e}")
            return False


class MCPError(Exception):
    """Exception raised for MCP server communication errors."""
    pass