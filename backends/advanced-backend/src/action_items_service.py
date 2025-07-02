import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
import logging
import ollama

# Set up logging
action_items_logger = logging.getLogger("action_items")

class ActionItemsService:
    """
    MongoDB-based action items service with full CRUD operations.
    Replaces the Mem0-based implementation for better update capabilities.
    """
    
    def __init__(self, collection: AsyncIOMotorCollection, ollama_client: ollama.Client):
        self.collection = collection
        self.ollama_client = ollama_client
        self._initialized = False
    
    async def initialize(self):
        """Initialize the service and create indexes for performance."""
        if self._initialized:
            return
        
        try:
            # Create indexes for better query performance
            await self.collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.collection.create_index([("user_id", 1), ("status", 1)])
            await self.collection.create_index([("user_id", 1), ("assignee", 1)])
            await self.collection.create_index([("audio_uuid", 1)])
            await self.collection.create_index([("description", "text")])  # Text search index
            
            self._initialized = True
            action_items_logger.info("Action items service initialized with MongoDB")
        except Exception as e:
            action_items_logger.error(f"Failed to initialize action items service: {e}")
            raise
    
    async def extract_and_store_action_items(self, transcript: str, client_id: str, audio_uuid: str) -> int:
        """
        Extract action items from transcript and store them in MongoDB.
        Returns the number of action items extracted and stored.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Extract action items from the transcript
            action_items = await self._extract_action_items_from_transcript(transcript, client_id, audio_uuid)
            
            if not action_items:
                action_items_logger.info(f"No action items found in transcript for {audio_uuid}")
                return 0
            
            # Store action items in MongoDB
            success_count = await self._store_action_items(action_items, client_id, audio_uuid)
            
            action_items_logger.info(f"Successfully extracted and stored {success_count}/{len(action_items)} action items for {audio_uuid}")
            return success_count
                
        except Exception as e:
            action_items_logger.error(f"Error extracting action items for {audio_uuid}: {e}")
            return 0
    
    async def _extract_action_items_from_transcript(self, transcript: str, client_id: str, audio_uuid: str) -> List[Dict[str, Any]]:
        """Extract action items from transcript using Ollama."""
        try:
            extraction_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intelligent assistant that reads transcripts and extracts all potential action items, even informal or implied ones.

Your output must be a **JSON array**, where action item includes:
- description: A short summary of the task
- assignee: Who should do it ("unassigned" if unclear)
- due_date: When it should be done ("not_specified" if not mentioned)
- priority: high / medium / low / not_specified
- context: Why or how the task came up
- tool: The name of the tool required, if any ("check_email", "check_calendar", "set_alarm"), or "none" if no tool is needed

Rules:
- Identify both explicit tasks and implied ones.
- Suggest a tool only when the task obviously requires it or could be automated.
- If it's a general human task with no clear automation, use `"none"` for tool.

Return **only** a JSON array. No explanation or extra text.

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Transcript:
<start_transcript>
{transcript}
<end_transcript>
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            response = self.ollama_client.generate(
                model="llama3.1:latest",
                prompt=extraction_prompt,
                options={"temperature": 0.1}
            )
            
            response_text = response['response'].strip()
            
            # Handle empty responses
            if not response_text or response_text.lower() in ['none', 'no action items', '[]']:
                return []
            
            # Parse JSON response
            action_items = json.loads(response_text)
            
            # Validate response format
            if not isinstance(action_items, list):
                action_items_logger.warning(f"Action item extraction returned non-list for {audio_uuid}: {type(action_items)}")
                return []
            
            # Enrich each action item with metadata
            for i, item in enumerate(action_items):
                item.update({
                    "id": f"action_{audio_uuid}_{i}_{int(time.time())}",
                    "status": "open",
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                    "source": "transcript_extraction"
                })
                
                # TODO: Handle all tools here, these can be imported from other files
                # Handle set_alarm tool, this can be another llm call to mcp with description as input 
                if item.get("tool") == "set_alarm":
                    description = item.get("description", "")
                    action_items_logger.info(f"Calling set alarm service with description: {description}")
            
            action_items_logger.info(f"Extracted {len(action_items)} action items from {audio_uuid}")
            return action_items
            
        except json.JSONDecodeError as e:
            action_items_logger.error(f"Failed to parse action items JSON for {audio_uuid}: {e}")
            return []
        except Exception as e:
            action_items_logger.error(f"Error extracting action items from transcript for {audio_uuid}: {e}")
            return []
    
    async def _store_action_items(self, action_items: List[Dict[str, Any]], client_id: str, audio_uuid: str) -> int:
        """Store action items in MongoDB."""
        try:
            if not action_items:
                return 0
            
            # Prepare documents for insertion
            documents = []
            for item in action_items:
                document = {
                    "action_item_id": item.get("id"),
                    "user_id": client_id,
                    "audio_uuid": audio_uuid,
                    "description": item.get("description", ""),
                    "assignee": item.get("assignee", "unassigned"),
                    "due_date": item.get("due_date", "not_specified"),
                    "priority": item.get("priority", "not_specified"),
                    "status": item.get("status", "open"),
                    "context": item.get("context", ""),
                    "source": item.get("source", "transcript_extraction"),
                    "created_at": item.get("created_at", int(time.time())),
                    "updated_at": item.get("updated_at", int(time.time()))
                }
                documents.append(document)
            
            # Insert all action items
            result = await self.collection.insert_many(documents)
            success_count = len(result.inserted_ids)
            
            action_items_logger.info(f"Stored {success_count} action items for {audio_uuid}")
            return success_count
            
        except Exception as e:
            action_items_logger.error(f"Error storing action items for {audio_uuid}: {e}")
            return 0
    
    async def get_action_items(self, user_id: str, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get action items for a user with optional status filtering."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build query filter
            query = {"user_id": user_id}
            if status_filter:
                query["status"] = status_filter
            
            # Execute query with sorting (newest first)
            cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
            action_items = []
            
            async for doc in cursor:
                # Convert MongoDB ObjectId to string and remove it
                doc["_id"] = str(doc["_id"])
                action_items.append(doc)
            
            action_items_logger.info(f"Retrieved {len(action_items)} action items for user {user_id} (status_filter: {status_filter})")
            return action_items
            
        except Exception as e:
            action_items_logger.error(f"Error fetching action items for user {user_id}: {e}")
            return []
    
    async def update_action_item_status(self, action_item_id: str, new_status: str, user_id: Optional[str] = None) -> bool:
        """Update the status of an action item."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build query - use action_item_id or _id
            query = {}
            if action_item_id.startswith("action_"):
                query["action_item_id"] = action_item_id
            else:
                # Assume it's a MongoDB ObjectId
                from bson import ObjectId
                try:
                    query["_id"] = ObjectId(action_item_id)
                except:
                    query["action_item_id"] = action_item_id
            
            # Add user_id to query if provided for additional security
            if user_id:
                query["user_id"] = user_id
            
            # Update the document
            update_data = {
                "$set": {
                    "status": new_status,
                    "updated_at": int(time.time())
                }
            }
            
            result = await self.collection.update_one(query, update_data)
            
            if result.modified_count > 0:
                action_items_logger.info(f"Updated action item {action_item_id} status to {new_status}")
                return True
            else:
                action_items_logger.warning(f"No action item found with id {action_item_id}")
                return False
            
        except Exception as e:
            action_items_logger.error(f"Error updating action item status for {action_item_id}: {e}")
            return False
    
    async def search_action_items(self, query: str, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search action items by text query using MongoDB text search."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use MongoDB text search if available, otherwise regex search
            search_query = {
                "user_id": user_id,
                "$or": [
                    {"description": {"$regex": query, "$options": "i"}},
                    {"context": {"$regex": query, "$options": "i"}},
                    {"assignee": {"$regex": query, "$options": "i"}}
                ]
            }
            
            cursor = self.collection.find(search_query).sort("created_at", -1).limit(limit)
            action_items = []
            
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                action_items.append(doc)
            
            action_items_logger.info(f"Search found {len(action_items)} action items for query '{query}'")
            return action_items
            
        except Exception as e:
            action_items_logger.error(f"Error searching action items for user {user_id}: {e}")
            return []
    
    async def delete_action_item(self, action_item_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a specific action item."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build query - use action_item_id or _id
            query = {}
            if action_item_id.startswith("action_"):
                query["action_item_id"] = action_item_id
            else:
                from bson import ObjectId
                try:
                    query["_id"] = ObjectId(action_item_id)
                except:
                    query["action_item_id"] = action_item_id
            
            # Add user_id to query if provided for additional security
            if user_id:
                query["user_id"] = user_id
            
            result = await self.collection.delete_one(query)
            
            if result.deleted_count > 0:
                action_items_logger.info(f"Deleted action item with id {action_item_id}")
                return True
            else:
                action_items_logger.warning(f"No action item found with id {action_item_id}")
                return False
                
        except Exception as e:
            action_items_logger.error(f"Error deleting action item {action_item_id}: {e}")
            return False
    
    async def create_action_item(self, user_id: str, description: str, assignee: str = "unassigned", 
                               due_date: str = "not_specified", priority: str = "medium", 
                               context: str = "") -> Optional[Dict[str, Any]]:
        """Create a new action item manually."""
        if not self._initialized:
            await self.initialize()
        
        try:
            current_time = int(time.time())
            action_item_id = f"manual_{user_id}_{current_time}"
            
            document = {
                "action_item_id": action_item_id,
                "user_id": user_id,
                "audio_uuid": None,  # No associated conversation
                "description": description,
                "assignee": assignee,
                "due_date": due_date,
                "priority": priority,
                "status": "open",
                "context": context,
                "source": "manual_creation",
                "created_at": current_time,
                "updated_at": current_time
            }
            
            result = await self.collection.insert_one(document)
            
            if result.inserted_id:
                document["_id"] = str(result.inserted_id)
                action_items_logger.info(f"Created manual action item {action_item_id} for user {user_id}")
                return document
            else:
                action_items_logger.error(f"Failed to create action item for user {user_id}")
                return None
                
        except Exception as e:
            action_items_logger.error(f"Error creating action item for user {user_id}: {e}")
            return None
    
    async def get_action_item_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for user's action items."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use aggregation pipeline for statistics
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "by_status": {"$push": "$status"},
                    "by_priority": {"$push": "$priority"},
                    "by_assignee": {"$push": "$assignee"}
                }}
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if not result:
                return {
                    "total": 0,
                    "by_status": {},
                    "by_priority": {},
                    "by_assignee": {},
                    "recent_count": 0
                }
            
            data = result[0]
            
            # Count by status
            status_counts = {}
            for status in data["by_status"]:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by priority
            priority_counts = {}
            for priority in data["by_priority"]:
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Count by assignee
            assignee_counts = {}
            for assignee in data["by_assignee"]:
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
            
            # Get recent count (last 7 days)
            seven_days_ago = int(time.time()) - (7 * 24 * 60 * 60)
            recent_count = await self.collection.count_documents({
                "user_id": user_id,
                "created_at": {"$gte": seven_days_ago}
            })
            
            return {
                "total": data["total"],
                "by_status": status_counts,
                "by_priority": priority_counts,
                "by_assignee": assignee_counts,
                "recent_count": recent_count
            }
            
        except Exception as e:
            action_items_logger.error(f"Error getting action item stats for user {user_id}: {e}")
            return {
                "total": 0,
                "by_status": {},
                "by_priority": {},
                "by_assignee": {},
                "recent_count": 0
            } 
        


# import pyperclip
# transcript = "set an alarm for 10am"
# extraction_prompt = f"""
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are an intelligent assistant that reads transcripts and extracts all potential action items, even informal or implied ones.

# Your output must be a **JSON**, where action item includes:
# - description: A short summary of the task
# - assignee: Who should do it ("unassigned" if unclear)
# - due_date: When it should be done ("not_specified" if not mentioned)
# - priority: high / medium / low / not_specified
# - context: Why or how the task came up
# - tool: The name of the tool required, if any ("check_email", "check_calendar", "set_alarm"), or "none" if no tool is needed

# Rules:
# - Identify both explicit tasks and implied ones.
# - Suggest a tool only when the task obviously requires it or could be automated.
# - If it's a general human task with no clear automation, use `"none"` for tool.

# Return **only** a JSON. No explanation or extra text.

# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Transcript:
# <start_transcript>
# {transcript}
# <end_transcript>
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """
# pyperclip.copy(extraction_prompt)