# Action Items Configuration and Usage

> 📖 **Prerequisite**: Read [quickstart.md](./quickstart.md) first for system overview.

## Overview

The friend-lite backend includes a comprehensive action items system that automatically extracts tasks and commitments from conversations. This system operates in **real-time** alongside the memory extraction system, providing immediate task detection and management capabilities.

**Code References**:
- **Main Implementation**: `src/action_items_service.py` (MongoDB-based storage and processing)
- **Real-time Processing**: `main.py:1341-1378` (per-transcript-segment processing)
- **API Endpoints**: `main.py:2671-2800` (action items CRUD operations)
- **Configuration**: `memory_config.yaml` (action_item_extraction section)

## Architecture

### Dual Processing System

The action items system operates in parallel with memory extraction:

```
Audio → Transcription → Dual Processing
                      ├─ Memory Pipeline (end-of-conversation)
                      └─ Action Item Pipeline (real-time per-segment)
```

### Key Components

1. **Real-time Detection**: Each transcript segment is checked for action item triggers
2. **Configurable Extraction**: YAML-based configuration for prompts and triggers
3. **MongoDB Storage**: Action items stored in dedicated collection with full CRUD
4. **Debug Tracking**: SQLite-based tracking of extraction process
5. **User-Centric Design**: All action items keyed by user_id, not client_id

### Architecture Cleanup

**Previous Issue**: The system had duplicated action item processing in two places:
- `ActionItemsService` (MongoDB-based, primary handler)
- `MemoryService` (Mem0-based, unused legacy code)

**Current Architecture**: Clean separation of concerns:
- **`ActionItemsService`**: Handles ALL action item operations (MongoDB-based)
- **`MemoryService`**: Handles ONLY memory operations (Mem0-based)
- **Debug System**: Tracks both memories and action items in unified SQLite database

## Configuration

### Basic Configuration (`memory_config.yaml`)

**Configuration Loading**: See `src/memory_config_loader.py:get_action_item_extraction_config()` for how this configuration is loaded and used.

```yaml
action_item_extraction:
  # Enable/disable action item extraction
  enabled: true
  
  # Trigger phrases that indicate action items
  trigger_phrases:
    - "simon says"        # Primary trigger (case-insensitive)
    - "action item"       # Explicit action item
    - "todo"              # Simple todo
    - "follow up"         # Follow-up tasks
    - "next step"         # Next steps
    - "homework"          # Assignments
    - "deliverable"       # Project deliverables
    - "deadline"          # Time-sensitive tasks
    - "schedule"          # Scheduling tasks
    - "reminder"          # Reminders
  
  # LLM extraction prompt
  prompt: |
    Extract actionable tasks and commitments from this conversation.
    
    Look for:
    - Explicit commitments ("I'll send you the report")
    - Requested actions ("Can you review the document?")
    - Scheduled tasks ("We need to meet next week")
    - Follow-up items ("Let's check on this tomorrow")
    - Deliverables mentioned ("The presentation is due Friday")
    
    For each action item, determine:
    - What needs to be done (clear, specific description)
    - Who is responsible (assignee)
    - When it's due (deadline if mentioned)
    - Priority level (high/medium/low)
    
    Return ONLY valid JSON array. If no action items found, return [].
    
    Example format:
    [
      {
        "description": "Send project status report to team",
        "assignee": "John",
        "due_date": "Friday",
        "priority": "high",
        "context": "Discussed in weekly team meeting"
      }
    ]
  
  # LLM settings for action item extraction
  llm_settings:
    temperature: 0.1      # Low temperature for consistent extraction
    max_tokens: 1000      # Sufficient for multiple action items
    model: "llama3.1:latest"  # Can be overridden by environment
```

### Advanced Configuration

```yaml
action_item_extraction:
  enabled: true
  
  # Enhanced trigger detection
  trigger_phrases:
    - "simon says"
    - "action item"
    - "i need to"
    - "we should"
    - "let's"
    - "can you"
    - "please"
    - "remember to"
    - "don't forget"
    - "make sure"
  
  # Custom extraction prompt with specific instructions
  prompt: |
    You are an expert task manager. Extract actionable items from this conversation.
    
    Focus on:
    1. Specific commitments with clear ownership
    2. Time-bound tasks with deadlines
    3. Follow-up actions requiring completion
    4. Deliverables with clear outcomes
    
    For each action item, provide:
    - description: Clear, specific task description
    - assignee: Person responsible (use "unassigned" if unclear)
    - due_date: Deadline if mentioned (use "not_specified" if not clear)
    - priority: Based on urgency (high/medium/low/not_specified)
    - context: Brief context about when/why this was mentioned
    
    Return ONLY valid JSON array. Empty array if no action items found.
  
  # Fine-tuned LLM parameters
  llm_settings:
    temperature: 0.05     # Very low for consistent extraction
    max_tokens: 1500      # More tokens for detailed extraction
    model: "llama3.1:latest"
```

## Usage Examples

### Trigger Phrase Examples

The system detects action items when trigger phrases are present:

```
✅ "Simon says we need to schedule a follow-up meeting"
✅ "Action item: John will send the report by Friday"
✅ "Todo: Review the contract before tomorrow"
✅ "Follow up with the client about their requirements"
✅ "Next step is to finalize the budget proposal"
✅ "Can you please update the documentation?"
✅ "Let's schedule a review meeting for next week"
✅ "Don't forget to submit the quarterly report"
```

### Action Item Data Structure

```json
{
  "description": "Send project status report to team",
  "assignee": "John Smith",
  "due_date": "Friday, December 15th",
  "priority": "high",
  "status": "open",
  "context": "Discussed in weekly team meeting",
  "audio_uuid": "audio_12345",
  "client_id": "user1-laptop",
  "user_id": "user1",
  "created_at": 1703548800,
  "updated_at": 1703548800
}
```

## API Endpoints

### Action Items Management

**API Implementation**: See `main.py:2671-2800` for complete CRUD endpoint implementations.

```bash
# Get user's action items
GET /api/action_items?status=open&limit=20

# Get specific action item
GET /api/action_items/{action_item_id}

# Update action item status
PUT /api/action_items/{action_item_id}
Content-Type: application/json
{
  "status": "completed"
}

# Search action items
GET /api/action_items/search?query=report&status=open

# Delete action item
DELETE /api/action_items/{action_item_id}
```

### Debug & Monitoring

```bash
# View action item extraction stats
GET /api/debug/memory/stats

# View recent action item sessions
GET /api/debug/memory/sessions

# Debug specific session
GET /api/debug/memory/session/{audio_uuid}

# View pipeline trace
GET /api/debug/memory/pipeline/{audio_uuid}
```

## Debug Tracking

The system tracks all action item extraction attempts:

### What's Tracked

- **Extraction Attempts**: Success/failure of each extraction
- **Processing Time**: How long each extraction takes
- **Prompt Used**: Which prompt was used for extraction
- **LLM Model**: Which model performed the extraction
- **Transcript Length**: Size of input text
- **Error Details**: Specific error messages for failed extractions

### Debug Database Schema

```sql
-- Action item extractions are stored as memory_extractions with type='action_item'
SELECT 
    audio_uuid,
    memory_text,
    extraction_prompt,
    metadata_json,
    created_at
FROM memory_extractions 
WHERE memory_type = 'action_item';

-- Processing attempts show success/failure patterns
SELECT 
    audio_uuid,
    attempt_type,
    success,
    error_message,
    processing_time_ms
FROM extraction_attempts 
WHERE attempt_type = 'action_item_extraction';
```

## Performance Optimization

### Configuration Tips

1. **Adjust Trigger Phrases**: Add domain-specific triggers for your use case
2. **Tune LLM Parameters**: Lower temperature for consistency, higher for creativity
3. **Optimize Prompts**: Include examples specific to your workflow
4. **Monitor Processing Time**: Use debug endpoints to identify bottlenecks

### Quality Control

```yaml
quality_control:
  # Skip very short transcripts
  min_conversation_length: 10
  
  # Skip transcripts with low meaningful content
  skip_low_content: true
  min_content_ratio: 0.2
  
  # Skip common filler patterns
  skip_patterns:
    - "^(um|uh|hmm|yeah|ok|okay)\\s*$"
    - "^test\\s*$"
```

## Integration with Memory System

Action items and memories work together:

1. **Shared Debug Tracking**: Both use the same SQLite debug database
2. **Coordinated Processing**: Both respect the same quality control settings
3. **User-Centric Storage**: Both keyed by user_id for proper isolation
4. **Unified Configuration**: Single YAML file controls both systems

## Troubleshooting

### Common Issues

1. **No Action Items Detected**
   - Check if trigger phrases are present in transcript
   - Verify `action_item_extraction.enabled: true` in config
   - Check debug logs for extraction attempts

2. **JSON Parsing Errors**
   - Review extraction prompt for clarity
   - Lower LLM temperature for more consistent output
   - Check debug database for exact error messages

3. **Performance Issues**
   - Monitor processing times in debug stats
   - Adjust `max_tokens` and `temperature` settings
   - Consider using quality control to filter low-value transcripts

### Debug Commands

```bash
# Test action item configuration
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/debug/memory/config/test?test_text=Simon%20says%20we%20need%20to%20schedule%20a%20meeting"

# View extraction statistics
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/debug/memory/stats

# Check recent processing
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/debug/memory/sessions?limit=10
```

## Best Practices

1. **Use Specific Trigger Phrases**: Add domain-specific triggers for your use case
2. **Test Prompts Regularly**: Use the debug API to test prompt effectiveness
3. **Monitor Performance**: Check debug stats for processing times and success rates
4. **Customize for Your Workflow**: Adjust prompts and triggers based on your conversation patterns
5. **Regular Configuration Updates**: Reload configuration without restart using the API

This action items system provides comprehensive task management capabilities with full configurability and debugging support, integrating seamlessly with the memory extraction pipeline.