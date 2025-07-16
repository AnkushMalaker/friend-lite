# Memory Configuration Guide for New Users

> 🎯 **Goal**: Understand and customize how Friend-Lite extracts memories from conversations

## Quick Start: 3-Step Memory Setup

### Step 1: Choose Your LLM Provider

**Option A: OpenAI (Recommended)**
```bash
# In your .env file
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o  # Best for reliable JSON extraction
```

**Option B: Local Ollama**
```bash
# In your .env file  
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 2: Configure Memory Extraction

Edit `memory_config.yaml` to customize memory behavior:

```yaml
# Basic settings
memory_extraction:
  enabled: true  # Turn memory extraction on/off
  prompt: "Extract key information from this conversation."
  
fact_extraction:
  enabled: true  # Extract structured facts (names, dates, etc.)
  
categorization:
  enabled: false  # Categorize memories (work, personal, etc.)
```

### Step 3: Quality Control (Optional)

Control when memories are created:

```yaml
quality_control:
  min_conversation_length: 10  # Skip very short conversations
  min_content_ratio: 0.3       # Skip conversations with too little actual content
  skip_patterns:               # Skip conversations containing these phrases
    - "testing"
    - "hello world"
```

## Understanding Memory Types

Friend-Lite extracts **3 types of memories**:

### 1. **General Memories** 
- **What**: Conversation summaries, topics discussed, insights
- **Example**: "User discussed vacation plans to Japan in spring"
- **Configure**: `memory_extraction` section

### 2. **Structured Facts**
- **What**: Specific, verifiable information  
- **Example**: "Meeting with John Smith on Dec 15th at 2 PM"
- **Configure**: `fact_extraction` section

### 3. **Categories** (Optional)
- **What**: Auto-classify memories into topics
- **Example**: "work", "family", "health", "finance"
- **Configure**: `categorization` section

## Memory Extraction Triggers

Memories are extracted when:
- ✅ Conversation ends (timeout or manual close)
- ✅ Audio stream disconnects  
- ✅ "Close conversation" button clicked
- ❌ Not extracted in real-time during conversation

## Customization Examples

### Example 1: Work-Focused Memory Extraction
```yaml
memory_extraction:
  prompt: |
    Extract work-related information from this conversation including:
    - Project updates and deadlines
    - Meeting notes and action items  
    - Team members and their responsibilities
    - Technical discussions and decisions

fact_extraction:
  enabled: true
  prompt: |
    Extract work facts: project names, deadlines, team member names, 
    meeting times, budget numbers, technical requirements.
```

### Example 2: Personal Life Focus
```yaml
memory_extraction:
  prompt: |
    Extract personal information including:
    - Family and friend interactions
    - Plans and appointments
    - Interests and hobbies discussed
    - Personal goals and thoughts

quality_control:
  min_conversation_length: 20  # Longer threshold for personal conversations
```

### Example 3: Minimal Memory Storage
```yaml
memory_extraction:
  enabled: true
  prompt: "Extract only the most important key points."

fact_extraction:
  enabled: false  # Disable detailed fact extraction

quality_control:
  min_conversation_length: 30     # Only long conversations
  min_content_ratio: 0.5          # High content quality requirement
```

## Testing Your Configuration

### 1. **Create Test Conversation**
- Record a short audio clip discussing a topic
- Wait for conversation to end (timeout or close manually)
- Check memories in the dashboard

### 2. **Check Memory Debug Logs**
```bash
# View memory extraction logs
docker compose logs friend-backend | grep memory_service

# Look for these log messages:
# ✅ "Successfully created X memories"
# ❌ "Memory extraction failed"
# ⚠️ "Skipping conversation due to quality control"
```

### 3. **Use Debug API**
```bash
# Check recent memory processing
curl "http://localhost:8000/api/debug/memory/recent" -H "Authorization: Bearer $TOKEN"

# View specific conversation processing
curl "http://localhost:8000/api/debug/memory/conversation/AUDIO_UUID" -H "Authorization: Bearer $TOKEN"
```

## Troubleshooting Common Issues

### Issue 1: No Memories Created
**Symptoms**: Conversations finish but no memories appear

**Solutions**:
1. Check if conversation is too short (`min_conversation_length`)
2. Verify LLM provider is configured correctly
3. Check logs for errors: `docker compose logs friend-backend | grep ERROR`

### Issue 2: Poor Memory Quality  
**Symptoms**: Memories are vague or irrelevant

**Solutions**:
1. Improve the extraction prompt with more specific instructions
2. Switch to OpenAI GPT-4o for better results
3. Increase `temperature` for more creative extraction

### Issue 3: JSON Parsing Errors
**Symptoms**: Logs show "Failed to parse JSON response"

**Solutions**:
1. Use OpenAI instead of Ollama (more reliable JSON)
2. Add JSON format examples to your prompt
3. Reduce `max_tokens` to prevent truncated JSON

## Advanced Configuration

### Custom Categories
```yaml
categorization:
  enabled: true
  categories:
    - name: "work"
      keywords: ["project", "meeting", "deadline", "team"]
    - name: "health" 
      keywords: ["doctor", "exercise", "medication", "diet"]
    - name: "family"
      keywords: ["kids", "spouse", "parents", "family"]
```

### Processing Optimization
```yaml
processing:
  parallel_extraction: true   # Process memory types in parallel
  timeout_seconds: 60        # Max time for memory extraction
  retry_attempts: 3          # Retry failed extractions
```

### Quality Control Filters
```yaml
quality_control:
  max_conversation_length: 3600  # Skip very long conversations (1 hour)
  skip_patterns:
    - "test"
    - "1 2 3"
    - "hello hello"
  content_filters:
    - "audio_quality_threshold": 0.7
    - "speech_ratio_threshold": 0.4
```

## Getting Help

### Documentation Files
- **Complete Memory Reference**: `Docs/memories.md`
- **Configuration Reference**: `memory_config.yaml` (with comments)
- **API Reference**: `Docs/api.md`
- **Troubleshooting**: `Docs/troubleshooting.md`

### Debug Tools
- **Memory Dashboard**: Streamlit UI → "🧠 View All User Memories"
- **Debug API**: `/api/debug/memory/*` endpoints
- **Logs**: `docker compose logs friend-backend | grep memory`

### Support Channels
- **Issues**: GitHub repository issues
- **Discussions**: GitHub discussions for questions
- **Documentation**: All docs in `backends/advanced-backend/Docs/`

---

💡 **Pro Tip**: Start with the default configuration and gradually customize based on your specific use case. The defaults work well for most scenarios!