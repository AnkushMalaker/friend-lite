  1. Docs/quickstart.md (15 min)
  2. Docs/architecture.md (20 min)
  3. main.py - just the imports and WebSocket sections (15 min)
  4. memory_config.yaml (10 min)

  🔧 "I want to work on memory extraction"

  1. Docs/quickstart.md → Docs/memories.md
  2. memory_config.yaml (memory_extraction section)
  3. main.py lines 1047-1065 (trigger)
  4. main.py lines 1163-1195 (processing)
  5. src/memory/memory_service.py
  6. src/memory_debug.py (for tracking)

  📋 "I want to work on action items"

  1. Docs/quickstart.md → Docs/action-items.md
  2. memory_config.yaml (action_item_extraction section)
  3. main.py lines 1341-1378 (real-time processing)
  4. src/action_items_service.py
  5. ACTION_ITEMS_CLEANUP_SUMMARY.md (architecture)

  🐛 "I want to debug pipeline issues"

  1. MEMORY_DEBUG_IMPLEMENTATION.md
  2. src/memory_debug.py
  3. src/memory_debug_api.py  
  4. API endpoints: /api/debug/memory/*

  🏗️ "I want to understand the full architecture"

  1. Docs/architecture.md
  2. main.py (full file, focusing on class structures)
  3. src/auth.py (authentication flow)
  4. src/users.py (user management)
  5. All service files (memory_service.py, action_items_service.py)

  🎯 Key Concepts to Understand

  Data Flow

  Audio → Transcription → Dual Processing
                        ├─ Memory Pipeline (end-of-conversation)
                        └─ Action Item Pipeline (real-time per-segment)