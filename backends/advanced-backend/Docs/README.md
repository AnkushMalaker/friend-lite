# Friend-Lite Backend Documentation Guide

## 📖 **New Developer Reading Order**

Welcome to friend-lite! This guide provides the optimal reading sequence to understand the complete voice → transcript → memories system.

---

## 🎯 **Start Here: System Overview**

### 1. **[Overview & Quick Start](./quickstart.md)** ⭐ *START HERE*
**Read first** - Complete system overview and setup guide
- What the system does (voice → memories)
- Key features and capabilities
- Basic setup and configuration
- **Code References**: `main.py`, `memory_config.yaml`, `docker-compose.yml`

### 2. **[System Architecture](./architecture.md)** 
**Read second** - Complete technical architecture with diagrams
- Component relationships and data flow
- Authentication and security architecture
- Deployment structure and containers
- **Code References**: `main.py:1-100`, `auth.py`, `users.py`

---

## 🔧 **Core Components Deep Dive**

### 3. **[Memory System](./memories.md)**
**Memory extraction and semantic search**

### 3a. **[Memory Configuration Guide](./memory-configuration-guide.md)** 🎯 *NEW USER GUIDE*
**Easy guide for configuring memory extraction**
- 3-step setup for memory extraction
- Understanding memory types (general, facts, categories)
- Customization examples and troubleshooting
- **Perfect for**: New users wanting to customize memory behavior
- How conversations become memories
- Mem0 integration and vector storage
- Configuration and customization options
- **Code References**: 
  - `src/memory/memory_service.py:159-282` (main processing)
  - `main.py:1047-1065` (conversation end trigger)
  - `main.py:1163-1195` (background processing)

### 4. **[Authentication System](./auth.md)**
**User management and security**
- Dual authentication (email + user_id)
- JWT tokens and OAuth integration
- User-centric data architecture
- **Code References**:
  - `src/auth.py` (authentication logic)
  - `src/users.py` (user management)
  - `main.py:1555-1563` (auth router setup)

---

## 🐛 **Advanced Topics**

### 5. **Memory Debug System** → `../MEMORY_DEBUG_IMPLEMENTATION.md`
**Pipeline tracking and debugging**
- How to track transcript → memory conversion
- Debug database schema and API endpoints
- Performance monitoring and troubleshooting
- **Code References**:
  - `src/memory_debug.py` (SQLite tracking)
  - `src/memory_debug_api.py` (debug endpoints)
  - `main.py:1562-1563` (debug router integration)

---

## 🔍 **Configuration & Customization**

### 6. **Configuration File** → `../memory_config.yaml`
**Central configuration for all extraction**
- Memory extraction settings and prompts
- Quality control and debug settings
- **Code References**:
  - `src/memory_config_loader.py` (config loading)
  - `src/memory/memory_service.py:176-204` (config usage)

---

## 🚀 **Quick Reference by Use Case**

### **"I want to understand the system quickly"** (30 min)
1. [quickstart.md](./quickstart.md) - System overview
2. [architecture.md](./architecture.md) - Technical architecture  
3. `main.py:1-200` - Core imports and setup
4. `memory_config.yaml` - Configuration overview

### **"I want to work on memory extraction"**
1. [memories.md](./memories.md) - Memory system details
2. `../memory_config.yaml` - Memory configuration
3. `src/memory/memory_service.py` - Implementation
4. `main.py:1047-1065, 1163-1195` - Processing triggers

### **"I want to debug pipeline issues"**
1. `../MEMORY_DEBUG_IMPLEMENTATION.md` - Debug system overview
2. `src/memory_debug.py` - Debug tracking implementation
3. API: `GET /api/debug/memory/stats` - Live debugging
4. `src/memory_debug_api.py` - Debug endpoints

### **"I want to understand authentication"**
1. [auth.md](./auth.md) - Authentication system
2. `src/auth.py` - Authentication implementation
3. `src/users.py` - User management
4. `main.py:1555-1563` - Auth router setup

---

## 📂 **File Organization Reference**

```
backends/advanced-backend/
├── Docs/                           # 📖 Documentation
│   ├── README.md                   # 👈 This file (start here)
│   ├── quickstart.md              # System overview & setup
│   ├── architecture.md            # Technical architecture
│   ├── memories.md                # Memory system details
│   └── auth.md                    # Authentication system
│
├── src/                           # 🔧 Source Code
│   ├── main.py                    # Core application (WebSocket, API)
│   ├── auth.py                    # Authentication system
│   ├── users.py                   # User management
│   ├── memory/
│   │   └── memory_service.py      # Memory system (Mem0)
│   ├── memory_debug.py            # Debug tracking (SQLite)
│   ├── memory_debug_api.py        # Debug API endpoints
│   └── memory_config_loader.py    # Configuration loading
│
├── memory_config.yaml             # 📋 Central configuration
├── MEMORY_DEBUG_IMPLEMENTATION.md # Debug system details
```

---

## 🎯 **Key Code Entry Points**

### **Audio Processing Pipeline**
- **Entry**: WebSocket endpoints in `main.py:1562+`
- **Transcription**: `main.py:1258-1340` (transcription processor)
- **Memory Trigger**: `main.py:1047-1065` (conversation end)

### **Data Storage**
- **Memories**: `src/memory/memory_service.py` → Mem0 → Qdrant
- **Debug Data**: `src/memory_debug.py` → SQLite

### **Configuration**
- **Loading**: `src/memory_config_loader.py`
- **File**: `memory_config.yaml`
- **Usage**: `src/memory/memory_service.py:176-204`

### **Authentication**
- **Setup**: `src/auth.py`
- **Users**: `src/users.py`  
- **Integration**: `main.py:1555-1563`

---

## 💡 **Reading Tips**

1. **Follow the references**: Each doc links to specific code files and line numbers
2. **Use the debug API**: `GET /api/debug/memory/stats` shows live system status
3. **Check configuration first**: Many behaviors are controlled by `memory_config.yaml`
4. **Understand the memory pipeline**: Memories (end-of-conversation)
5. **Test with curl**: All API endpoints have curl examples in the docs

---

## 🎯 **After Reading This Guide**

### **Next Steps for New Developers**

1. **Set up the system**: Follow [quickstart.md](./quickstart.md) to get everything running
2. **Test the API**: Use the curl examples in the documentation to test endpoints
3. **Explore the debug system**: Check `GET /api/debug/memory/stats` to see live data
4. **Modify configuration**: Edit `memory_config.yaml` to see how it affects extraction
5. **Read the code**: Start with `main.py` and follow the references in each doc

### **Contributing Guidelines**

- **Add code references**: When updating docs, include file paths and line numbers
- **Test your changes**: Use the debug API to verify your modifications work
- **Update configuration**: Add new settings to `memory_config.yaml` when needed
- **Follow the architecture**: Keep memories in their respective services

### **Getting Help**

- **Debug API**: `GET /api/debug/memory/*` endpoints show real-time system status
- **Configuration**: Check `memory_config.yaml` for behavior controls
- **Logs**: Check Docker logs with `docker compose logs friend-backend`
- **Documentation**: Each doc file links to relevant code sections

---

This documentation structure ensures you understand both the **big picture** and **implementation details** in a logical progression!