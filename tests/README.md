# Friend-Lite API Tests

Comprehensive Robot Framework test suite for the Friend-Lite advanced backend API endpoints.

## Test Structure

### Test Files
- **`auth_tests.robot`** - Authentication and user management tests
- **`memory_tests.robot`** - Memory management and search tests
- **`conversation_tests.robot`** - Conversation management and versioning tests
- **`health_tests.robot`** - Health check and status endpoint tests
- **`chat_tests.robot`** - Chat service and session management tests
- **`client_queue_tests.robot`** - Client management and queue monitoring tests
- **`system_admin_tests.robot`** - System administration and configuration tests
- **`all_api_tests.robot`** - Master test suite runner

### Resource Files
- **`resources/auth_keywords.robot`** - Authentication helper keywords
- **`resources/memory_keywords.robot`** - Memory management keywords
- **`resources/conversation_keywords.robot`** - Conversation management keywords
- **`resources/chat_keywords.robot`** - Chat service keywords
- **`resources/setup_resources.robot`** - Basic setup and health check keywords
- **`resources/login_resources.robot`** - Login-specific utilities

### Configuration
- **`test_env.py`** - Environment configuration and test data
- **`.env`** - Environment variables (create from template)

## Running Tests

### Prerequisites
1. Friend-Lite backend running at `http://localhost:8001` (or set `API_URL` in `.env`)
2. Admin user credentials configured in `.env`
3. Robot Framework and RequestsLibrary installed

### Environment Setup
```bash
# Copy environment template
cp .env.template .env

# Edit .env with your configuration
API_URL=http://localhost:8001
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=your-secure-admin-password
```

### Running Individual Test Suites
```bash
# Authentication and user tests
robot auth_tests.robot

# Memory management tests
robot memory_tests.robot

# Conversation management tests
robot conversation_tests.robot

# Health and status tests
robot health_tests.robot

# Chat service tests
robot chat_tests.robot

# Client and queue tests
robot client_queue_tests.robot

# System administration tests
robot system_admin_tests.robot
```

### Running All Tests
```bash
# Run complete test suite
robot *.robot

# Run with specific tags
robot --include auth *.robot
robot --include positive *.robot
robot --include admin *.robot
```

### Test Output
```bash
# Custom output directory
robot --outputdir results *.robot

# Verbose logging
robot --loglevel DEBUG *.robot

# Parallel execution
pabot --processes 4 *.robot
```

## Test Coverage

### Authentication & Users (`/api/users`, `/auth`)
- ✅ Login with valid/invalid credentials
- ✅ Get current user information
- ✅ Create/update/delete users (admin only)
- ✅ User authorization and access control
- ✅ Admin privilege enforcement

### Memory Management (`/api/memories`)
- ✅ Get user memories with pagination
- ✅ Search memories with similarity thresholds
- ✅ Get memories with transcripts
- ✅ Delete specific memories
- ✅ Admin memory access across users
- ✅ Unfiltered memory access for debugging

### Conversation Management (`/api/conversations`)
- ✅ List and retrieve conversations
- ✅ Conversation version history
- ✅ Transcript reprocessing
- ✅ Memory reprocessing with version selection
- ✅ Version activation (transcript/memory)
- ✅ Conversation deletion and cleanup
- ✅ User data isolation

### Health & Status (`/health`, `/readiness`)
- ✅ Main health check with service details
- ✅ Readiness check for orchestration
- ✅ Authentication service health
- ✅ Queue system health status
- ✅ Chat service health check
- ✅ System metrics (admin only)

### Chat Service (`/api/chat`)
- ✅ Session creation and management
- ✅ Session title updates
- ✅ Message retrieval
- ✅ Chat statistics
- ✅ Memory extraction from sessions
- ✅ Session deletion and cleanup

### Client & Queue Management
- ✅ Active client monitoring
- ✅ Queue job listing with pagination
- ✅ Queue statistics and health
- ✅ User job isolation
- ✅ Processing task monitoring (admin only)

### System Administration
- ✅ Authentication configuration
- ✅ Diarization settings management
- ✅ Speaker configuration
- ✅ Memory configuration (YAML)
- ✅ Configuration validation and reload
- ✅ Bulk memory deletion

## Test Categories

### By Access Level
- **Public**: Health checks, auth config
- **User**: Memories, conversations, chat sessions
- **Admin**: User management, system config, metrics

### By Test Type
- **Positive**: Valid operations and expected responses
- **Negative**: Invalid inputs, unauthorized access
- **Security**: Authentication, authorization, data isolation
- **Integration**: Cross-service functionality

### By Component
- **Auth**: Authentication and authorization
- **Memory**: Memory storage and retrieval
- **Conversation**: Audio processing and transcription
- **Chat**: Interactive chat functionality
- **System**: Configuration and administration

## Key Features Tested

### Security
- JWT token authentication
- Role-based access control (admin vs user)
- Data isolation between users
- Unauthorized access prevention

### Data Management
- CRUD operations for all entities
- Pagination and filtering
- Search functionality with thresholds
- Versioning and history tracking

### System Integration
- Service health monitoring
- Configuration management
- Queue system monitoring
- Cross-service communication

### Error Handling
- Invalid input validation
- Non-existent resource handling
- Permission denied scenarios
- Service unavailability graceful degradation

## Maintenance

### Adding New Tests
1. Create test file or add to existing suite
2. Use appropriate resource keywords
3. Follow naming conventions (`Test Name Test`)
4. Include proper tags and documentation
5. Add cleanup in teardown if needed

### Updating Keywords
1. Modify resource files for reusable functionality
2. Keep keywords focused and single-purpose
3. Use proper argument handling
4. Include documentation strings

### Environment Variables
Update `test_env.py` when adding new configuration options or test data.

## Troubleshooting

### Common Issues
- **401 Unauthorized**: Check admin credentials in `.env`
- **Connection Refused**: Ensure backend is running
- **Test Failures**: Check service health endpoints first
- **Timeout Errors**: Increase timeouts in test configuration

### Debug Mode
```bash
# Run with detailed logging
robot --loglevel TRACE auth_tests.robot

# Stop on first failure
robot --exitonfailure *.robot
```