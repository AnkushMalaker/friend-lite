# Friend-Lite WebUI Usage Guide

## Overview
The enhanced WebUI now includes comprehensive user management and memory viewing capabilities across three main tabs.

## Getting Started

### 1. Start the WebUI
```bash
cd backends/advanced-backend/webui
streamlit run streamlit_app.py
```

### 2. Set Environment Variables
The WebUI requires the following environment variables:
- `BACKEND_API_URL` - URL of the backend API (default: http://localhost:8000)
- `MONGODB_URI_STREAMLIT` - MongoDB connection string for Streamlit
- `QDRANT_HOST_STREAMLIT` - Qdrant host for Streamlit (optional override)
- `QDRANT_PORT_STREAMLIT` - Qdrant port for Streamlit (optional override)

## Features by Tab

### 📞 Conversations Tab
- **View Latest Conversations**: See recent audio transcriptions
- **Enhanced Display**: 
  - Shows timestamp in readable format
  - Distinguishes between User IDs (👤) and random Client IDs
  - Displays full transcription text
  - Includes audio playback when available
- **Refresh**: Update the conversation list in real-time

### 🧠 Memories Tab
- **User-Specific Memory Viewing**: Enter a User ID to view memories for that specific user
- **Real-time Updates**: Refresh memories for any user
- **Smart Display**: Shows memory content, creation date, and memory IDs
- **Cross-tab Integration**: Can be pre-populated from User Management tab

### 👥 User Management Tab

#### Create Users
- **Manual Creation**: Add new users with custom User IDs
- **Validation**: Ensures User IDs are not empty
- **Success Feedback**: Confirms user creation

#### View & Manage Users
- **User List**: See all users in the system with their IDs
- **User Count**: Total number of registered users
- **Delete Users**: Remove users from the system (one-click deletion)
- **Auto-refresh**: Update user list after changes

#### Quick Actions
- **View User Memories**: Quick jump to memories for specific users
- **Integration**: Seamlessly connects with the Memories tab

### 💡 Tips & Best Practices
- **User ID Format**: Use clear identifiers like usernames or email prefixes
- **Auto-creation**: Users are automatically created when they connect with audio
- **Data Persistence**: Deleting users doesn't delete their conversations or memories
- **Memory Viewing**: Always specify a User ID in the Memories tab for best results

## Workflow Examples

### 1. Monitor New User Activity
1. Go to **Conversations** tab to see latest audio activity
2. Note User IDs (👤) vs random Client IDs
3. Switch to **User Management** to see registered users
4. Use **Memories** tab to view specific user memories

### 2. Manage Users
1. Go to **User Management** tab
2. Create new users manually if needed
3. View existing users and their details
4. Delete users that are no longer needed
5. Use quick actions to view user memories

### 3. Analyze User Memories
1. Go to **User Management** tab
2. Select a user and click "View Memories"
3. Switch to **Memories** tab (pre-populated with User ID)
4. Browse through user's discovered memories
5. Refresh for real-time updates

## API Integration
The WebUI communicates with the backend through these endpoints:
- `GET /api/conversations` - Fetch conversations
- `GET /api/users` - Get all users
- `POST /api/create_user` - Create new user
- `DELETE /api/delete_user` - Delete user
- `GET /api/memories?user_id=<id>` - Get user memories

## Troubleshooting
- **Connection Issues**: Ensure backend is running and BACKEND_API_URL is correct
- **No Data**: Check MongoDB and backend connections
- **Memory Issues**: Verify Qdrant configuration and mem0 setup
- **User Creation Fails**: Check if user already exists or backend is accessible 