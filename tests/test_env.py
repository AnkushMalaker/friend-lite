# Test Environment Configuration
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_URL = os.getenv('API_URL', 'http://localhost:8001')
API_BASE = f"{API_URL}/api"

# Admin user credentials (Robot Framework format)
ADMIN_USER = {
    "email": os.getenv('ADMIN_EMAIL', 'test-admin@example.com'),
    "password": os.getenv('ADMIN_PASSWORD', 'test-admin-password-123')
}

# Individual variables for Robot Framework
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'test-admin@example.com')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'test-admin-password-123')

TEST_USER = {
    "email": "test@example.com",
    "password": "test-password"
}

# Individual variables for Robot Framework
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "test-password"

# Test Data
SAMPLE_CONVERSATIONS = [
    {
        "id": "conv_001",
        "transcript": "This is a test conversation about AI development.",
        "created_at": "2025-01-15T10:00:00Z"
    },
    {
        "id": "conv_002",
        "transcript": "Another test conversation discussing machine learning.",
        "created_at": "2025-01-15T11:00:00Z"
    }
]

SAMPLE_MEMORIES = [
    {
        "text": "User prefers AI discussions in the morning",
        "importance": 0.8
    },
    {
        "text": "User is interested in machine learning applications",
        "importance": 0.7
    }
]

# API Endpoints
ENDPOINTS = {
    "health": "/health",
    "readiness": "/readiness",
    "auth": "/auth/jwt/login",
    "conversations": "/api/conversations",
    "memories": "/api/memories",
    "memory_search": "/api/memories/search",
    "users": "/api/users"
}

# Test Configuration
TEST_CONFIG = {
    "retry_count": 3,
    "retry_delay": 1,
    "default_timeout": 30
}