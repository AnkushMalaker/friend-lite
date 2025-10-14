# Test Environment Configuration
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env.test from the tests directory
test_env_path = Path(__file__).parent / ".env.test"
load_dotenv(test_env_path)

# API Configuration
API_URL = 'http://localhost:8001'  # Use BACKEND_URL from test.env
API_BASE = 'http://localhost:8001/api'

WEB_URL = os.getenv('FRONTEND_URL', 'http://localhost:3001')  # Use FRONTEND_URL from test.env
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

# API Keys (loaded from test.env)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

# Test Configuration
TEST_CONFIG = {
    "retry_count": 3,
    "retry_delay": 1,
    "default_timeout": 30
}