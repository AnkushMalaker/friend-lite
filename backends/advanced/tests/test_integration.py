#!/usr/bin/env python3
"""
End-to-end integration test for Friend-Lite backend with unified transcription support.

This test validates the complete audio processing pipeline using isolated test environment:
1. Service startup with docker-compose-test.yml (isolated ports and databases)
2. ASR service startup (if Parakeet provider selected)
3. Authentication with test credentials
4. Audio file upload
5. Transcription (Deepgram API or Parakeet ASR service)
6. Memory extraction (OpenAI)
7. Data storage verification

Run with:
  # Deepgram API transcription (default)
  source .env && export DEEPGRAM_API_KEY && export OPENAI_API_KEY && uv run pytest tests/test_integration.py::test_full_pipeline_integration -v -s

  # Parakeet ASR transcription (HTTP/WebSocket service)
  source .env && export OPENAI_API_KEY && TRANSCRIPTION_PROVIDER=parakeet uv run pytest tests/test_integration.py::test_full_pipeline_integration -v -s

Test Environment:
- Uses docker-compose-test.yml for service isolation
- Backend runs on port 8001 (vs dev 8000)
- MongoDB on port 27018 (vs dev 27017)
- Qdrant on ports 6335/6336 (vs dev 6333/6334)
- Parakeet ASR on port 8767 (parakeet provider)
- Test credentials configured via environment variables
- Provider selection via TRANSCRIPTION_PROVIDER environment variable
"""

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import openai
import pytest
import requests
from pymongo import MongoClient

# Configure logging with immediate output (no buffering)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)
# Ensure immediate output
logger.handlers[0].flush() if logger.handlers else None
from dotenv import load_dotenv

# Test Configuration Flags
# REBUILD=True: Force rebuild of containers (useful when code changes)
# FRESH_RUN=True: Start with fresh data and containers (default)
# CLEANUP_CONTAINERS=True: Stop and remove containers after test (default)
REBUILD = os.environ.get("REBUILD", "true").lower() == "true"
FRESH_RUN = os.environ.get("FRESH_RUN", "true").lower() == "true"
CLEANUP_CONTAINERS = os.environ.get("CLEANUP_CONTAINERS", "true").lower() == "true"

# Transcription Provider Configuration
# TRANSCRIPTION_PROVIDER: 'deepgram' (Deepgram API) or 'parakeet' (Parakeet ASR service)
TRANSCRIPTION_PROVIDER = os.environ.get("TRANSCRIPTION_PROVIDER", "deepgram")  # Default to deepgram
# Get Parakeet URL from environment, fallback to port 8080
PARAKEET_ASR_URL = os.environ.get("PARAKEET_ASR_URL", "http://host.docker.internal:8080")

# Test Environment Configuration
# Base configuration for both providers
TEST_ENV_VARS_BASE = {
    "AUTH_SECRET_KEY": "test-jwt-signing-key-for-integration-tests",
    "ADMIN_PASSWORD": "test-admin-password-123",
    "ADMIN_EMAIL": "test-admin@example.com",
    "LLM_PROVIDER": "openai",
    "OPENAI_MODEL": "gpt-4o-mini",  # Cheaper model for tests
    "MONGODB_URI": "mongodb://localhost:27018",  # Test port (database specified in backend)
    "QDRANT_BASE_URL": "localhost",
    "DISABLE_SPEAKER_RECOGNITION": "true",  # Prevent segment duplication in tests
}

# Deepgram provider configuration (API)
TEST_ENV_VARS_DEEPGRAM = {
    **TEST_ENV_VARS_BASE,
    "TRANSCRIPTION_PROVIDER": "deepgram",
    # Deepgram API key loaded from environment
}

# Parakeet provider configuration (HTTP/WebSocket ASR service)
TEST_ENV_VARS_PARAKEET = {
    **TEST_ENV_VARS_BASE,
    "TRANSCRIPTION_PROVIDER": "parakeet",
    "PARAKEET_ASR_URL": PARAKEET_ASR_URL,
}

# Select configuration based on provider
if TRANSCRIPTION_PROVIDER == "parakeet":
    TEST_ENV_VARS = TEST_ENV_VARS_PARAKEET
else:  # Default to deepgram
    TEST_ENV_VARS = TEST_ENV_VARS_DEEPGRAM

tests_dir = Path(__file__).parent

# Test constants
BACKEND_URL = "http://localhost:8001"  # Test backend port
TEST_AUDIO_PATH = tests_dir.parent.parent.parent / "extras/test-audios/DIY Experts Glass Blowing_16khz_mono_4min.wav"
TEST_AUDIO_PATH_OFFLINE = tests_dir / "assets" / "test_clip_10s.wav"  # Shorter clip for offline testing
MAX_STARTUP_WAIT = 60  # seconds
PROCESSING_TIMEOUT = 300  # seconds for audio processing (5 minutes)


# Path to expected transcript file
EXPECTED_TRANSCRIPT_PATH = tests_dir / "assets/test_transcript.txt"

# Path to expected memories file
EXPECTED_MEMORIES_PATH = tests_dir / "assets/expected_memories.json"


class IntegrationTestRunner:
    """Manages the integration test lifecycle."""
    
    def __init__(self):
        print(f"🔧 Initializing IntegrationTestRunner", flush=True)
        print(f"   FRESH_RUN={FRESH_RUN}, CLEANUP_CONTAINERS={CLEANUP_CONTAINERS}, REBUILD={REBUILD}", flush=True)
        print(f"   TRANSCRIPTION_PROVIDER={TRANSCRIPTION_PROVIDER}", flush=True)
        sys.stdout.flush()
        
        self.token: Optional[str] = None
        self.services_started = False
        self.services_started_by_test = False  # Track if WE started the services
        self.mongo_client: Optional[MongoClient] = None
        self.fresh_run = FRESH_RUN  # Use global configuration flag
        self.cleanup_containers = CLEANUP_CONTAINERS  # Use global cleanup flag
        self.rebuild = REBUILD  # Use global rebuild flag
        self.asr_services_started = False  # Track ASR services for parakeet provider
        self.provider = TRANSCRIPTION_PROVIDER  # Store provider type
        
    def load_expected_transcript(self) -> str:
        """Load the expected transcript from the test assets file."""
        try:
            # Use provider-specific expectations if available
            if self.provider == "parakeet":
                transcript_path = tests_dir / "assets/test_transcript_parakeet.txt"
                if not transcript_path.exists():
                    transcript_path = EXPECTED_TRANSCRIPT_PATH  # Fallback to default
            else:
                transcript_path = EXPECTED_TRANSCRIPT_PATH
            
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"⚠️ Expected transcript file not found: {transcript_path}")
            return ""
        except Exception as e:
            logger.warning(f"⚠️ Error loading expected transcript: {e}")
            return ""
    
    def load_expected_memories(self) -> list:
        """Load the expected memories from the test assets file."""
        try:
            # Use provider-specific expectations if available
            if self.provider == "parakeet":
                memories_path = tests_dir / "assets/expected_memories_parakeet.json"
                if not memories_path.exists():
                    memories_path = EXPECTED_MEMORIES_PATH  # Fallback to default
            else:
                memories_path = EXPECTED_MEMORIES_PATH
            
            with open(memories_path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                # Handle both formats: list or dict with 'memories' key
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'memories' in data:
                    return data['memories']
                else:
                    logger.warning(f"⚠️ Unexpected memories file format: {type(data)}")
                    return []
        except FileNotFoundError:
            logger.warning(f"⚠️ Expected memories file not found: {memories_path}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error loading expected memories: {e}")
            return []
    
    def cleanup_test_data(self):
        """Clean up test-specific data directories using lightweight Docker container."""
        if not self.fresh_run:
            logger.info("🗂️ Skipping test data cleanup (reusing existing data)")
            return
            
        logger.info("🗂️ Cleaning up test-specific data directories...")
        
        # Use lightweight Docker container to clean root-owned files
        try:
            result = subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{Path.cwd()}/data:/data",
                "alpine:latest",
                "sh", "-c", "rm -rf /data/test_*"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ Docker cleanup successful")
            else:
                logger.warning(f"Error during Docker cleanup: {result.stderr}")
                            
        except Exception as e:
            logger.warning(f"⚠️ Docker cleanup failed: {e}")
            logger.warning("💡 Ensure Docker is running and accessible")
                
        logger.info("✓ Test data cleanup complete")
        
    def start_asr_services(self):
        """Start ASR services for Parakeet transcription testing."""
        if self.provider != "parakeet":
            logger.info(f"🔄 Skipping ASR services ({self.provider} provider uses API)")
            return
            
        logger.info(f"🚀 Starting Parakeet ASR service...")
        
        try:
            asr_dir = Path(__file__).parent.parent.parent.parent / "extras/asr-services"
            
            # Stop any existing ASR services first
            subprocess.run(
                ["docker", "compose", "-f", "docker-compose-test.yml", "down"],
                cwd=asr_dir,
                capture_output=True
            )
            
            # Start Parakeet ASR service
            result = subprocess.run(
                ["docker", "compose", "-f", "docker-compose-test.yml", "up", "--build", "-d", "parakeet-asr-test"],
                cwd=asr_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for service startup
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to start Parakeet ASR service: {result.stderr}")
                raise RuntimeError(f"Parakeet ASR service failed to start: {result.stderr}")
                
            self.asr_services_started = True
            logger.info("✅ Parakeet ASR service started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Parakeet ASR service: {e}")
            raise
            
    def wait_for_asr_ready(self):
        """Wait for ASR services to be ready."""
        if self.provider != "parakeet":
            logger.info(f"🔄 Skipping ASR readiness check ({self.provider} provider uses API)")
            return
        
        # Cascade failure check - don't wait for ASR if backend services failed
        if not hasattr(self, 'services_started') or not self.services_started:
            raise RuntimeError("Backend services are not running - cannot start ASR services")
            
        logger.info("🔍 Waiting for Parakeet ASR service to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < MAX_STARTUP_WAIT:
            try:
                # Check container status directly instead of HTTP health check
                # This avoids the curl dependency issue in the container
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=asr-services-parakeet-asr-test-1", "--format", "{{.Status}}"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    status = result.stdout.strip()
                    logger.debug(f"Container status: {status}")
                    
                    # Early exit on unhealthy containers
                    if "(unhealthy)" in status:
                        raise RuntimeError(f"Parakeet ASR container is unhealthy: {status}")
                    if "Exited" in status or "Dead" in status:
                        raise RuntimeError(f"Parakeet ASR container failed: {status}")
                    
                    # Look for 'Up' status and ideally '(healthy)' status
                    if "Up" in status:
                        # If container is healthy, we can skip the HTTP check
                        if "(healthy)" in status:
                            logger.info("✓ Parakeet ASR container is healthy")
                            return
                        # Additional check: try to connect to the service
                        try:
                            import requests

                            # Use the same URL that the backend will use
                            response = requests.get(f"{PARAKEET_ASR_URL}/health", timeout=5)
                            if response.status_code == 200:
                                health_data = response.json()
                                if health_data.get("status") == "healthy":
                                    logger.info("✓ Parakeet ASR service is healthy and accessible")
                                    return
                                elif health_data.get("status") == "unhealthy":
                                    raise RuntimeError(f"Parakeet ASR service reports unhealthy: {health_data}")
                                else:
                                    logger.debug(f"Service responding but not ready: {health_data}")
                            elif response.status_code >= 500:
                                raise RuntimeError(f"Parakeet ASR service error: HTTP {response.status_code}")
                            elif response.status_code >= 400:
                                logger.warning(f"Parakeet ASR client error: HTTP {response.status_code}")
                            else:
                                logger.debug(f"Health check failed with status {response.status_code}")
                        except requests.exceptions.ConnectionError as e:
                            logger.debug(f"Connection failed, but container is up: {e}")
                        except Exception as e:
                            logger.debug(f"HTTP health check failed, but container is up: {e}")
                    else:
                        logger.debug(f"Container not ready yet: {status}")
                else:
                    logger.debug("Container not found or not running")
                    
            except Exception as e:
                logger.debug(f"Container status check failed: {e}")
                
            time.sleep(2)
            
        raise RuntimeError("Parakeet ASR service failed to become ready within timeout")
        
    def cleanup_asr_services(self):
        """Clean up ASR services."""
        if not self.asr_services_started:
            return
            
        if not self.fresh_run:
            logger.info("🔄 Skipping ASR services cleanup (reusing existing services)")
            return
            
        logger.info("🧹 Cleaning up ASR services...")
        
        try:
            asr_dir = Path(__file__).parent.parent.parent.parent / "extras/asr-services"
            subprocess.run(
                ["docker", "compose", "-f", "docker-compose-test.yml", "down"],
                cwd=asr_dir,
                capture_output=True
            )
            logger.info("✅ ASR services stopped")
        except Exception as e:
            logger.warning(f"Error stopping ASR services: {e}")
        
    def setup_environment(self):
        """Set up environment variables for testing."""
        logger.info("Setting up test environment variables...")
        
        # Set test environment variables directly from TEST_ENV_VARS
        logger.info("Setting test environment variables from TEST_ENV_VARS...")
        for key, value in TEST_ENV_VARS.items():
            os.environ.setdefault(key, value)
            logger.info(f"✓ {key} set")
        
        # Load API keys from .env file if not already in environment
        if not os.environ.get('DEEPGRAM_API_KEY') or not os.environ.get('OPENAI_API_KEY'):
            logger.info("Loading API keys from .env file...")
            try:
                # Try to load .env.test first (CI environment), then fall back to .env (local development)
                env_test_path = '.env.test'
                env_path = '.env'
                
                # Check if we're in the right directory (tests directory vs backend directory)
                if not os.path.exists(env_test_path) and os.path.exists('../.env.test'):
                    env_test_path = '../.env.test'
                if not os.path.exists(env_path) and os.path.exists('../.env'):
                    env_path = '../.env'
                
                if os.path.exists(env_test_path):
                    logger.info(f"Loading from {env_test_path}")
                    load_dotenv(env_test_path)
                elif os.path.exists(env_path):
                    logger.info(f"Loading from {env_path}")
                    load_dotenv(env_path)
                else:
                    logger.warning("No .env.test or .env file found")
            except ImportError:
                logger.warning("python-dotenv not available, relying on shell environment")
        
        # Debug: Log API key status (masked for security)
        logger.info("API key status:")
        for key in ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]:
            value = os.environ.get(key)
            if value:
                masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "***"
                logger.info(f"  ✓ {key}: {masked_value}")
            else:
                logger.warning(f"  ⚠️ {key}: NOT SET")
        
        # Log environment readiness based on provider type
        deepgram_key = os.environ.get('DEEPGRAM_API_KEY')
        openai_key = os.environ.get('OPENAI_API_KEY')
        
        # Validate based on transcription provider (streaming/batch architecture)
        if self.provider == "deepgram":
            # Deepgram provider validation (API-based)
            if deepgram_key and openai_key:
                logger.info("✓ All required keys for Deepgram transcription are available")
            else:
                logger.warning("⚠️ Some keys missing for Deepgram transcription - test may fail")
                if not deepgram_key:
                    logger.warning("  Missing DEEPGRAM_API_KEY (required for Deepgram transcription)")
                if not openai_key:
                    logger.warning("  Missing OPENAI_API_KEY (required for memory processing)")
        elif self.provider == "parakeet":
            # Parakeet provider validation (local ASR service)
            parakeet_url = os.environ.get('PARAKEET_ASR_URL')
            if parakeet_url and openai_key:
                logger.info("✓ All required configuration for Parakeet transcription is available")
                logger.info(f"  Using Parakeet ASR service at: {parakeet_url}")
            else:
                logger.warning("⚠️ Missing configuration for Parakeet transcription - test may fail")
                if not parakeet_url:
                    logger.warning("  Missing PARAKEET_ASR_URL (required for Parakeet ASR service)")
                if not openai_key:
                    logger.warning("  Missing OPENAI_API_KEY (required for memory processing)")
        else:
            # Unknown or auto-select provider - check what's available
            logger.info(f"Provider '{self.provider}' - checking available configuration...")
            if deepgram_key and openai_key:
                logger.info("✓ Deepgram configuration available")
            elif os.environ.get('PARAKEET_ASR_URL') and openai_key:
                logger.info("✓ Parakeet configuration available")
            else:
                logger.warning("⚠️ No valid transcription provider configuration found")
                if not openai_key:
                    logger.warning("  Missing OPENAI_API_KEY (required for memory processing)")
                
    def start_services(self):
        """Start all services using docker compose."""
        logger.info("🚀 Starting services with docker compose...")
        
        # Change to backend directory
        os.chdir(tests_dir.parent)
        
        # Clean up test data directories first (unless cached)
        self.cleanup_test_data()
        
        try:
            # Check if test services are already running
            check_result = subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "ps", "-q"], capture_output=True, text=True)
            running_services = check_result.stdout.strip().split('\n') if check_result.stdout.strip() else []
            
            if len(running_services) > 0 and not self.rebuild:
                logger.info(f"🔄 Found {len(running_services)} running test services")
                # Check if test backend is healthy (only skip if not rebuilding)
                try:
                    health_check = subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "ps", "friend-backend-test"], capture_output=True, text=True)
                    if "healthy" in health_check.stdout or "Up" in health_check.stdout:
                        logger.info("✅ Test services already running and healthy, skipping restart")
                        self.services_started = True
                        self.services_started_by_test = True  # We'll manage test services
                        return
                except:
                    pass
            elif self.rebuild:
                logger.info("🔨 Rebuild flag is True, will rebuild containers with latest code")
            
            logger.info("🔄 Need to start/restart test services...")
            
            # Handle container management based on rebuild and cached flags
            if self.rebuild:
                logger.info("🔨 Rebuild mode: stopping containers and rebuilding with latest code...")
                # Stop existing test services and remove volumes for fresh rebuild
                subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "down", "-v"], capture_output=True)
            elif not self.fresh_run:
                logger.info("🔄 Reuse mode: restarting existing containers...")
                subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "restart"], capture_output=True)
            else:
                logger.info("🔄 Fresh mode: stopping containers and removing volumes...")
                # Stop existing test services and remove volumes for fresh start
                subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "down", "-v"], capture_output=True)
            
            # Ensure memory_config.yaml exists by copying from template
            memory_config_path = "memory_config.yaml"
            memory_template_path = "memory_config.yaml.template"
            if not os.path.exists(memory_config_path) and os.path.exists(memory_template_path):
                logger.info(f"📋 Creating {memory_config_path} from template...")
                shutil.copy2(memory_template_path, memory_config_path)
            
            # Check if we're in CI environment
            is_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
            
            if is_ci:
                # In CI, use simpler build process
                logger.info("🤖 CI environment detected, using optimized build...")
                if self.rebuild:
                    # Force rebuild in CI when rebuild flag is set with BuildKit disabled
                    env = os.environ.copy()
                    env['DOCKER_BUILDKIT'] = '0'
                    logger.info("🔨 Running Docker build command...")
                    build_result = subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "build"], env=env)
                    if build_result.returncode != 0:
                        logger.error(f"❌ Build failed with exit code {build_result.returncode}")
                        raise RuntimeError("Docker compose build failed")
                cmd = ["docker", "compose", "-f", "docker-compose-test.yml", "up", "-d", "--no-build"]
            else:
                # Local development - use rebuild flag to determine build behavior
                if self.rebuild:
                    cmd = ["docker", "compose", "-f", "docker-compose-test.yml", "up", "--build", "-d"]
                    logger.info("🔨 Local rebuild: will rebuild containers with latest code")
                else:
                    cmd = ["docker", "compose", "-f", "docker-compose-test.yml", "up", "-d"]
                    logger.info("🚀 Local start: using existing container images")
            
            # Start test services with BuildKit disabled to avoid bake issues
            env = os.environ.copy()
            env['DOCKER_BUILDKIT'] = '0'
            logger.info(f"🚀 Running Docker compose command: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to start services with exit code {result.returncode}")
                
                # Check individual container logs for better error details
                logger.error("🔍 Checking individual container logs for details...")
                try:
                    container_logs_result = subprocess.run(
                        ["docker", "compose", "-f", "docker-compose-test.yml", "logs", "--tail=50"],
                        capture_output=True, text=True, timeout=15
                    )
                    if container_logs_result.stdout:
                        logger.error("📋 Container logs:")
                        logger.error(container_logs_result.stdout)
                    if container_logs_result.stderr:
                        logger.error("📋 Container logs stderr:")
                        logger.error(container_logs_result.stderr)
                except Exception as e:
                    logger.warning(f"Could not fetch container logs: {e}")
                
                # Check container status
                logger.error("🔍 Checking container status...")
                try:
                    status_result = subprocess.run(
                        ["docker", "compose", "-f", "docker-compose-test.yml", "ps"],
                        capture_output=True, text=True, timeout=10
                    )
                    if status_result.stdout:
                        logger.error("📋 Container status:")
                        logger.error(status_result.stdout)
                except Exception as e:
                    logger.warning(f"Could not fetch container status: {e}")
                
                # Fail fast - no retry attempts
                raise RuntimeError("Docker compose failed to start")
                
            self.services_started = True
            self.services_started_by_test = True  # Mark that we started the services
            logger.info("✅ Docker compose started successfully")
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            raise
            
    def wait_for_services(self):
        """Wait for all services to be ready with comprehensive health checks."""
        logger.info("🔍 Performing comprehensive service health validation...")
        
        start_time = time.time()
        services_status = {
            "backend": False,
            "mongo": False,
            "auth": False,
            "readiness": False
        }
        
        while time.time() - start_time < MAX_STARTUP_WAIT:
            try:
                # 1. Check backend basic health
                if not services_status["backend"]:
                    try:
                        health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
                        if health_response.status_code == 200:
                            logger.info("✓ Backend health check passed")
                            services_status["backend"] = True
                        elif health_response.status_code >= 500:
                            raise RuntimeError(f"Backend service error: HTTP {health_response.status_code}")
                        elif health_response.status_code >= 400:
                            logger.warning(f"Backend client error: HTTP {health_response.status_code}")
                    except requests.exceptions.RequestException:
                        pass
                
                # 2. Check MongoDB connection via backend health check
                if not services_status["mongo"] and services_status["backend"]:
                    try:
                        health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
                        if health_response.status_code == 200:
                            data = health_response.json()
                            mongo_health = data.get("services", {}).get("mongodb", {})
                            if mongo_health.get("healthy", False):
                                logger.info("✓ MongoDB connection validated via backend health check")
                                services_status["mongo"] = True
                    except Exception:
                        pass
                
                # 3. Check comprehensive readiness (includes Qdrant validation)
                if not services_status["readiness"] and services_status["backend"] and services_status["auth"]:
                    try:
                        readiness_response = requests.get(f"{BACKEND_URL}/readiness", timeout=5)
                        if readiness_response.status_code == 200:
                            data = readiness_response.json()
                            logger.info(f"📋 Readiness report: {json.dumps(data, indent=2)}")
                            
                            # Validate readiness data - backend validates Qdrant internally
                            if data.get("status") in ["healthy", "ready"]:
                                logger.info("✓ Backend reports all services ready (including Qdrant)")
                                services_status["readiness"] = True
                            elif data.get("status") == "unhealthy":
                                raise RuntimeError(f"Backend reports unhealthy status: {data}")
                            else:
                                logger.warning(f"⚠️ Backend readiness check not fully healthy: {data}")
                        elif readiness_response.status_code >= 500:
                            raise RuntimeError(f"Backend readiness error: HTTP {readiness_response.status_code}")
                        elif readiness_response.status_code >= 400:
                            logger.warning(f"Backend readiness client error: HTTP {readiness_response.status_code}")
                                
                    except requests.exceptions.RequestException as e:
                        logger.debug(f"Readiness endpoint not ready yet: {e}")
                
                # 4. Check authentication endpoint
                if not services_status["auth"] and services_status["backend"]:
                    try:
                        # Just check that the auth endpoint exists (will return error without credentials)
                        auth_response = requests.post(f"{BACKEND_URL}/auth/jwt/login", timeout=3)
                        # Expecting 422 (validation error) not connection error
                        if auth_response.status_code in [422, 400]:
                            logger.info("✓ Authentication endpoint accessible")
                            services_status["auth"] = True
                    except requests.exceptions.RequestException:
                        pass
                
                # 5. Final validation - all services ready
                if all(services_status.values()):
                    logger.info("🎉 All services validated and ready!")
                    return True
                
                # Log current status
                ready_services = [name for name, status in services_status.items() if status]
                pending_services = [name for name, status in services_status.items() if not status]
                
                elapsed = time.time() - start_time
                logger.info(f"⏳ Health check progress ({elapsed:.1f}s): ✓ {ready_services} | ⏳ {pending_services}")
                
            except Exception as e:
                logger.warning(f"⚠️ Health check error: {e}")
                
            time.sleep(3)
            
        # Final status report
        logger.error("❌ Service readiness timeout!")
        failed_services = []
        for service, status in services_status.items():
            status_emoji = "✓" if status else "❌"
            logger.error(f"  {status_emoji} {service}: {'Ready' if status else 'Not ready'}")
            if not status:
                failed_services.append(service)
        
        # Check for cascade failures - if backend failed, everything else will fail
        if not services_status["backend"]:
            logger.error("💥 CRITICAL: Backend service failed - all dependent services will fail")
            logger.error("   This indicates a fundamental infrastructure issue")
        elif not services_status["mongo"]:
            logger.error("💥 CRITICAL: MongoDB connection failed - memory and auth will not work")
        elif not services_status["readiness"]:
            logger.error("💥 WARNING: Readiness check failed - Qdrant or other dependencies may be down")
            
        raise TimeoutError(f"Services did not become ready in {MAX_STARTUP_WAIT}s. Failed services: {failed_services}")
        
    def authenticate(self):
        """Authenticate and get admin token."""
        logger.info("🔑 Authenticating as admin...")
        
        # Always use test credentials for test environment
        logger.info("Using test environment credentials")
        admin_email = TEST_ENV_VARS["ADMIN_EMAIL"]
        admin_password = TEST_ENV_VARS["ADMIN_PASSWORD"]
        
        logger.info(f"Authenticating with email: {admin_email}")
        
        auth_url = f"{BACKEND_URL}/auth/jwt/login"
        
        response = requests.post(
            auth_url,
            data={
                'username': admin_email,
                'password': admin_password
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        
        if response.status_code != 200:
            logger.error(f"Authentication failed with {admin_email}")
            logger.error(f"Response: {response.text}")
            raise RuntimeError(f"Authentication failed: {response.text}")
            
        data = response.json()
        self.token = data.get('access_token')
        
        if not self.token:
            raise RuntimeError("No access token received")
            
        logger.info("✓ Authentication successful")
        
    def upload_test_audio(self):
        """Upload test audio file and monitor processing."""
        # Use different audio file for offline provider
        audio_path = TEST_AUDIO_PATH_OFFLINE if self.provider == "offline" else TEST_AUDIO_PATH
        
        logger.info(f"📤 Uploading test audio: {audio_path.name}")
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Test audio file not found: {audio_path}")
            
        # Log audio file details
        file_size = audio_path.stat().st_size
        logger.info(f"📊 Audio file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Upload file
        with open(audio_path, 'rb') as f:
            files = {'files': (audio_path.name, f, 'audio/wav')}
            data = {'device_name': 'integration_test'}
            headers = {'Authorization': f'Bearer {self.token}'}
            
            logger.info("📤 Sending upload request...")
            response = requests.post(
                f"{BACKEND_URL}/api/process-audio-files",
                files=files,
                data=data,
                headers=headers,
                timeout=300
            )
            
        logger.info(f"📤 Upload response status: {response.status_code}")
        
        if response.status_code != 200:
            raise RuntimeError(f"Upload failed: {response.text}")
            
        result = response.json()
        logger.info(f"📤 Upload response: {json.dumps(result, indent=2)}")
        
        # Extract client_id from response
        client_id = None
        if result.get('conversations'):
            client_id = result['conversations'][0].get('client_id')
        elif result.get('processed_files'):
            client_id = result['processed_files'][0].get('client_id')
        elif result.get('files'):
            client_id = result['files'][0].get('client_id')
            
        if not client_id:
            raise RuntimeError("No client_id in upload response")
            
        logger.info(f"📤 Generated client_id: {client_id}")
        return client_id
        
    def verify_processing_results(self, client_id: str):
        """Verify that audio was processed correctly."""
        logger.info(f"🔍 Verifying processing results for client: {client_id}")
        
        # Use backend API instead of direct MongoDB connection
        
        # First, wait for processing to complete using processor status endpoint
        logger.info("🔍 Waiting for processing to complete...")
        start_time = time.time()
        processing_complete = False
        
        while time.time() - start_time < 60:  # Wait up to 60 seconds for processing
            try:
                # Check processor status for this client
                response = requests.get(
                    f"{BACKEND_URL}/api/processor/tasks/{client_id}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    stages = data.get("stages", {})
                    
                    # Check if transcription stage is complete
                    transcription_stage = stages.get("transcription", {})
                    if transcription_stage.get("completed", False):
                        logger.info(f"✅ Transcription processing completed for client_id: {client_id}")
                        processing_complete = True
                        break
                    
                    # Check for errors
                    if transcription_stage.get("error"):
                        logger.error(f"❌ Transcription error: {transcription_stage.get('error')}")
                        break
                    
                    # Show processing status
                    logger.info(f"📊 Processing status: {data.get('status', 'unknown')}")
                    for stage_name, stage_info in stages.items():
                        completed = stage_info.get("completed", False)
                        error = stage_info.get("error")
                        status = "✅" if completed else "❌" if error else "⏳"
                        logger.info(f"  {status} {stage_name}: {'completed' if completed else 'error' if error else 'processing'}")
                        
                else:
                    logger.warning(f"❌ Processor status API call failed with status: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"❌ Error calling processor status API: {e}")
                
            logger.info(f"⏳ Still waiting for processing... ({time.time() - start_time:.1f}s)")
            time.sleep(3)
        
        if not processing_complete:
            logger.error(f"❌ Processing did not complete within timeout for client_id: {client_id}")
            # Don't fail immediately, try to get conversation anyway
        
        # Now get the conversation via API
        logger.info("🔍 Retrieving conversation...")
        conversation = None
        
        try:
            # Get conversations via API
            response = requests.get(
                f"{BACKEND_URL}/api/conversations",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                conversations = data.get("conversations", {})
                
                # Look for our client_id in the conversations
                if client_id in conversations:
                    conversation_list = conversations[client_id]
                    if conversation_list:
                        conversation = conversation_list[0]  # Get the first (most recent) conversation
                        logger.info(f"✅ Found conversation for client_id: {client_id}")
                    else:
                        logger.warning(f"⚠️ Client ID found but no conversations in list")
                else:
                    # Debug: show available conversations
                    available_clients = list(conversations.keys())
                    logger.error(f"❌ Client ID {client_id} not found in conversations")
                    logger.error(f"📊 Available client_ids: {available_clients}")
                    
            else:
                logger.error(f"❌ Conversations API call failed with status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error calling conversations API: {e}")
            
        if not conversation:
            logger.error(f"❌ No conversation found for client_id: {client_id}")
            raise AssertionError(f"No conversation found for client_id: {client_id}")
            
        logger.info(f"✓ Conversation found: {conversation['audio_uuid']}")
        
        # Log conversation details
        logger.info("📋 Conversation details:")
        logger.info(f"  - Audio UUID: {conversation['audio_uuid']}")
        logger.info(f"  - Client ID: {conversation.get('client_id')}")
        logger.info(f"  - Audio Path: {conversation.get('audio_path', 'N/A')}")
        logger.info(f"  - Timestamp: {conversation.get('timestamp', 'N/A')}")
        
        # Verify transcription (stored as array in conversation)
        transcript_segments = conversation.get('transcript', [])
        logger.info(f"📝 Transcription details:")
        logger.info(f"  - Transcript segments: {len(transcript_segments)}")
        
        # Extract full transcription text from segments
        transcription = ""
        if transcript_segments:
            # Combine all transcript segments
            transcription = " ".join([segment.get('text', '') for segment in transcript_segments])
        
        logger.info(f"  - Length: {len(transcription)} characters")
        logger.info(f"  - Word count: {len(transcription.split()) if transcription else 0}")
        
        if transcription:
            # Show first 200 characters of transcription
            preview = transcription[:200] + "..." if len(transcription) > 200 else transcription
            logger.info(f"  - Preview: {preview}")
            
            # Load expected transcript for comparison
            expected_transcript = self.load_expected_transcript()
            logger.info(f"  - Expected transcript length: {len(expected_transcript)} characters")
            
            # Log first 200 characters for comparison
            logger.info(f"  - Actual start: {transcription[:200]}...")
            if expected_transcript:
                logger.info(f"  - Expected start: {expected_transcript[:200]}...")
            
            # Call OpenAI to verify transcript similarity
            if os.environ.get("OPENAI_API_KEY") and expected_transcript:
                similarity_result = self.check_transcript_similarity_simple(transcription, expected_transcript)
                logger.info(f"  - AI similarity assessment:")
                logger.info(f"    • Similar: {similarity_result.get('similar', 'unknown')}")
                logger.info(f"    • Reason: {similarity_result.get('reason', 'No reason provided')}")
                
                # Store result for validation
                self.transcript_similarity_result = similarity_result
            elif not expected_transcript:
                logger.warning("⚠️ No expected transcript available for comparison")
                self.transcript_similarity_result = None
        else:
            logger.error("❌ No transcription found")
            
        # Verify conversation has required fields
        assert conversation.get('transcript'), "Conversation missing transcript"
        assert len(conversation['transcript']) > 0, "Transcript array is empty"
        assert transcription.strip(), "Transcription text is empty"
        
        # Check for memory extraction (if LLM is configured)
        if os.environ.get("OPENAI_API_KEY"):
            logger.info("🧠 Checking for memory extraction...")
            
            # Check debug tracker for memory processing
            response = requests.get(
                f"{BACKEND_URL}/metrics",
                headers={'Authorization': f'Bearer {self.token}'}
            )
            
            if response.status_code == 200:
                metrics = response.json()
                logger.info(f"📊 System metrics: {json.dumps(metrics, indent=2)}")
                
        logger.info("✅ Processing verification complete")
        
        return conversation, transcription
    
    def validate_memory_extraction(self, client_id: str):
        """Validate that memory extraction worked correctly."""
        logger.info(f"🧠 Validating memory extraction for client: {client_id}")
        
        # Wait for memory processing to complete
        client_memories = self.wait_for_memory_processing(client_id)
        
        # Check if we're using OpenMemory MCP provider
        memory_provider = os.environ.get("MEMORY_PROVIDER", "friend_lite")
        
        if not client_memories:
            if memory_provider == "openmemory_mcp":
                # For OpenMemory MCP, check if there are any memories at all (deduplication is OK)
                all_memories = self.get_memories_from_api()
                if all_memories:
                    logger.info(f"✅ OpenMemory MCP: Found {len(all_memories)} existing memories (deduplication successful)")
                    client_memories = all_memories  # Use existing memories for validation
                else:
                    raise AssertionError("No memories found in OpenMemory MCP - memory processing failed")
            else:
                raise AssertionError("No memories were extracted - memory processing failed")
        
        logger.info(f"✅ Found {len(client_memories)} memories")
        
        # Load expected memories and compare
        expected_memories = self.load_expected_memories()
        if not expected_memories:
            logger.warning("⚠️ No expected memories available for comparison")
            return client_memories
        
        # Use OpenAI to check if memories are similar
        if os.environ.get("OPENAI_API_KEY"):
            memory_similarity = self.check_memory_similarity_simple(client_memories, expected_memories)
            logger.info(f"🧠 Memory similarity assessment:")
            logger.info(f"  • Similar: {memory_similarity.get('similar', 'unknown')}")
            logger.info(f"  • Reason: {memory_similarity.get('reason', 'No reason provided')}")
            
            # Store result for validation
            self.memory_similarity_result = memory_similarity
        else:
            logger.warning("⚠️ No OpenAI API key available for memory comparison")
            self.memory_similarity_result = None
        
        return client_memories
        
    def check_transcript_similarity_simple(self, actual_transcript: str, expected_transcript: str) -> dict:
        """Use OpenAI to check transcript similarity with simple boolean response."""
        try:
            
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            prompt = f"""
            Compare these two transcripts to determine if they represent the same audio content.
            
            EXPECTED TRANSCRIPT:
            "{expected_transcript}"
            
            ACTUAL TRANSCRIPT:
            "{actual_transcript}"
            
            **MARK AS SIMILAR if:**
            - Core content and topics match (e.g., glass blowing class, participants, activities)
            - Key facts and events are present in both (names, numbers, objects, actions)
            - Overall narrative flow is recognizable
            - At least 70% semantic overlap exists
            
            **ACCEPTABLE DIFFERENCES (still mark as similar):**
            - Minor word variations or ASR errors
            - Different punctuation or capitalization
            - Missing or extra filler words
            - Small sections missing or repeated
            - Slightly different word order
            - Speaker diarization differences
            
            **ONLY MARK AS DISSIMILAR if:**
            - Core content is fundamentally different
            - Major sections (>30%) are missing or wrong
            - It appears to be a different audio file entirely
            
            Respond in JSON format:
            {{
                "reason": "brief explanation (1-3 sentences)"
                "similar": true/false,
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            response_text = (response.choices[0].message.content or "").strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, return a basic result
                return {
                    "similar": False,
                    "reason": f"Could not parse response: {response_text}"
                }
            
        except Exception as e:
            logger.warning(f"⚠️ Could not check transcript similarity: {e}")
            return {
                "similar": False,
                "reason": f"API call failed: {str(e)}"
            }
    
    def check_memory_similarity_simple(self, actual_memories: list, expected_memories: list) -> dict:
        """Use OpenAI to check if extracted memories are similar to expected memories."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            # Extract just the memory text from actual memories
            actual_memory_texts = [mem.get('memory', '') for mem in actual_memories]
            
            prompt = f"""
            Compare these two lists of memories to determine if they represent content from the same audio source and indicate successful memory extraction.
            
            **KEY CRITERIA FOR SIMILARITY (Return "similar": true if ANY of these are met):**
            
            1. **Topic/Context Match**: Both lists should be about the same main activity/event (e.g., glass blowing class)
            2. **Core Facts Overlap**: At least 3-4 significant factual details should overlap (people, places, numbers, objects)  
            3. **Semantic Coverage**: The same general knowledge should be captured, even if from different perspectives
            
            **ACCEPTABLE DIFFERENCES (Do NOT mark as dissimilar for these):**
            - Different focus areas (one list more personal/emotional, other more technical/factual)
            - Different level of detail (one more granular, other more high-level) 
            - Different speakers/participants emphasized
            - Different organization or memory chunking
            - Emotional vs factual framing of the same events
            - Missing some details in either list (as long as core overlap exists)
            
            **MARK AS DISSIMILAR ONLY IF:**
            - The memories seem to be from completely different audio/conversations
            - No meaningful factual overlap (suggests wrong audio or major transcription failure)
            - Core subject matter is entirely different
            
            **EVALUATION APPROACH:**
            1. Identify overlapping factual elements (people, places, objects, numbers, activities)
            2. Count significant semantic overlaps 
            3. If 3+ substantial overlaps exist AND same general topic/context → mark as similar
            4. Focus on "are these from the same source" rather than "are these identical"
            
            EXPECTED MEMORIES:
            {expected_memories}
            
            EXTRACTED MEMORIES:
            {actual_memory_texts}
            
            Respond in JSON format with:
            {{
                "reasoning": "detailed analysis of overlapping elements and why they indicate same/different source",
                "reason": "brief explanation of the decision", 
                "similar": true/false
            }}
            """
            
            logger.info(f"Making GPT-5-mini API call for memory similarity...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            response_text = (response.choices[0].message.content or "").strip()
            logger.info(f"Memory similarity GPT-5-mini response: '{response_text}'")
            
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as json_err:
                # If JSON parsing fails, return a basic result
                logger.error(f"JSON parsing failed: {json_err}")
                logger.error(f"Response text that failed to parse: '{response_text}'")
                return {
                    "reason": f"Could not parse response: {response_text}",
                    "similar": False,
                }
            
        except Exception as e:
            logger.error(f"⚠️ Could not check memory similarity: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception details: {str(e)}")
            return {
                "similar": False,
                "reason": f"API call failed: {str(e)}"
            }
    
    def get_memories_from_api(self) -> list:
        """Fetch memories from the backend API."""
        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.get(f"{BACKEND_URL}/api/memories", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('memories', [])
            else:
                logger.error(f"Failed to fetch memories: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching memories: {e}")
            return []
    
    def wait_for_memory_processing(self, client_id: str, timeout: int = 120):
        """Wait for memory processing to complete using processor status API."""
        logger.info(f"⏳ Waiting for memory processing to complete for client: {client_id}")
        
        start_time = time.time()
        memory_processing_complete = False
        
        # First, wait for memory processing completion using processor status API
        while time.time() - start_time < timeout:
            try:
                # Check processor status for this client (same pattern as transcription)
                response = requests.get(
                    f"{BACKEND_URL}/api/processor/tasks/{client_id}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # DEBUG: Log full API response to see exactly what we're getting
                    logger.info(f"🔍 Full processor status API response: {data}")
                    
                    stages = data.get("stages", {})
                    
                    # Check if memory stage is complete
                    memory_stage = stages.get("memory", {})
                    logger.info(f"🧠 Memory stage data: {memory_stage}")
                    
                    if memory_stage.get("completed", False):
                        logger.info(f"✅ Memory processing completed for client_id: {client_id}")
                        memory_processing_complete = True
                        break
                    
                    # Check for errors
                    if memory_stage.get("error"):
                        logger.error(f"❌ Memory processing error: {memory_stage.get('error')}")
                        break
                    
                    # Show processing status for memory stage
                    logger.info(f"📊 Memory processing status: {data.get('status', 'unknown')}")
                    for stage_name, stage_info in stages.items():
                        if stage_name == "memory":  # Focus on memory stage
                            completed = stage_info.get("completed", False)
                            error = stage_info.get("error")
                            status = "✅" if completed else "❌" if error else "⏳"
                            logger.info(f"  {status} {stage_name}: {'completed' if completed else 'error' if error else 'processing'}")
                            # DEBUG: Show all fields in memory stage
                            logger.info(f"    All memory stage fields: {stage_info}")
                            
                else:
                    logger.warning(f"❌ Processor status API call failed with status: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"❌ Error calling processor status API: {e}")
                
            logger.info(f"⏳ Still waiting for memory processing... ({time.time() - start_time:.1f}s)")
            time.sleep(3)
        
        if not memory_processing_complete:
            logger.warning(f"⚠️ Memory processing did not complete within {timeout}s, trying to fetch existing memories anyway")
        
        # Now fetch the memories from the API
        memories = self.get_memories_from_api()
        
        # Filter by client_id for test isolation in fresh mode, or get all user memories in reuse mode
        if not self.fresh_run:
            # In reuse mode, get all user memories (API already filters by user_id)
            user_memories = memories
            if user_memories:
                logger.info(f"✅ Found {len(user_memories)} total user memories (reusing existing data)")
                return user_memories
        else:
            # In fresh mode, filter by client_id for test isolation since we cleaned all data
            client_memories = [mem for mem in memories if mem.get('metadata', {}).get('client_id') == client_id]
            if client_memories:
                logger.info(f"✅ Found {len(client_memories)} memories for client {client_id}")
                return client_memories
        
        logger.warning(f"⚠️ No memories found after processing")
        return []
    
    async def create_chat_session(self, title: str = "Integration Test Session", description: str = "Testing memory integration") -> Optional[str]:
        """Create a new chat session and return session ID."""
        logger.info(f"📝 Creating chat session: {title}")
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/chat/sessions",
                headers={"Authorization": f"Bearer {self.token}"},
                json={
                    "title": title,
                    "description": description
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                session_id = data.get("session_id")
                logger.info(f"✅ Chat session created: {session_id}")
                return session_id
            else:
                logger.error(f"❌ Chat session creation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating chat session: {e}")
            return None
    
    async def send_chat_message(self, session_id: str, message: str) -> dict:
        """Send a message to chat session and parse response."""
        logger.info(f"💬 Sending message: {message}")
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/chat/send",
                headers={"Authorization": f"Bearer {self.token}"},
                json={
                    "message": message,
                    "session_id": session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                # Parse SSE response
                full_response = ""
                memory_ids = []
                
                for line in response.text.split('\n'):
                    if line.startswith('data: '):
                        try:
                            event_data = json.loads(line[6:])
                            event_type = event_data.get("type")
                            
                            if event_type == "memory_context":
                                mem_ids = event_data.get("data", {}).get("memory_ids", [])
                                memory_ids.extend(mem_ids)
                            elif event_type == "content":
                                content = event_data.get("data", {}).get("content", "")
                                full_response += content
                            elif event_type == "done":
                                break
                        except json.JSONDecodeError:
                            pass
                
                logger.info(f"🤖 Response received ({len(full_response)} chars)")
                if memory_ids:
                    logger.info(f"📚 Memories used: {len(memory_ids)} memory IDs")
                
                return {
                    "response": full_response,
                    "memories_used": memory_ids,
                    "success": True
                }
            else:
                logger.error(f"❌ Chat message failed: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Error sending chat message: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_chat_conversation(self, session_id: str) -> bool:
        """Run a test conversation with memory integration."""
        logger.info("🎭 Starting chat conversation test...")
        
        # Test messages designed to trigger memory retrieval
        test_messages = [
            "Hello! I'm testing the chat system with memory integration.",
            "What do you know about glass blowing? Have I mentioned anything about it?",
        ]
        
        memories_used_total = []
        
        for i, message in enumerate(test_messages, 1):
            logger.info(f"📨 Message {i}/{len(test_messages)}")
            result = await self.send_chat_message(session_id, message)
            
            if not result.get("success"):
                logger.error(f"❌ Chat message {i} failed: {result.get('error')}")
                return False
            
            # Track memory usage
            memories_used = result.get("memories_used", [])
            memories_used_total.extend(memories_used)
            
            # Small delay between messages
            time.sleep(1)
        
        logger.info(f"✅ Chat conversation completed. Total memories used: {len(set(memories_used_total))}")
        return True
    
    async def extract_memories_from_chat(self, session_id: str) -> dict:
        """Extract memories from the chat session."""
        logger.info(f"🧠 Extracting memories from chat session: {session_id}")
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/chat/sessions/{session_id}/extract-memories",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ Memory extraction successful: {data.get('count', 0)} memories created")
                    return data
                else:
                    logger.warning(f"⚠️ Memory extraction completed but no memories: {data.get('message', 'Unknown')}")
                    return data
            else:
                logger.error(f"❌ Memory extraction failed: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Error extracting memories from chat: {e}")
            return {"success": False, "error": str(e)}
        
    def cleanup(self):
        """Clean up test resources based on cached and rebuild flags."""
        logger.info("Cleaning up...")
        
        if self.mongo_client:
            self.mongo_client.close()
            
        # Handle container cleanup based on cleanup_containers flag (rebuild flag doesn't affect cleanup)
        if self.cleanup_containers and self.services_started_by_test:
            logger.info("🔄 Cleanup mode: stopping test docker compose services...")
            subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "down", "-v"], capture_output=True)
            logger.info("✓ Test containers stopped and volumes removed")
        elif not self.cleanup_containers:
            logger.info("🗂️ No cleanup: leaving containers running for debugging")
            if self.rebuild:
                logger.info("   (containers were rebuilt with latest code during this test)")
        else:
            logger.info("🔄 Test services were already running, leaving them as-is")
        
        logger.info("✓ Cleanup complete")


@pytest.fixture
def test_runner():
    """Pytest fixture for test runner."""
    runner = IntegrationTestRunner()
    yield runner
    runner.cleanup()


@pytest.mark.integration
def test_full_pipeline_integration(test_runner):
    """Test the complete audio processing pipeline."""
    # Immediate output to confirm test is starting
    print("🚀 TEST STARTING - test_full_pipeline_integration", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        # Test timing tracking
        test_start_time = time.time()
        phase_times = {}
        
        # Immediate logging to debug environment
        print("=" * 80, flush=True)
        print("🚀 STARTING INTEGRATION TEST", flush=True)
        print("=" * 80, flush=True)
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        logger.info(f"CI environment: {os.environ.get('CI', 'NOT SET')}")
        logger.info(f"GITHUB_ACTIONS: {os.environ.get('GITHUB_ACTIONS', 'NOT SET')}")
        sys.stdout.flush()
        
        # Phase 1: Environment setup
        phase_start = time.time()
        logger.info("📋 Phase 1: Setting up test environment...")
        test_runner.setup_environment()
        phase_times['env_setup'] = time.time() - phase_start
        logger.info(f"✅ Environment setup completed in {phase_times['env_setup']:.2f}s")
        
        # Phase 2: Service startup  
        phase_start = time.time()
        logger.info("🐳 Phase 2: Starting services...")
        test_runner.start_services()
        phase_times['service_startup'] = time.time() - phase_start
        logger.info(f"✅ Service startup completed in {phase_times['service_startup']:.2f}s")
        
        # Phase 2b: ASR service startup (offline only)
        phase_start = time.time()
        logger.info(f"🎤 Phase 2b: Starting ASR services ({TRANSCRIPTION_PROVIDER} provider)...")
        test_runner.start_asr_services()
        phase_times['asr_startup'] = time.time() - phase_start
        logger.info(f"✅ ASR service startup completed in {phase_times['asr_startup']:.2f}s")
        
        # Phase 3: Wait for services
        phase_start = time.time()
        logger.info("⏳ Phase 3: Waiting for services to be ready...")
        test_runner.wait_for_services()
        phase_times['service_readiness'] = time.time() - phase_start
        logger.info(f"✅ Service readiness check completed in {phase_times['service_readiness']:.2f}s")
        
        # Phase 3b: Wait for ASR services (offline only)
        phase_start = time.time()
        logger.info("⏳ Phase 3b: Waiting for ASR services to be ready...")
        test_runner.wait_for_asr_ready()
        phase_times['asr_readiness'] = time.time() - phase_start
        logger.info(f"✅ ASR readiness check completed in {phase_times['asr_readiness']:.2f}s")
        
        # Phase 4: Authentication
        phase_start = time.time()
        logger.info("🔑 Phase 4: Authentication...")
        test_runner.authenticate()
        phase_times['authentication'] = time.time() - phase_start
        logger.info(f"✅ Authentication completed in {phase_times['authentication']:.2f}s")
        
        # Phase 5: Audio upload and processing
        phase_start = time.time()
        logger.info("📤 Phase 5: Audio upload...")
        client_id = test_runner.upload_test_audio()
        phase_times['audio_upload'] = time.time() - phase_start
        logger.info(f"✅ Audio upload completed in {phase_times['audio_upload']:.2f}s")
        
        # Phase 6: Transcription processing
        phase_start = time.time()
        logger.info("🎤 Phase 6: Transcription processing...")
        conversation, transcription = test_runner.verify_processing_results(client_id)
        phase_times['transcription_processing'] = time.time() - phase_start
        logger.info(f"✅ Transcription processing completed in {phase_times['transcription_processing']:.2f}s")
        
        # Phase 7: Memory extraction
        phase_start = time.time()
        logger.info("🧠 Phase 7: Memory extraction...")
        memories = test_runner.validate_memory_extraction(client_id)
        phase_times['memory_extraction'] = time.time() - phase_start
        logger.info(f"✅ Memory extraction completed in {phase_times['memory_extraction']:.2f}s")
        
        # Phase 8: Chat with Memory Integration
        # phase_start = time.time()
        # logger.info("💬 Phase 8: Chat with Memory Integration...")
        
        # # Create chat session
        # session_id = asyncio.run(test_runner.create_chat_session(
        #     title="Integration Test Chat",
        #     description="Testing chat functionality with memory retrieval"
        # ))
        # assert session_id is not None, "Failed to create chat session"
        
        # # Run chat conversation
        # chat_success = asyncio.run(test_runner.run_chat_conversation(session_id))
        # assert chat_success, "Chat conversation failed"
        
        # # Extract memories from chat session (optional - may create additional memories)
        # chat_memory_result = asyncio.run(test_runner.extract_memories_from_chat(session_id))
        
        # phase_times['chat_integration'] = time.time() - phase_start
        # logger.info(f"✅ Chat integration completed in {phase_times['chat_integration']:.2f}s")
        
        # Basic assertions
        assert conversation is not None
        assert len(conversation['transcript']) > 0
        assert transcription.strip()  # Ensure we have actual text content
        
        # Transcript similarity assertion
        if hasattr(test_runner, 'transcript_similarity_result') and test_runner.transcript_similarity_result:
            assert test_runner.transcript_similarity_result.get('similar') == True, f"Transcript not similar enough: {test_runner.transcript_similarity_result.get('reason')}"
        
        # Memory validation assertions
        assert memories is not None and len(memories) > 0, "No memories were extracted"
        
        # Memory similarity assertion
        if hasattr(test_runner, 'memory_similarity_result') and test_runner.memory_similarity_result:
            if test_runner.memory_similarity_result.get('similar') != True:
                # Log transcript for debugging before failing
                logger.error("=" * 80)
                logger.error("❌ MEMORY SIMILARITY CHECK FAILED - DEBUGGING INFO")
                logger.error("=" * 80)
                logger.error("📝 Generated Transcript:")
                logger.error("-" * 60)
                logger.error(transcription)
                logger.error("-" * 60)
                
                # Format detailed error with both memory sets
                expected_memories = test_runner.load_expected_memories()
                extracted_memories = [mem.get('memory', '') for mem in memories]
                
                error_msg = f"""
Memory similarity check failed:
Reason: {test_runner.memory_similarity_result.get('reason', 'No reason provided')}
Reasoning: {test_runner.memory_similarity_result.get('reasoning', 'No detailed reasoning provided')}

Expected memories ({len(expected_memories)}):
{chr(10).join(f"  {i+1}. {mem}" for i, mem in enumerate(expected_memories))}

Extracted memories ({len(extracted_memories)}):
{chr(10).join(f"  {i+1}. {mem}" for i, mem in enumerate(extracted_memories))}

Generated Transcript ({len(transcription)} chars):
{transcription[:500]}{'...' if len(transcription) > 500 else ''}
"""
                assert False, error_msg
        
        # Calculate total test time
        total_test_time = time.time() - test_start_time
        phase_times['total_test'] = total_test_time
        
        # Log success with detailed timing
        logger.info("=" * 80)
        logger.info("🎉 INTEGRATION TEST PASSED!")
        logger.info("=" * 80)
        logger.info(f"⏱️  TIMING BREAKDOWN:")
        logger.info(f"  📋 Environment Setup:      {phase_times['env_setup']:>6.2f}s")
        logger.info(f"  🐳 Service Startup:        {phase_times['service_startup']:>6.2f}s")
        logger.info(f"  ⏳ Service Readiness:      {phase_times['service_readiness']:>6.2f}s")
        logger.info(f"  🔑 Authentication:         {phase_times['authentication']:>6.2f}s")
        logger.info(f"  📤 Audio Upload:           {phase_times['audio_upload']:>6.2f}s")
        logger.info(f"  🎤 Transcription:          {phase_times['transcription_processing']:>6.2f}s")
        logger.info(f"  🧠 Memory Extraction:      {phase_times['memory_extraction']:>6.2f}s")
        # logger.info(f"  💬 Chat Integration:       {phase_times['chat_integration']:>6.2f}s")
        logger.info(f"  {'─' * 35}")
        logger.info(f"  🏁 TOTAL TEST TIME:        {total_test_time:>6.2f}s ({total_test_time/60:.1f}m)")
        logger.info("")
        logger.info(f"📊 Test Results:")
        logger.info(f"  ✅ Audio file processed successfully")
        logger.info(f"  ✅ Transcription generated: {len(transcription)} characters")
        logger.info(f"  ✅ Word count: {len(transcription.split())}")
        logger.info(f"  ✅ Audio UUID: {conversation.get('audio_uuid')}")
        logger.info(f"  ✅ Client ID: {conversation.get('client_id')}")
        logger.info(f"  ✅ Memories extracted: {len(memories)}")
        logger.info(f"  ✅ Transcript similarity: {getattr(test_runner, 'transcript_similarity_result', {}).get('similar', 'N/A')}")
        logger.info(f"  ✅ Memory similarity: {getattr(test_runner, 'memory_similarity_result', {}).get('similar', 'N/A')}")
        logger.info("")
        logger.info("📝 Full Transcription:")
        logger.info("-" * 60)
        logger.info(transcription)
        logger.info("-" * 60)
        logger.info("")
        logger.info("🧠 Extracted Memories:")
        logger.info("-" * 60)
        for i, memory in enumerate(memories[:10], 1):  # Show first 10 memories
            logger.info(f"{i}. {memory.get('memory', 'No content')}")
        if len(memories) > 10:
            logger.info(f"... and {len(memories) - 10} more memories")
        logger.info("-" * 60)
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise
    finally:
        # Cleanup ASR services
        test_runner.cleanup_asr_services()


if __name__ == "__main__":
    # Run the test directly
    pytest.main([__file__, "-v", "-s"])