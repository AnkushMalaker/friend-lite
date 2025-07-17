#!/usr/bin/env python3
"""
End-to-end integration test for Friend-Lite backend.

This test validates the complete audio processing pipeline using isolated test environment:
1. Service startup with docker-compose-test.yml (isolated ports and databases)
2. Authentication with test credentials
3. Audio file upload
4. Transcription (Deepgram)
5. Memory extraction (OpenAI)
6. Data storage verification

Run with: uv run pytest test_integration.py

Test Environment:
- Uses docker-compose-test.yml for service isolation
- Backend runs on port 8001 (vs dev 8000)
- MongoDB on port 27018 (vs dev 27017)
- Qdrant on ports 6335/6336 (vs dev 6333/6334)
- Pre-configured test credentials in .env.test
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest
import requests
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

# Test Configuration Flags
# Set CACHED_MODE=True for fast development iteration (skip cleanup, reuse containers)
# Set CACHED_MODE=False for fresh test environment (default, recommended)
CACHED_MODE = False  # Default to fresh mode for reliable testing

# Test Environment Configuration
TEST_ENV_VARS = {
    "AUTH_SECRET_KEY": "test-jwt-signing-key-for-integration-tests",
    "ADMIN_PASSWORD": "test-admin-password-123",
    "ADMIN_EMAIL": "test-admin@example.com",
    "LLM_PROVIDER": "openai",
    "OPENAI_MODEL": "gpt-4o-mini",  # Cheaper model for tests
    "MONGODB_URI": "mongodb://localhost:27018/test_db",  # Test port
    "QDRANT_BASE_URL": "localhost",
}

tests_dir = Path(__file__).parent

# Test constants
BACKEND_URL = "http://localhost:8001"  # Test backend port
TEST_AUDIO_PATH = tests_dir.parent.parent.parent / "extras/test-audios/DIY Experts Glass Blowing_16khz_mono_4min.wav"
MAX_STARTUP_WAIT = 120  # seconds
PROCESSING_TIMEOUT = 300  # seconds for audio processing (5 minutes)


# Path to expected transcript file
EXPECTED_TRANSCRIPT_PATH = tests_dir / "assets/test_transcript.txt"

# Path to expected memories file
EXPECTED_MEMORIES_PATH = tests_dir / "assets/expected_memories.json"


class IntegrationTestRunner:
    """Manages the integration test lifecycle."""
    
    def __init__(self):
        self.token: Optional[str] = None
        self.services_started = False
        self.services_started_by_test = False  # Track if WE started the services
        self.mongo_client: Optional[MongoClient] = None
        self.cached = CACHED_MODE  # Use global configuration flag
        
    def load_expected_transcript(self) -> str:
        """Load the expected transcript from the test assets file."""
        try:
            with open(EXPECTED_TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"⚠️ Expected transcript file not found: {EXPECTED_TRANSCRIPT_PATH}")
            return ""
        except Exception as e:
            logger.warning(f"⚠️ Error loading expected transcript: {e}")
            return ""
    
    def load_expected_memories(self) -> list:
        """Load the expected memories from the test assets file."""
        try:
            with open(EXPECTED_MEMORIES_PATH, 'r', encoding='utf-8') as f:
                import json
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️ Expected memories file not found: {EXPECTED_MEMORIES_PATH}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error loading expected memories: {e}")
            return []
    
    def cleanup_test_data(self):
        """Clean up test-specific data directories (preserve development data)."""
        if self.cached:
            logger.info("🗂️ Skipping test data cleanup (--cached mode)")
            return
            
        logger.info("🗂️ Cleaning up test-specific data directories...")
        
        # Test data directories to clean
        test_directories = [
            "./test_audio_chunks/",
            "./test_data/", 
            "./test_debug_dir/",
            "./mongo_data_test/",
            "./qdrant_data_test/",
            "./test_neo4j/"
        ]
        
        for test_dir in test_directories:
            test_path = Path(test_dir)
            if test_path.exists():
                try:
                    import shutil
                    shutil.rmtree(test_path)
                    logger.info(f"✓ Cleaned {test_dir}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean {test_dir}: {e}")
            else:
                logger.debug(f"📁 {test_dir} does not exist, skipping")
                
        logger.info("✓ Test data cleanup complete")
        
    def setup_environment(self):
        """Set up environment variables for CI testing."""
        logger.info("Setting up CI environment variables...")
        
        # Debug: Check current working directory and files
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        # Load .env.test file to get CI credentials
        try:
            # Try to load .env.test first (CI environment), then fall back to .env (local development)
            env_test_path = '.env.test'
            env_path = '.env'
            
            # Check if we're in the right directory (tests directory vs backend directory)
            if not os.path.exists(env_test_path) and os.path.exists('../.env.test'):
                env_test_path = '../.env.test'
                logger.info("Found .env.test in parent directory")
            if not os.path.exists(env_path) and os.path.exists('../.env'):
                env_path = '../.env'
                logger.info("Found .env in parent directory")
            
            if os.path.exists(env_test_path):
                logger.info(f"Found .env.test file at {env_test_path}, loading it...")
                load_dotenv(env_test_path)
                logger.info("✓ Loaded .env.test file (CI environment)")
            elif os.path.exists(env_path):
                logger.info(f"Found .env file at {env_path}, loading it...")
                load_dotenv(env_path)
                logger.info("✓ Loaded .env file (local development)")
            else:
                logger.warning("⚠️ No .env.test or .env file found in current or parent directory")
        except ImportError:
            logger.warning("⚠️ python-dotenv not available, relying on shell environment")
            # why would this ever happen?
        
        # Debug: Log all environment variables (masked for security)
        logger.info("Environment variables after loading:")
        for key in ["DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CI", "GITHUB_ACTIONS"]:
            value = os.environ.get(key)
            if value:
                masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "***"
                logger.info(f"  {key}: {masked_value}")
            else:
                logger.info(f"  {key}: NOT SET")
        
        # Get API keys from environment (GitHub secrets)
        required_keys = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
        missing_keys = [key for key in required_keys if not os.environ.get(key)]
        
        if missing_keys:
            logger.error(f"❌ Missing environment variables: {missing_keys}")
            logger.error(f"Available environment variables: {list(os.environ.keys())}")
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing_keys)}. These should be set as GitHub secrets in the CI environment.")
        
        # Pass through API keys
        for key in required_keys:
            if key in os.environ:
                logger.info(f"✓ {key} is set")
                
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
            
            if len(running_services) > 0:
                logger.info(f"🔄 Found {len(running_services)} running test services")
                # Check if test backend is healthy
                try:
                    health_check = subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "ps", "friend-backend-test"], capture_output=True, text=True)
                    if "healthy" in health_check.stdout or "Up" in health_check.stdout:
                        logger.info("✅ Test services already running and healthy, skipping restart")
                        self.services_started = True
                        self.services_started_by_test = True  # We'll manage test services
                        return
                except:
                    pass
            
            logger.info("🔄 Need to start/restart test services...")
            
            # Handle container management based on cached flag
            if self.cached:
                logger.info("🔄 Cached mode: restarting existing containers...")
                subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "restart"], capture_output=True)
            else:
                logger.info("🔄 Fresh mode: stopping containers and removing volumes...")
                # Stop existing test services and remove volumes for fresh start
                subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "down", "-v"], capture_output=True)
            
            # Check if we're in CI environment
            is_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
            
            if is_ci:
                # In CI, use simpler build process
                logger.info("🤖 CI environment detected, using optimized build...")
                cmd = ["docker", "compose", "-f", "docker-compose-test.yml", "up", "-d", "--no-build"]
                # Build first, then start
                build_result = subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "build"], capture_output=True, text=True)
                if build_result.returncode != 0:
                    logger.error(f"Build failed: {build_result.stderr}")
                    raise RuntimeError("Docker compose build failed")
            else:
                # Local development
                cmd = ["docker", "compose", "-f", "docker-compose-test.yml", "up", "--build", "-d"]
            
            # Start test services
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to start services: {result.stderr}")
                logger.error(f"Command output: {result.stdout}")
                
                # Try alternative approach for macOS
                if "permission denied" in result.stderr.lower():
                    logger.info("Permission issue detected, trying alternative approach...")
                    alt_result = subprocess.run(
                        ["docker", "compose", "-f", "docker-compose-test.yml", "up", "-d", "--no-build"],
                        capture_output=True,
                        text=True
                    )
                    if alt_result.returncode == 0:
                        logger.info("Alternative approach successful")
                        result = alt_result
                    else:
                        logger.error("Alternative approach also failed")
                        raise RuntimeError("Docker compose failed to start - try running:\n" +
                                         "  sudo chown -R $(whoami):staff \"$HOME/.docker/buildx\"\n" +
                                         "  sudo chmod -R 755 \"$HOME/.docker/buildx\"")
                else:
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
                    except requests.exceptions.RequestException:
                        pass
                
                # 2. Check MongoDB connection
                if not services_status["mongo"] and services_status["backend"]:
                    try:
                        # Try to connect to test MongoDB
                        from pymongo import MongoClient
                        test_client = MongoClient(TEST_ENV_VARS["MONGODB_URI"], serverSelectionTimeoutMS=3000)
                        test_client.admin.command('ping')
                        test_client.close()
                        logger.info("✓ MongoDB connection validated")
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
                            else:
                                logger.warning(f"⚠️ Backend readiness check not fully healthy: {data}")
                                
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
        for service, status in services_status.items():
            status_emoji = "✓" if status else "❌"
            logger.error(f"  {status_emoji} {service}: {'Ready' if status else 'Not ready'}")
            
        raise TimeoutError(f"Services did not become ready in {MAX_STARTUP_WAIT}s. Failed services: {[name for name, status in services_status.items() if not status]}")
        
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
        logger.info(f"📤 Uploading test audio: {TEST_AUDIO_PATH.name}")
        
        if not TEST_AUDIO_PATH.exists():
            raise FileNotFoundError(f"Test audio file not found: {TEST_AUDIO_PATH}")
            
        # Log audio file details
        file_size = TEST_AUDIO_PATH.stat().st_size
        logger.info(f"📊 Audio file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Upload file
        with open(TEST_AUDIO_PATH, 'rb') as f:
            files = {'files': (TEST_AUDIO_PATH.name, f, 'audio/wav')}
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
            
        if not client_id:
            raise RuntimeError("No client_id in upload response")
            
        logger.info(f"📤 Generated client_id: {client_id}")
        return client_id
        
    def verify_processing_results(self, client_id: str):
        """Verify that audio was processed correctly."""
        logger.info(f"🔍 Verifying processing results for client: {client_id}")
        
        # Connect to test MongoDB
        self.mongo_client = MongoClient(TEST_ENV_VARS["MONGODB_URI"])
        db = self.mongo_client.test_db
        
        # Check audio_chunks collection (where conversations are actually stored)
        start_time = time.time()
        conversation = None
        
        logger.info("🔍 Waiting for conversation to be created...")
        while time.time() - start_time < 30:  # Wait up to 30 seconds
            conversation = db.audio_chunks.find_one({"client_id": client_id})
            if conversation:
                break
            logger.info(f"⏳ Still waiting... ({time.time() - start_time:.1f}s)")
            time.sleep(2)
            
        if not conversation:
            # Show what conversations exist
            all_conversations = list(db.audio_chunks.find({}, {"client_id": 1, "audio_uuid": 1}))
            logger.error(f"❌ No conversation found for client_id: {client_id}")
            logger.error(f"Available conversations: {all_conversations}")
            raise AssertionError(f"No conversation found for client_id: {client_id}")
            
        logger.info(f"✓ Conversation found: {conversation['_id']}")
        
        # Log conversation details
        logger.info("📋 Conversation details:")
        logger.info(f"  - ID: {conversation['_id']}")
        logger.info(f"  - Client ID: {conversation.get('client_id')}")
        logger.info(f"  - Created: {conversation.get('created_at', 'N/A')}")
        logger.info(f"  - User ID: {conversation.get('user_id', 'N/A')}")
        
        # Verify transcription (stored as array in audio_chunks collection)
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
        
        if not client_memories:
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
            import openai
            
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            prompt = f"""
            Are these two transcripts similar enough that they represent the same content? 
            Compare the actual transcript against the expected transcript.
            
            EXPECTED TRANSCRIPT:
            "{expected_transcript}"
            
            ACTUAL TRANSCRIPT:
            "{actual_transcript}"
            
            Respond in JSON format with:
            {{
                "similar": true/false,
                "reason": "brief explanation"
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            response_text = (response.choices[0].message.content or "").strip()
            
            # Try to parse JSON response
            try:
                import json
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
            Compare these two lists of memories to see if they represent similar content.
            The actual memories should cover the same key points as the expected memories.
            
            IGNORE differences in:
            - Client IDs, user IDs, timestamps, and other metadata
            - Exact wording (focus on semantic content)
            - Order of memories
            
            Focus on whether the core content and key information is preserved.
            
            EXPECTED MEMORIES:
            {expected_memories}
            
            ACTUAL MEMORIES:
            {actual_memory_texts}
            
            Respond in JSON format with:
            {{
                "reasoning": "detailed analysis of what matches/differs and why",
                "reason": "brief explanation of the decision",
                "similar": true/false
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            response_text = (response.choices[0].message.content or "").strip()
            
            # Try to parse JSON response
            try:
                import json
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, return a basic result
                return {
                    "similar": False,
                    "reason": f"Could not parse response: {response_text}"
                }
            
        except Exception as e:
            logger.warning(f"⚠️ Could not check memory similarity: {e}")
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
        """Wait for memory processing to complete."""
        logger.info(f"⏳ Waiting for memory processing to complete for client: {client_id}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            memories = self.get_memories_from_api()
            
            # Filter by client_id for test isolation in fresh mode, or get all user memories in cached mode
            if self.cached:
                # In cached mode, get all user memories (API already filters by user_id)
                user_memories = memories
                if user_memories:
                    logger.info(f"✅ Found {len(user_memories)} total user memories (cached mode)")
                    return user_memories
            else:
                # In fresh mode, filter by client_id for test isolation since we cleaned all data
                client_memories = [mem for mem in memories if mem.get('metadata', {}).get('client_id') == client_id]
                if client_memories:
                    logger.info(f"✅ Found {len(client_memories)} memories for client {client_id}")
                    return client_memories
            
            logger.info(f"⏳ Still waiting for memories... ({time.time() - start_time:.1f}s)")
            time.sleep(2)
        
        logger.warning(f"⚠️ Memory processing timeout after {timeout}s")
        return []
        
    def cleanup(self):
        """Clean up test resources based on cached flag."""
        logger.info("Cleaning up...")
        
        if self.mongo_client:
            self.mongo_client.close()
            
        # Handle container cleanup based on cached flag
        if not self.cached and self.services_started_by_test:
            logger.info("🔄 Fresh mode: stopping test docker compose services...")
            subprocess.run(["docker", "compose", "-f", "docker-compose-test.yml", "down", "-v"], capture_output=True)
            logger.info("✓ Test containers stopped and volumes removed")
        elif self.cached:
            logger.info("🗂️ Cached mode: leaving containers running for reuse")
        else:
            logger.info("🔄 Test services were already running, leaving them as-is")
        
        logger.info("✓ Cleanup complete")


@pytest.fixture
def test_runner():
    """Pytest fixture for test runner."""
    runner = IntegrationTestRunner()
    yield runner
    runner.cleanup()


def test_full_pipeline_integration(test_runner):
    """Test the complete audio processing pipeline."""
    try:
        # Immediate logging to debug environment
        logger.info("=" * 80)
        logger.info("🚀 STARTING INTEGRATION TEST")
        logger.info("=" * 80)
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        logger.info(f"CI environment: {os.environ.get('CI', 'NOT SET')}")
        logger.info(f"GITHUB_ACTIONS: {os.environ.get('GITHUB_ACTIONS', 'NOT SET')}")
        
        # Setup
        test_runner.setup_environment()
        test_runner.start_services()
        test_runner.wait_for_services()
        test_runner.authenticate()
        
        # Test audio processing
        client_id = test_runner.upload_test_audio()
        conversation, transcription = test_runner.verify_processing_results(client_id)
        
        # Validate memory extraction
        memories = test_runner.validate_memory_extraction(client_id)
        
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
"""
                assert False, error_msg
        
        # Log success  
        logger.info("=" * 80)
        logger.info("🎉 INTEGRATION TEST PASSED!")
        logger.info("=" * 80)
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


if __name__ == "__main__":
    # Run the test directly
    pytest.main([__file__, "-v", "-s"])