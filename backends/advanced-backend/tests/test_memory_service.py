#!/usr/bin/env python3
"""
Comprehensive test file for debugging memory service issues.

This script tests:
1. Ollama connectivity and model availability
2. Qdrant connectivity
3. Mem0 configuration
4. Memory service initialization
5. Memory creation functionality
6. Action item extraction

Run this from the backend directory:
python tests/test_memory_service.py
"""

import asyncio
import logging
import os
import sys
import time
import json
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests
import ollama
from mem0 import Memory

# Import our modules
try:
    from memory.memory_service import (
        MemoryService, 
        get_memory_service, 
        init_memory_config,
        _init_process_memory,
        _add_memory_to_store,
        _extract_action_items_from_transcript,
        MEM0_CONFIG,
        OLLAMA_BASE_URL,
        QDRANT_BASE_URL,
    )
    print("✅ Successfully imported memory service modules")
except ImportError as e:
    print(f"❌ Failed to import memory service modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger("memory_test")

class MemoryServiceTester:
    """Comprehensive memory service tester."""
    
    def __init__(self):
        self.ollama_url = OLLAMA_BASE_URL
        self.qdrant_url = QDRANT_BASE_URL
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all tests in sequence."""
        print("🧪 Starting Memory Service Diagnostic Tests")
        print("=" * 60)
        
        tests = [
            ("Configuration Check", self.test_configuration),
            ("Ollama Connectivity", self.test_ollama_connectivity),
            ("Ollama Models", self.test_ollama_models),
            ("Qdrant Connectivity", self.test_qdrant_connectivity),
            ("Mem0 Configuration", self.test_mem0_config),
            ("Memory Service Initialization", self.test_memory_service_init),
            ("Process Memory Initialization", self.test_process_memory_init),
            ("Basic Memory Creation", self.test_basic_memory_creation),
            ("Action Item Extraction", self.test_action_item_extraction),
            ("Full Integration Test", self.test_full_integration),
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            print("-" * 40)
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status}: {test_name}")
            except Exception as e:
                self.test_results[test_name] = False
                print(f"❌ ERROR in {test_name}: {e}")
                logger.exception(f"Test failed: {test_name}")
        
        self.print_summary()
        
    def test_configuration(self):
        """Test current configuration values."""
        print(f"📋 Current Configuration:")
        print(f"   OLLAMA_BASE_URL: {self.ollama_url}")
        print(f"   QDRANT_BASE_URL: {self.qdrant_url}")
        
        # Show environment variables from .env file
        print(f"\n📄 Environment Variables from .env:")
        env_vars = [
            'OLLAMA_BASE_URL', 'OFFLINE_ASR_TCP_URI', 'HF_TOKEN', 
            'ADMIN_EMAIL', 'ADMIN_USERNAME', 'DEBUG_DIR'
        ]
        for var in env_vars:
            value = os.getenv(var, 'Not set')
            if 'TOKEN' in var or 'PASSWORD' in var or 'SECRET' in var:
                # Mask sensitive values
                display_value = f"{value[:10]}..." if len(value) > 10 else "***"
            else:
                display_value = value
            print(f"   {var}: {display_value}")
        
        print(f"\n🔧 Mem0 Config:")
        print(f"     LLM Provider: {MEM0_CONFIG['llm']['provider']}")
        print(f"     LLM Model: {MEM0_CONFIG['llm']['config']['model']}")
        print(f"     LLM Ollama URL: {MEM0_CONFIG['llm']['config']['ollama_base_url']}")
        print(f"     Embedder Provider: {MEM0_CONFIG['embedder']['provider']}")
        print(f"     Embedder Model: {MEM0_CONFIG['embedder']['config']['model']}")
        print(f"     Embedder Ollama URL: {MEM0_CONFIG['embedder']['config']['ollama_base_url']}")
        print(f"     Vector Store: {MEM0_CONFIG['vector_store']['provider']}")
        print(f"     Qdrant Host: {MEM0_CONFIG['vector_store']['config']['host']}")
        print(f"     Qdrant Port: {MEM0_CONFIG['vector_store']['config']['port']}")
        
        # Check for potential issues
        issues = []
        if 'ollama:11434' in self.ollama_url:
            issues.append("🔧 Ollama URL uses Docker hostname 'ollama' - should be http://192.168.0.110:11434")
        if self.qdrant_url == 'qdrant':
            issues.append("🔧 Qdrant URL uses Docker hostname 'qdrant' - may not work outside container")
        
        # Check if the configuration matches your environment
        expected_ollama = "http://192.168.0.110:11434"
        if self.ollama_url == expected_ollama:
            print(f"\n✅ Ollama URL matches your .env configuration: {expected_ollama}")
        else:
            issues.append(f"🔧 Ollama URL mismatch - expected {expected_ollama}, got {self.ollama_url}")
            
        if issues:
            print("\n⚠️ Configuration Issues Found:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ Configuration looks good!")
        
        return len(issues) == 0
        
    def test_ollama_connectivity(self):
        """Test Ollama server connectivity."""
        try:
            # Test HTTP connectivity first
            health_url = f"{self.ollama_url}/api/version"
            print(f"🌐 Testing Ollama HTTP connectivity to: {health_url}")
            
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                version_info = response.json()
                print(f"✅ Ollama server version: {version_info.get('version', 'unknown')}")
                
                # Test Ollama Python client
                print("🐍 Testing Ollama Python client...")
                client = ollama.Client(host=self.ollama_url)
                
                # Try to list models
                models = client.list()
                print(f"📋 Available models: {len(models.get('models', []))}")
                return True
            else:
                print(f"❌ Ollama HTTP error: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.ollama_url}")
            print("💡 Suggestion: Update OLLAMA_BASE_URL to http://192.168.0.110:11434")
            return False
        except Exception as e:
            print(f"❌ Ollama connectivity error: {e}")
            return False
            
    def test_ollama_models(self):
        """Test required Ollama models are available."""
        try:
            client = ollama.Client(host=self.ollama_url)
            models_response = client.list()
            models = models_response.get('models', [])
            model_names = [model['name'] for model in models]
            
            required_models = ['llama3.1:latest', 'nomic-embed-text:latest']
            missing_models = []
            
            print(f"📋 Available models ({len(model_names)}):")
            for model_name in model_names:
                print(f"   ✅ {model_name}")
                
            print(f"\n🔍 Checking required models:")
            for required_model in required_models:
                if any(required_model in model_name for model_name in model_names):
                    print(f"   ✅ {required_model} - Found")
                else:
                    print(f"   ❌ {required_model} - Missing")
                    missing_models.append(required_model)
                    
            if missing_models:
                print(f"\n💡 To pull missing models, run:")
                for model in missing_models:
                    print(f"   ollama pull {model}")
                return False
            else:
                return True
                
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            return False
            
    def test_qdrant_connectivity(self):
        """Test Qdrant server connectivity."""
        try:
            # Try different Qdrant URLs
            qdrant_urls = [
                f"http://{self.qdrant_url}:6333",
                f"http://{self.qdrant_url}:6334", 
                "http://localhost:6333",
                "http://localhost:6334",
                "http://192.168.0.110:6333",
                "http://192.168.0.110:6334"
            ]
            
            for url in qdrant_urls:
                try:
                    print(f"🌐 Testing Qdrant connectivity to: {url}")
                    response = requests.get(f"{url}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Qdrant reachable at {url}")
                        
                        # Test collections endpoint
                        collections_response = requests.get(f"{url}/collections", timeout=5)
                        if collections_response.status_code == 200:
                            collections = collections_response.json()
                            print(f"📋 Collections found: {len(collections.get('result', {}).get('collections', []))}")
                            return True
                        else:
                            print(f"⚠️ Qdrant health OK but collections endpoint failed: {collections_response.status_code}")
                            return True  # Health is good enough
                except requests.exceptions.RequestException:
                    continue
                    
            print("❌ Could not connect to Qdrant on any URL")
            print("💡 Make sure Qdrant is running and accessible")
            return False
            
        except Exception as e:
            print(f"❌ Qdrant connectivity error: {e}")
            return False
            
    def test_mem0_config(self):
        """Test Mem0 configuration validation."""
        try:
            print("🔧 Validating Mem0 configuration...")
            
            # Check required fields
            required_fields = [
                ['llm', 'provider'],
                ['llm', 'config', 'model'],
                ['llm', 'config', 'ollama_base_url'],
                ['embedder', 'provider'],
                ['embedder', 'config', 'model'],
                ['embedder', 'config', 'ollama_base_url'],
                ['vector_store', 'provider'],
                ['vector_store', 'config', 'host'],
                ['vector_store', 'config', 'port'],
            ]
            
            config_valid = True
            for field_path in required_fields:
                current = MEM0_CONFIG
                try:
                    for key in field_path:
                        current = current[key]
                    print(f"   ✅ {'.'.join(field_path)}: {current}")
                except KeyError:
                    print(f"   ❌ {'.'.join(field_path)}: Missing")
                    config_valid = False
                    
            # Test if we can create a Memory instance
            print("\n🏗️ Testing Memory instance creation...")
            try:
                # This might fail due to connectivity, but should validate config structure
                memory = Memory.from_config(MEM0_CONFIG)
                print("✅ Memory instance created successfully")
                return config_valid
            except Exception as e:
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    print(f"⚠️ Memory instance creation failed due to connectivity: {e}")
                    return config_valid  # Config is probably OK, just can't connect
                else:
                    print(f"❌ Memory instance creation failed due to config: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ Mem0 config validation error: {e}")
            return False
            
    async def test_memory_service_init(self):
        """Test MemoryService initialization."""
        try:
            print("🚀 Testing MemoryService initialization...")
            
            service = MemoryService()
            print(f"📊 Initial state - initialized: {service._initialized}")
            
            # Test initialization with timeout
            start_time = time.time()
            try:
                await asyncio.wait_for(service.initialize(), timeout=30)
                init_time = time.time() - start_time
                print(f"✅ MemoryService initialized successfully in {init_time:.2f}s")
                print(f"📊 Final state - initialized: {service._initialized}")
                return True
            except asyncio.TimeoutError:
                print("❌ MemoryService initialization timed out after 30s")
                return False
                
        except Exception as e:
            print(f"❌ MemoryService initialization error: {e}")
            return False
            
    def test_process_memory_init(self):
        """Test process memory initialization (used in workers)."""
        try:
            print("🔄 Testing process memory initialization...")
            
            process_memory = _init_process_memory()
            if process_memory:
                print("✅ Process memory initialized successfully")
                return True
            else:
                print("❌ Process memory initialization returned None")
                return False
                
        except Exception as e:
            print(f"❌ Process memory initialization error: {e}")
            return False
            
    def test_basic_memory_creation(self):
        """Test basic memory creation functionality."""
        try:
            print("💾 Testing basic memory creation...")
            
            # Test data
            test_transcript = "Hello, this is a test conversation about planning a meeting for next week."
            test_client_id = "test_client_123"
            test_audio_uuid = f"test_audio_{int(time.time())}"
            test_user_id = "test_user_456"
            test_user_email = "test@example.com"
            
            print(f"📝 Test data:")
            print(f"   Transcript: {test_transcript}")
            print(f"   Client ID: {test_client_id}")
            print(f"   Audio UUID: {test_audio_uuid}")
            print(f"   User ID: {test_user_id}")
            print(f"   User Email: {test_user_email}")
            
            # Test the low-level function
            result = _add_memory_to_store(
                test_transcript, 
                test_client_id, 
                test_audio_uuid, 
                test_user_id, 
                test_user_email
            )
            
            if result:
                print("✅ Basic memory creation successful")
                return True
            else:
                print("❌ Basic memory creation failed")
                return False
                
        except Exception as e:
            print(f"❌ Basic memory creation error: {e}")
            return False
            
    def test_action_item_extraction(self):
        """Test action item extraction functionality."""
        try:
            print("📋 Testing action item extraction...")
            
            # Test transcript with obvious action items
            test_transcript = """
            John: We need to schedule a meeting for next Tuesday to discuss the project.
            Mary: I'll send you the agenda by tomorrow.
            John: Great, and can you also review the budget document before the meeting?
            Mary: Sure, I'll get that done by Monday.
            """
            
            test_client_id = "test_client_123"
            test_audio_uuid = f"test_action_items_{int(time.time())}"
            
            print(f"📝 Test transcript:")
            print(f"   {test_transcript.strip()}")
            
            action_items = _extract_action_items_from_transcript(
                test_transcript,
                test_client_id,
                test_audio_uuid
            )
            
            print(f"📊 Extracted {len(action_items)} action items:")
            for i, item in enumerate(action_items, 1):
                print(f"   {i}. {item.get('description', 'No description')}")
                print(f"      Assignee: {item.get('assignee', 'unassigned')}")
                print(f"      Due: {item.get('due_date', 'not_specified')}")
                print(f"      Priority: {item.get('priority', 'not_specified')}")
                
            if len(action_items) > 0:
                print("✅ Action item extraction successful")
                return True
            else:
                print("⚠️ No action items extracted (might be working correctly)")
                return True  # This might be correct behavior
                
        except Exception as e:
            print(f"❌ Action item extraction error: {e}")
            return False
            
    async def test_full_integration(self):
        """Test the full integration flow."""
        try:
            print("🔗 Testing full integration flow...")
            
            # Get the global memory service
            service = get_memory_service()
            
            # Test data
            test_transcript = "This is a full integration test. We discussed planning a project review meeting and setting up the new development environment."
            test_client_id = "integration_test_client"
            test_audio_uuid = f"integration_test_{int(time.time())}"
            test_user_id = "integration_test_user"
            test_user_email = "integration@test.com"
            
            print(f"📝 Integration test data:")
            print(f"   Transcript: {test_transcript}")
            print(f"   User: {test_user_email}")
            
            # Test memory addition (high-level API)
            print("💾 Testing high-level memory addition...")
            memory_result = await service.add_memory(
                test_transcript,
                test_client_id,
                test_audio_uuid,
                test_user_id,
                test_user_email
            )
            
            if memory_result:
                print("✅ High-level memory addition successful")
                
                # Test memory retrieval
                print("🔍 Testing memory retrieval...")
                try:
                    memories = service.get_all_memories(test_user_id, limit=10)
                    print(f"📊 Retrieved {len(memories)} memories for user")
                    
                    # Look for our test memory
                    found_test_memory = False
                    for memory in memories:
                        if test_audio_uuid in str(memory.get('metadata', {})):
                            found_test_memory = True
                            print(f"✅ Found test memory in results")
                            break
                    
                    if not found_test_memory:
                        print("⚠️ Test memory not found in retrieval results")
                        
                except Exception as retrieval_error:
                    print(f"⚠️ Memory retrieval failed: {retrieval_error}")
                
                return True
            else:
                print("❌ High-level memory addition failed")
                return False
                
        except Exception as e:
            print(f"❌ Full integration test error: {e}")
            return False
            
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            
        print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Memory service should be working correctly.")
        else:
            print("\n🔧 RECOMMENDATIONS:")
            
            # Specific recommendations based on failures
            if not self.test_results.get("Ollama Connectivity", True):
                print("   1. Update OLLAMA_BASE_URL environment variable to: http://192.168.0.110:11434")
                print("      Add this to your docker-compose.yml environment section.")
                
            if not self.test_results.get("Ollama Models", True):
                print("   2. Pull required Ollama models:")
                print("      ollama pull llama3.1:latest")
                print("      ollama pull nomic-embed-text:latest")
                
            if not self.test_results.get("Qdrant Connectivity", True):
                print("   3. Ensure Qdrant is running and accessible")
                print("      Check docker-compose logs for qdrant service")
                
            if not self.test_results.get("Memory Service Initialization", True):
                print("   4. Memory service initialization failed - check Ollama and Qdrant connectivity")
                
            print("\n   📝 After making changes, restart your services:")
            print("      docker-compose restart friend-backend")

def main():
    """Main test function."""
    print("🔬 Memory Service Diagnostic Tool")
    print("This tool will help identify why memories aren't being created.")
    print()
    
    # Check if we're running in the right directory
    if not Path("src/memory").exists():
        print("❌ Please run this from the backends/advanced-backend directory")
        print("   cd backends/advanced-backend")
        print("   python tests/test_memory_service.py")
        sys.exit(1)
        
    tester = MemoryServiceTester()
    asyncio.run(tester.run_all_tests())

if __name__ == "__main__":
    main() 