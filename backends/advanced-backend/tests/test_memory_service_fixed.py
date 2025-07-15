#!/usr/bin/env python3
"""
Fixed version of memory service test using public API.

This script tests:
1. Memory service initialization via public API
2. Memory creation through proper channels
3. Memory retrieval and search

Run this from the backend directory:
python tests/test_memory_service_fixed.py
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests

# Import public API only
try:
    from memory import (
        MemoryService, 
        get_memory_service, 
        init_memory_config,
    )
    print("✅ Successfully imported memory service modules")
except ImportError as e:
    print(f"❌ Failed to import memory service modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger("memory_test")

class MemoryServiceTester:
    """Memory service tester using public API."""
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.qdrant_url = os.getenv("QDRANT_BASE_URL", "qdrant")
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all tests in sequence."""
        print("🧪 Starting Memory Service Tests (Public API)")
        print("=" * 60)
        
        tests = [
            ("Configuration Check", self.test_configuration),
            ("External Dependencies", self.test_external_dependencies),
            ("Memory Service Initialization", self.test_memory_service_init),
            ("Memory Creation (Public API)", self.test_memory_creation),
            ("Memory Retrieval", self.test_memory_retrieval),
            ("Action Items", self.test_action_items),
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
        """Test configuration setup."""
        print(f"📋 Environment Configuration:")
        print(f"   OLLAMA_BASE_URL: {self.ollama_url}")
        print(f"   QDRANT_BASE_URL: {self.qdrant_url}")
        
        # Test config initialization
        try:
            config = init_memory_config(
                ollama_base_url=self.ollama_url,
                qdrant_base_url=self.qdrant_url
            )
            print("✅ Memory config initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Config initialization failed: {e}")
            return False
            
    def test_external_dependencies(self):
        """Test external service connectivity."""
        results = []
        
        # Test Ollama
        try:
            print(f"🌐 Testing Ollama connectivity...")
            response = requests.get(f"{self.ollama_url}/api/version", timeout=10)
            if response.status_code == 200:
                print(f"✅ Ollama accessible")
                results.append(True)
            else:
                print(f"❌ Ollama returned {response.status_code}")
                results.append(False)
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            results.append(False)
            
        # Test Qdrant
        try:
            print(f"🌐 Testing Qdrant connectivity...")
            # Try different possible URLs
            qdrant_urls = [
                f"http://{self.qdrant_url}:6333",
                "http://localhost:6333",
                "http://192.168.0.110:6333"
            ]
            
            qdrant_accessible = False
            for url in qdrant_urls:
                try:
                    response = requests.get(f"{url}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Qdrant accessible at {url}")
                        qdrant_accessible = True
                        break
                except:
                    continue
                    
            if not qdrant_accessible:
                print("❌ Qdrant not accessible on any URL")
            results.append(qdrant_accessible)
            
        except Exception as e:
            print(f"❌ Qdrant connection test failed: {e}")
            results.append(False)
            
        return all(results)
            
    async def test_memory_service_init(self):
        """Test MemoryService initialization through public API."""
        try:
            print("🚀 Testing MemoryService initialization...")
            
            # Get the global memory service
            service = get_memory_service()
            print(f"📊 Service obtained: {type(service).__name__}")
            
            # Test connection
            connection_ok = await service.test_connection()
            print(f"🔗 Connection test: {'✅ OK' if connection_ok else '❌ Failed'}")
            
            return connection_ok
                
        except Exception as e:
            print(f"❌ MemoryService initialization error: {e}")
            return False
            
    async def test_memory_creation(self):
        """Test memory creation through public API."""
        try:
            print("💾 Testing memory creation...")
            
            # Get service
            service = get_memory_service()
            
            # Test data
            test_transcript = "This is a test conversation about planning a project meeting for next week."
            test_client_id = "test_client_123"
            test_audio_uuid = f"test_audio_{int(time.time())}"
            test_user_id = "test_user_456"
            test_user_email = "test@example.com"
            
            print(f"📝 Creating memory for: {test_user_email}")
            
            # Create memory using public API
            result = await service.add_memory(
                test_transcript,
                test_client_id,
                test_audio_uuid,
                test_user_id,
                test_user_email
            )
            
            if result:
                print("✅ Memory creation successful")
                return True
            else:
                print("❌ Memory creation failed")
                return False
                
        except Exception as e:
            print(f"❌ Memory creation error: {e}")
            return False
            
    def test_memory_retrieval(self):
        """Test memory retrieval."""
        try:
            print("🔍 Testing memory retrieval...")
            
            service = get_memory_service()
            test_user_id = "test_user_456"
            
            # Get memories
            memories = service.get_all_memories(test_user_id, limit=10)
            print(f"📊 Retrieved {len(memories)} memories")
            
            # Test search
            search_results = service.search_memories("test conversation", test_user_id, limit=5)
            print(f"🔎 Search returned {len(search_results)} results")
            
            return True
                
        except Exception as e:
            print(f"❌ Memory retrieval error: {e}")
            return False
            
    def test_action_items(self):
        """Test action item functionality."""
        try:
            print("📋 Testing action items...")
            
            service = get_memory_service()
            test_user_id = "test_user_456"
            
            # Get action items
            action_items = service.get_action_items(test_user_id, limit=10)
            print(f"📊 Retrieved {len(action_items)} action items")
            
            return True
                
        except Exception as e:
            print(f"❌ Action items test error: {e}")
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
            print("🎉 All tests passed! Memory service is working correctly.")
        else:
            print("\n🔧 Issues found - check the failing tests above.")

def main():
    """Main test function."""
    print("🔬 Memory Service Test (Public API)")
    print("This tests the memory service using its intended public interface.")
    print()
    
    # Check if we're running in the right directory
    if not Path("src/memory").exists():
        print("❌ Please run this from the backends/advanced-backend directory")
        print("   cd backends/advanced-backend")
        print("   python tests/test_memory_service_fixed.py")
        sys.exit(1)
        
    tester = MemoryServiceTester()
    asyncio.run(tester.run_all_tests())

if __name__ == "__main__":
    main() 