#!/usr/bin/env python3
"""Integration test for OpenMemory MCP with Friend-Lite backend.

This test should be run from the backends/advanced directory after
starting the OpenMemory MCP server in extras/openmemory-mcp.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from advanced_omi_backend.memory.config import build_memory_config_from_env, MemoryProvider
from advanced_omi_backend.memory.service_factory import create_memory_service
from advanced_omi_backend.memory.providers.openmemory_mcp_service import OpenMemoryMCPService
from advanced_omi_backend.memory.providers.mcp_client import MCPClient


async def test_mcp_client():
    """Test the MCP client directly."""
    print("\n" + "="*60)
    print("Testing MCP Client")
    print("="*60)
    
    server_url = os.getenv("OPENMEMORY_MCP_URL", "http://localhost:8765")
    
    client = MCPClient(
        server_url=server_url,
        client_name="test_friendlite",
        user_id=os.getenv("OPENMEMORY_USER_ID", "openmemory")
    )
    
    async with client:
        # Test connection
        print("\n🔌 Testing connection...")
        connected = await client.test_connection()
        if not connected:
            print("❌ Cannot connect to OpenMemory MCP server")
            print(f"   Ensure server is running at {server_url}")
            return False
        print("✅ Connected to OpenMemory MCP")
        
        # Test adding memory
        print("\n📝 Testing memory creation...")
        test_text = "I love hiking in the Pacific Northwest, especially Mount Rainier."
        memory_ids = await client.add_memories(test_text)
        print(f"✅ Created {len(memory_ids)} memory(ies): {memory_ids}")
        
        # Test listing
        print("\n📚 Testing memory listing...")
        memories = await client.list_memories(limit=5)
        print(f"✅ Listed {len(memories)} memory(ies)")
        
        # Test search
        print("\n🔍 Testing memory search...")
        results = await client.search_memory("hiking", limit=3)
        print(f"✅ Search returned {len(results)} result(s)")
        
        return True


async def test_openmemory_service():
    """Test the OpenMemoryMCPService directly."""
    print("\n" + "="*60)
    print("Testing OpenMemoryMCPService")
    print("="*60)
    
    server_url = os.getenv("OPENMEMORY_MCP_URL", "http://localhost:8765")
    
    service = OpenMemoryMCPService(
        server_url=server_url,
        client_name="friendlite_service",
        user_id=os.getenv("OPENMEMORY_USER_ID", "openmemory")
    )
    
    # Initialize
    print("\n🚀 Initializing service...")
    await service.initialize()
    print("✅ Service initialized")
    
    # Test adding memory
    print("\n📝 Adding memory through service...")
    transcript = """
    User: I'm planning a trip to Japan next spring. I want to visit Tokyo and Kyoto.
    Assistant: That sounds wonderful! Spring is perfect for cherry blossoms.
    User: Yes, I'm hoping to see them. My budget is around $3000.
    """
    
    success, memory_ids = await service.add_memory(
        transcript=transcript,
        client_id="test_client",
        audio_uuid="audio_123",
        user_id=os.getenv("OPENMEMORY_USER_ID", "openmemory"),
        user_email="test@example.com"
    )
    
    if success:
        print(f"✅ Memory added: {len(memory_ids)} memory(ies)")
    else:
        print("⚠️ No memories created")
    
    # Test search
    print("\n🔍 Searching memories...")
    results = await service.search_memories(
        query="Japan travel",
        user_id=os.getenv("OPENMEMORY_USER_ID", "openmemory"),
        limit=5
    )
    print(f"✅ Found {len(results)} result(s)")
    for i, result in enumerate(results[:2], 1):
        print(f"   {i}. {result.content[:80]}...")
    
    # Test get all
    print("\n📚 Getting all memories...")
    all_memories = await service.get_all_memories(
        user_id=os.getenv("OPENMEMORY_USER_ID", "openmemory"),
        limit=10
    )
    print(f"✅ Retrieved {len(all_memories)} memory(ies)")
    
    # Cleanup
    service.shutdown()
    print("✅ Service shutdown complete")
    
    return True


async def test_service_factory():
    """Test the service factory with OpenMemory MCP configuration."""
    print("\n" + "="*60)
    print("Testing Service Factory Integration")
    print("="*60)
    
    # Set environment for OpenMemory MCP
    os.environ["MEMORY_PROVIDER"] = "openmemory_mcp"
    os.environ["OPENMEMORY_MCP_URL"] = os.getenv("OPENMEMORY_MCP_URL", "http://localhost:8765")
    
    # Build config
    print("\n🔧 Building configuration...")
    config = build_memory_config_from_env()
    print(f"✅ Config built: provider={config.memory_provider.value}")
    
    if config.memory_provider != MemoryProvider.OPENMEMORY_MCP:
        print("❌ Expected OPENMEMORY_MCP provider")
        return False
    
    # Create service via factory
    print("\n🏭 Creating service via factory...")
    service = create_memory_service(config)
    
    if not isinstance(service, OpenMemoryMCPService):
        print(f"❌ Expected OpenMemoryMCPService, got {type(service)}")
        return False
    
    print("✅ Correct service type created")
    
    # Initialize and test
    await service.initialize()
    print("✅ Service initialized via factory")
    
    # Quick functionality test
    success, _ = await service.add_memory(
        transcript="Testing factory-created service",
        client_id="factory_test",
        audio_uuid="factory_audio_456",
        user_id=os.getenv("OPENMEMORY_USER_ID", "openmemory"),
        user_email="factory@example.com"
    )
    
    if success:
        print("✅ Factory-created service is functional")
    
    service.shutdown()
    return True


async def main():
    """Run all integration tests."""
    print("🧪 Friend-Lite + OpenMemory MCP Integration Tests")
    print("="*60)
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set - OpenMemory may not work properly")
    
    server_url = os.getenv("OPENMEMORY_MCP_URL", "http://localhost:8765")
    print(f"📍 Testing against: {server_url}")
    print(f"   (Set OPENMEMORY_MCP_URL to change)")
    
    try:
        # Test 1: MCP Client
        if not await test_mcp_client():
            print("\n❌ MCP Client test failed")
            print("   Ensure OpenMemory MCP is running:")
            print("   cd extras/openmemory-mcp && ./run.sh")
            return
        
        # Test 2: OpenMemoryMCPService
        if not await test_openmemory_service():
            print("\n❌ OpenMemoryMCPService test failed")
            return
        
        # Test 3: Service Factory
        if not await test_service_factory():
            print("\n❌ Service Factory test failed")
            return
        
        print("\n" + "="*60)
        print("✅ All integration tests passed!")
        print("🎉 OpenMemory MCP is properly integrated with Friend-Lite")
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())