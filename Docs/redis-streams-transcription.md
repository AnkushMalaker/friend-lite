# Redis Streams Multi-Provider Audio Transcription Implementation Guide

## Overview

This guide implements a Redis Streams-based architecture for real-time audio transcription using multiple providers (Deepgram, Whisper, Parakeet, etc.). The system provides intelligent routing, automatic fallbacks, and optimized performance for different use cases.

## Architecture Benefits

- **Real-time Processing**: Time-ordered audio chunks with millisecond precision
- **Multi-Provider Support**: Intelligent routing between Deepgram, Whisper, AssemblyAI, etc.
- **Automatic Failover**: Zero data loss with provider fallback chains
- **Scalability**: Consumer groups enable horizontal scaling
- **Persistence**: Audio chunks aren't lost if consumers disconnect
- **Load Balancing**: Distribute load based on provider strengths and capacity

## Core Architecture

```
Audio Input → Redis Streams Router → Provider-Specific Streams → Consumer Groups → Results Aggregation
                                  ↓
                     [deepgram_stream] → [deepgram_workers]
                     [whisper_stream]  → [whisper_workers] 
                     [parakeet_stream] → [parakeet_workers]
```

## Implementation

### 1. Core Dependencies

```python
# requirements.txt
redis>=4.5.0
faster-whisper>=0.10.0
deepgram-sdk>=3.0.0
assemblyai>=0.17.0
pydantic>=2.0.0
asyncio
uuid
```

### 2. Provider Configuration

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import redis
import time
import uuid
import json

class TranscriptionProvider(Enum):
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"
    PARAKEET = "parakeet"
    ASSEMBLYAI = "assemblyai"

@dataclass
class AudioChunk:
    data: bytes
    session_id: str
    chunk_id: str
    timestamp: float
    preferred_provider: Optional[TranscriptionProvider] = None
    fallback_providers: Optional[List[TranscriptionProvider]] = None
    real_time_required: bool = False
    accuracy_critical: bool = False
    cost_sensitive: bool = False

@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    provider: str
    processing_time: float
    chunk_id: str
    session_id: str
```

### 3. Redis Streams Producer

```python
class AudioStreamProducer:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    def add_audio_chunk(self, chunk: AudioChunk):
        """Add audio chunk to appropriate provider stream"""
        provider = self.select_optimal_provider(chunk)
        
        stream_name = f"audio:stream:{provider.value}"
        
        # Prepare chunk data for Redis
        chunk_data = {
            "chunk_data": chunk.data.hex(),  # Convert bytes to hex string
            "session_id": chunk.session_id,
            "chunk_id": chunk.chunk_id,
            "timestamp": chunk.timestamp,
            "real_time_required": str(chunk.real_time_required),
            "accuracy_critical": str(chunk.accuracy_critical),
            "cost_sensitive": str(chunk.cost_sensitive),
            "fallback_providers": ",".join([p.value for p in chunk.fallback_providers or []])
        }
        
        # Add to stream
        message_id = self.redis_client.xadd(stream_name, chunk_data)
        
        print(f"Added chunk {chunk.chunk_id} to {provider.value} stream: {message_id}")
        return message_id
    
    def select_optimal_provider(self, chunk: AudioChunk) -> TranscriptionProvider:
        """Intelligent provider selection"""
        
        # Explicit preference
        if chunk.preferred_provider:
            return chunk.preferred_provider
        
        # Real-time requirements → Deepgram (lowest latency)
        if chunk.real_time_required:
            return TranscriptionProvider.DEEPGRAM
        
        # High accuracy requirements → Whisper (highest accuracy)
        if chunk.accuracy_critical:
            return TranscriptionProvider.WHISPER
        
        # Cost optimization → Whisper (self-hosted)
        if chunk.cost_sensitive:
            return TranscriptionProvider.WHISPER
        
        # Load balancing - use least busy provider
        return self.get_least_loaded_provider()
    
    def get_least_loaded_provider(self) -> TranscriptionProvider:
        """Select provider with shortest queue"""
        queue_lengths = {}
        
        for provider in TranscriptionProvider:
            stream_name = f"audio:stream:{provider.value}"
            try:
                length = self.redis_client.xlen(stream_name)
                queue_lengths[provider] = length
            except:
                queue_lengths[provider] = 0
        
        return min(queue_lengths, key=queue_lengths.get)
```

### 4. Provider-Specific Consumers

```python
import os
import threading
from abc import ABC, abstractmethod

class BaseTranscriptionConsumer(ABC):
    def __init__(self, provider_name: str, redis_host='localhost', redis_port=6379):
        self.provider_name = provider_name
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.stream_name = f"audio:stream:{provider_name}"
        self.group_name = f"{provider_name}_workers"
        self.consumer_name = f"{provider_name}-worker-{os.getpid()}"
        
        # Create consumer group
        try:
            self.redis_client.xgroup_create(
                self.stream_name, 
                self.group_name, 
                "0", 
                mkstream=True
            )
        except redis.exceptions.ResponseError:
            pass  # Group already exists
    
    @abstractmethod
    def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        """Implement provider-specific transcription"""
        pass
    
    def start_consuming(self):
        """Start consuming messages from the stream"""
        print(f"Starting {self.provider_name} consumer: {self.consumer_name}")
        
        while True:
            try:
                messages = self.redis_client.xreadgroup(
                    self.group_name,
                    self.consumer_name,
                    {self.stream_name: ">"},
                    count=1,
                    block=1000
                )
                
                for stream, msgs in messages:
                    for message_id, fields in msgs:
                        self.process_message(message_id, fields)
                        
            except Exception as e:
                print(f"Error in {self.provider_name} consumer: {e}")
                time.sleep(1)
    
    def process_message(self, message_id: str, fields: dict):
        """Process a single message"""
        try:
            start_time = time.time()
            
            # Convert hex string back to bytes
            audio_data = bytes.fromhex(fields['chunk_data'])
            
            # Transcribe
            result = self.transcribe_audio(audio_data)
            result.chunk_id = fields['chunk_id']
            result.session_id = fields['session_id']
            result.provider = self.provider_name
            result.processing_time = time.time() - start_time
            
            # Store result
            self.store_result(result)
            
            # Acknowledge message
            self.redis_client.xack(self.group_name, self.stream_name, message_id)
            
            print(f"Processed {fields['chunk_id']} with {self.provider_name}")
            
        except Exception as e:
            print(f"Failed to process {fields['chunk_id']} with {self.provider_name}: {e}")
            self.handle_failure(fields, str(e))
    
    def store_result(self, result: TranscriptionResult):
        """Store transcription result"""
        result_data = {
            "text": result.text,
            "confidence": result.confidence,
            "provider": result.provider,
            "processing_time": result.processing_time,
            "chunk_id": result.chunk_id,
            "session_id": result.session_id,
            "timestamp": time.time()
        }
        
        # Store in results stream
        self.redis_client.xadd(
            f"transcription:results:{result.session_id}",
            result_data
        )
        
        # Also store in global results stream for monitoring
        self.redis_client.xadd("transcription:results:all", result_data)
    
    def handle_failure(self, fields: dict, error: str):
        """Handle transcription failure with fallback"""
        fallback_providers = fields.get('fallback_providers', '').split(',')
        fallback_providers = [p for p in fallback_providers if p and p != self.provider_name]
        
        if not fallback_providers:
            fallback_providers = self.get_default_fallback_chain()
        
        if fallback_providers:
            next_provider = fallback_providers[0]
            
            # Route to fallback provider
            fallback_data = {
                **fields,
                "original_provider": self.provider_name,
                "retry_count": str(int(fields.get("retry_count", "0")) + 1),
                "fallback_providers": ",".join(fallback_providers[1:]),
                "error_history": f"{fields.get('error_history', '')};{self.provider_name}:{error}"
            }
            
            self.redis_client.xadd(f"audio:stream:{next_provider}", fallback_data)
            
        # Log failure
        self.redis_client.xadd(
            "transcription:failures",
            {
                "failed_provider": self.provider_name,
                "chunk_id": fields['chunk_id'],
                "error": error,
                "timestamp": time.time()
            }
        )
    
    def get_default_fallback_chain(self):
        """Default fallback chain for this provider"""
        fallback_chains = {
            "deepgram": ["whisper", "assemblyai"],
            "whisper": ["deepgram", "assemblyai"],
            "parakeet": ["whisper", "deepgram"],
            "assemblyai": ["deepgram", "whisper"]
        }
        return fallback_chains.get(self.provider_name, ["whisper"])

class DeepgramConsumer(BaseTranscriptionConsumer):
    def __init__(self, api_key: str, **kwargs):
        super().__init__("deepgram", **kwargs)
        from deepgram import Deepgram
        self.deepgram = Deepgram(api_key)
    
    def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe using Deepgram API"""
        try:
            # Deepgram API call
            response = self.deepgram.transcription.sync_prerecorded(
                {"buffer": audio_data, "mimetype": "audio/wav"},
                {"punctuate": True, "model": "nova-2"}
            )
            
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            confidence = response["results"]["channels"][0]["alternatives"][0]["confidence"]
            
            return TranscriptionResult(
                text=transcript,
                confidence=confidence,
                provider="deepgram",
                processing_time=0,  # Will be set in process_message
                chunk_id="",
                session_id=""
            )
            
        except Exception as e:
            raise Exception(f"Deepgram transcription failed: {e}")

class WhisperConsumer(BaseTranscriptionConsumer):
    def __init__(self, model_size="large-v3", **kwargs):
        super().__init__("whisper", **kwargs)
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe using Whisper"""
        try:
            # Save audio data to temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                segments, info = self.model.transcribe(tmp_path, beam_size=5)
                
                transcript = ""
                total_confidence = 0
                segment_count = 0
                
                for segment in segments:
                    transcript += segment.text + " "
                    if hasattr(segment, 'avg_logprob'):
                        total_confidence += segment.avg_logprob
                        segment_count += 1
                
                confidence = total_confidence / segment_count if segment_count > 0 else 0
                
                return TranscriptionResult(
                    text=transcript.strip(),
                    confidence=confidence,
                    provider="whisper",
                    processing_time=0,
                    chunk_id="",
                    session_id=""
                )
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            raise Exception(f"Whisper transcription failed: {e}")
```

### 5. Results Aggregation

```python
class TranscriptionResultsConsumer:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    def get_session_results(self, session_id: str, wait_timeout: int = 30):
        """Get all transcription results for a session"""
        stream_name = f"transcription:results:{session_id}"
        
        try:
            # Read all messages from the beginning
            messages = self.redis_client.xrange(stream_name)
            
            results = []
            for message_id, fields in messages:
                results.append({
                    "message_id": message_id,
                    "text": fields['text'],
                    "confidence": float(fields['confidence']),
                    "provider": fields['provider'],
                    "processing_time": float(fields['processing_time']),
                    "chunk_id": fields['chunk_id'],
                    "timestamp": float(fields['timestamp'])
                })
            
            # Sort by chunk_id or timestamp
            results.sort(key=lambda x: x['timestamp'])
            return results
            
        except Exception as e:
            print(f"Error getting results for session {session_id}: {e}")
            return []
    
    def get_realtime_results(self, session_id: str, last_id: str = "0"):
        """Get new results since last_id for real-time streaming"""
        stream_name = f"transcription:results:{session_id}"
        
        try:
            messages = self.redis_client.xread({stream_name: last_id}, count=10, block=1000)
            
            results = []
            new_last_id = last_id
            
            for stream, msgs in messages:
                for message_id, fields in msgs:
                    results.append({
                        "message_id": message_id,
                        "text": fields['text'],
                        "confidence": float(fields['confidence']),
                        "provider": fields['provider'],
                        "chunk_id": fields['chunk_id']
                    })
                    new_last_id = message_id
            
            return results, new_last_id
            
        except Exception as e:
            print(f"Error getting realtime results: {e}")
            return [], last_id
```

### 6. Multi-Provider Orchestration

```python
class MultiProviderOrchestrator:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.producer = AudioStreamProducer(redis_host, redis_port)
        self.results_consumer = TranscriptionResultsConsumer(redis_host, redis_port)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    def transcribe_with_consensus(self, audio_data: bytes, session_id: str, providers: List[TranscriptionProvider] = None):
        """Send same audio to multiple providers for consensus"""
        if not providers:
            providers = [TranscriptionProvider.DEEPGRAM, TranscriptionProvider.WHISPER]
        
        correlation_id = str(uuid.uuid4())
        chunk_id = f"consensus_{correlation_id}"
        
        # Send to multiple providers
        for provider in providers:
            chunk = AudioChunk(
                data=audio_data,
                session_id=f"{session_id}_consensus_{correlation_id}",
                chunk_id=chunk_id,
                timestamp=time.time(),
                preferred_provider=provider
            )
            self.producer.add_audio_chunk(chunk)
        
        return correlation_id
    
    def get_consensus_result(self, correlation_id: str, timeout: int = 30):
        """Wait for and aggregate consensus results"""
        session_id = f"*_consensus_{correlation_id}"
        
        # Wait for results from all providers
        start_time = time.time()
        results = []
        
        while len(results) < 2 and (time.time() - start_time) < timeout:
            # Check for new results
            all_results = self.redis_client.keys(f"transcription:results:*_consensus_{correlation_id}")
            
            for result_stream in all_results:
                session_results = self.results_consumer.get_session_results(result_stream.split(':')[-1])
                results.extend(session_results)
            
            if len(results) < 2:
                time.sleep(0.5)
        
        if len(results) >= 2:
            return self.select_best_consensus_result(results)
        
        return results[0] if results else None
    
    def select_best_consensus_result(self, results: List[dict]) -> dict:
        """Select best result from consensus"""
        # Simple strategy: highest confidence
        return max(results, key=lambda x: x['confidence'])
```

### 7. Production Setup

```python
import threading
import signal
import sys

class TranscriptionService:
    def __init__(self, config: dict):
        self.config = config
        self.consumers = []
        self.producer = AudioStreamProducer(
            config['redis']['host'],
            config['redis']['port']
        )
        
    def start_all_consumers(self):
        """Start all provider consumers"""
        
        # Deepgram Consumer
        if 'deepgram' in self.config['providers']:
            deepgram_consumer = DeepgramConsumer(
                api_key=self.config['providers']['deepgram']['api_key'],
                redis_host=self.config['redis']['host'],
                redis_port=self.config['redis']['port']
            )
            consumer_thread = threading.Thread(target=deepgram_consumer.start_consuming)
            consumer_thread.daemon = True
            consumer_thread.start()
            self.consumers.append(consumer_thread)
        
        # Whisper Consumer
        if 'whisper' in self.config['providers']:
            whisper_consumer = WhisperConsumer(
                model_size=self.config['providers']['whisper']['model_size'],
                redis_host=self.config['redis']['host'],
                redis_port=self.config['redis']['port']
            )
            consumer_thread = threading.Thread(target=whisper_consumer.start_consuming)
            consumer_thread.daemon = True
            consumer_thread.start()
            self.consumers.append(consumer_thread)
        
        print(f"Started {len(self.consumers)} consumer threads")
    
    def shutdown(self):
        """Graceful shutdown"""
        print("Shutting down transcription service...")
        # Consumers will stop when main thread exits (daemon threads)

# Example configuration
config = {
    "redis": {
        "host": "localhost",
        "port": 6379
    },
    "providers": {
        "deepgram": {
            "api_key": "your_deepgram_api_key"
        },
        "whisper": {
            "model_size": "large-v3"
        }
    }
}

# Production startup
if __name__ == "__main__":
    service = TranscriptionService(config)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        service.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start all consumers
    service.start_all_consumers()
    
    print("Transcription service running. Press Ctrl+C to stop.")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.shutdown()
```

### 8. Usage Examples

```python
# Example: Basic transcription
def basic_example():
    producer = AudioStreamProducer()
    
    # Load audio file
    with open("audio.wav", "rb") as f:
        audio_data = f.read()
    
    # Create chunk
    chunk = AudioChunk(
        data=audio_data,
        session_id="session_123",
        chunk_id="chunk_001",
        timestamp=time.time(),
        real_time_required=True  # Will route to Deepgram
    )
    
    # Send for transcription
    message_id = producer.add_audio_chunk(chunk)
    print(f"Sent audio chunk: {message_id}")

# Example: Real-time streaming
def realtime_streaming_example():
    producer = AudioStreamProducer()
    results_consumer = TranscriptionResultsConsumer()
    
    session_id = "realtime_session_456"
    last_id = "0"
    
    # Simulate real-time audio chunks
    for i in range(10):
        # In real implementation, this would come from microphone
        audio_chunk = simulate_audio_chunk(i)
        
        chunk = AudioChunk(
            data=audio_chunk,
            session_id=session_id,
            chunk_id=f"chunk_{i:03d}",
            timestamp=time.time(),
            real_time_required=True
        )
        
        producer.add_audio_chunk(chunk)
        
        # Check for results
        results, last_id = results_consumer.get_realtime_results(session_id, last_id)
        for result in results:
            print(f"Real-time result: {result['text']}")
        
        time.sleep(2)  # 2-second chunks

def simulate_audio_chunk(chunk_num):
    # Placeholder - replace with actual audio data
    return b"fake_audio_data_" + str(chunk_num).encode()

if __name__ == "__main__":
    # Run examples
    basic_example()
    realtime_streaming_example()
```

## Monitoring and Troubleshooting

### Key Redis Commands for Monitoring

```bash
# Check stream lengths
XLEN audio:stream:deepgram
XLEN audio:stream:whisper

# Check consumer group info
XINFO GROUPS audio:stream:deepgram

# Check pending messages
XPENDING audio:stream:deepgram deepgram_workers

# Monitor failures
XRANGE transcription:failures - +

# View recent results
XREVRANGE transcription:results:all + - COUNT 10
```

### Performance Tuning

1. **Redis Configuration**:
   - Set `maxmemory-policy allkeys-lru`
   - Use `XTRIM` to prevent streams from growing unbounded
   - Enable persistence with AOF for reliability

2. **Consumer Scaling**:
   - Run multiple consumer processes per provider
   - Use different Redis consumer names for each process
   - Monitor queue lengths and scale accordingly

3. **Provider Selection**:
   - Deepgram: Best for real-time, low latency
   - Whisper: Best for accuracy, supports many languages
   - AssemblyAI: Good balance of speed and accuracy

## Deployment Checklist

- [ ] Redis server configured and running
- [ ] Provider API keys configured
- [ ] Consumer processes started for each provider
- [ ] Monitoring dashboards set up
- [ ] Fallback chains tested
- [ ] Stream retention policies configured
- [ ] Error handling and logging in place
- [ ] Load testing completed

This architecture provides a robust, scalable foundation for multi-provider audio transcription with Redis Streams handling all the orchestration!