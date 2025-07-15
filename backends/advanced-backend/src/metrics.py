import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure metrics logger
metrics_logger = logging.getLogger("metrics")


@dataclass
class ServiceMetrics:
    """Metrics for individual services"""
    name: str
    start_time: float = field(default_factory=time.time)
    total_uptime_seconds: float = 0.0
    last_health_check: Optional[float] = None
    health_check_successes: int = 0
    health_check_failures: int = 0
    reconnection_attempts: int = 0
    last_failure_time: Optional[float] = None
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class ClientMetrics:
    """Metrics for individual client connections"""
    client_id: str
    connection_start: float = field(default_factory=time.time)
    connection_end: Optional[float] = None
    total_connection_time: float = 0.0
    websocket_reconnections: int = 0
    audio_chunks_received: int = 0
    last_activity: float = field(default_factory=time.time)


@dataclass
class AudioProcessingMetrics:
    """Audio processing related metrics"""
    total_audio_duration_seconds: float = 0.0
    total_voice_activity_seconds: float = 0.0
    total_silence_seconds: float = 0.0
    chunks_processed_successfully: int = 0
    chunks_failed_processing: int = 0
    transcription_requests: int = 0
    transcription_successes: int = 0
    transcription_failures: int = 0
    memory_storage_requests: int = 0
    memory_storage_successes: int = 0
    memory_storage_failures: int = 0
    average_transcription_latency_ms: float = 0.0
    transcription_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class SystemMetrics:
    """Overall system metrics"""
    system_start_time: float = field(default_factory=time.time)
    last_report_time: Optional[float] = None
    services: Dict[str, ServiceMetrics] = field(default_factory=dict)
    clients: Dict[str, ClientMetrics] = field(default_factory=dict)
    audio: AudioProcessingMetrics = field(default_factory=AudioProcessingMetrics)
    active_client_count: int = 0


class MetricsCollector:
    """Central metrics collection and reporting system"""
    
    def __init__(self, debug_dir: str | Path):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = SystemMetrics()
        self._report_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize core services
        self._init_core_services()
        
        metrics_logger.info(f"Metrics collector initialized, reports will be saved to: {self.debug_dir}")
    
    def _init_core_services(self):
        """Initialize metrics tracking for core services"""
        core_services = [
            "friend-backend",
            "mongodb", 
            "qdrant",
            "asr-service",
            "memory-service",
            "speaker-service"
        ]
        
        for service_name in core_services:
            self.metrics.services[service_name] = ServiceMetrics(name=service_name)
    
    async def start(self):
        """Start the metrics collection and reporting"""
        if self._running:
            return
            
        self._running = True
        self._report_task = asyncio.create_task(self._periodic_report_loop())
        metrics_logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection and save final report"""
        if not self._running:
            return
            
        self._running = False
        if self._report_task:
            self._report_task.cancel()
            try:
                await self._report_task
            except asyncio.CancelledError:
                pass
        
        # Save final report
        await self._generate_report()
        metrics_logger.info("Metrics collection stopped")
    
    # Service Health Tracking
    def record_service_health_check(self, service_name: str, success: bool, failure_reason: str | None = None):
        """Record service health check result"""
        if service_name not in self.metrics.services:
            self.metrics.services[service_name] = ServiceMetrics(name=service_name)
        
        service = self.metrics.services[service_name]
        service.last_health_check = time.time()
        
        if success:
            service.health_check_successes += 1
        else:
            service.health_check_failures += 1
            service.last_failure_time = time.time()
            if failure_reason:
                service.failure_reasons.append(f"{datetime.now().isoformat()}: {failure_reason}")
                # Keep only last 10 failure reasons
                service.failure_reasons = service.failure_reasons[-10:]
    
    def record_service_reconnection(self, service_name: str):
        """Record service reconnection attempt"""
        if service_name not in self.metrics.services:
            self.metrics.services[service_name] = ServiceMetrics(name=service_name)
        
        self.metrics.services[service_name].reconnection_attempts += 1
    
    def update_service_uptime(self, service_name: str, uptime_seconds: float):
        """Update service uptime"""
        if service_name not in self.metrics.services:
            self.metrics.services[service_name] = ServiceMetrics(name=service_name)
        
        self.metrics.services[service_name].total_uptime_seconds = uptime_seconds
    
    # Client Connection Tracking
    def record_client_connection(self, client_id: str):
        """Record new client connection"""
        self.metrics.clients[client_id] = ClientMetrics(client_id=client_id)
        self.metrics.active_client_count = len([c for c in self.metrics.clients.values() if c.connection_end is None])
        metrics_logger.info(f"Client connected: {client_id}, active clients: {self.metrics.active_client_count}")
    
    def record_client_disconnection(self, client_id: str):
        """Record client disconnection"""
        if client_id in self.metrics.clients:
            client = self.metrics.clients[client_id]
            client.connection_end = time.time()
            client.total_connection_time = client.connection_end - client.connection_start
            self.metrics.active_client_count = len([c for c in self.metrics.clients.values() if c.connection_end is None])
            metrics_logger.info(f"Client disconnected: {client_id}, active clients: {self.metrics.active_client_count}")
    
    def record_client_reconnection(self, client_id: str):
        """Record client WebSocket reconnection"""
        if client_id in self.metrics.clients:
            self.metrics.clients[client_id].websocket_reconnections += 1
    
    def record_client_activity(self, client_id: str):
        """Update client last activity time"""
        if client_id in self.metrics.clients:
            self.metrics.clients[client_id].last_activity = time.time()
    
    def record_audio_chunk_received(self, client_id: str):
        """Record audio chunk received from client"""
        if client_id in self.metrics.clients:
            self.metrics.clients[client_id].audio_chunks_received += 1
    
    # Audio Processing Tracking
    def record_audio_chunk_saved(self, duration_seconds: float, voice_activity_seconds: float | None = None):
        """Record successful audio chunk save"""
        self.metrics.audio.total_audio_duration_seconds += duration_seconds
        self.metrics.audio.chunks_processed_successfully += 1
        
        if voice_activity_seconds is not None:
            self.metrics.audio.total_voice_activity_seconds += voice_activity_seconds
            self.metrics.audio.total_silence_seconds += (duration_seconds - voice_activity_seconds)
    
    def record_audio_chunk_failed(self):
        """Record failed audio chunk processing"""
        self.metrics.audio.chunks_failed_processing += 1
    
    def record_transcription_request(self):
        """Record transcription request sent"""
        self.metrics.audio.transcription_requests += 1
    
    def record_transcription_result(self, success: bool, latency_ms: float | None = None):
        """Record transcription result"""
        if success:
            self.metrics.audio.transcription_successes += 1
        else:
            self.metrics.audio.transcription_failures += 1
        
        if latency_ms is not None:
            self.metrics.audio.transcription_latencies.append(latency_ms)
            # Update rolling average
            if self.metrics.audio.transcription_latencies:
                self.metrics.audio.average_transcription_latency_ms = sum(self.metrics.audio.transcription_latencies) / len(self.metrics.audio.transcription_latencies)
    
    def record_memory_storage_request(self):
        """Record memory storage request"""
        self.metrics.audio.memory_storage_requests += 1
    
    def record_memory_storage_result(self, success: bool):
        """Record memory storage result"""
        if success:
            self.metrics.audio.memory_storage_successes += 1
        else:
            self.metrics.audio.memory_storage_failures += 1
    
    # Report Generation
    async def _periodic_report_loop(self):
        """Run periodic report generation loop (every 30 minutes)"""
        while self._running:
            try:
                # Wait 30 minutes between reports
                sleep_seconds = 30 * 60  # 30 minutes in seconds
                
                metrics_logger.info(f"Next metrics report in {sleep_seconds/60:.0f} minutes")
                
                await asyncio.sleep(sleep_seconds)
                await self._generate_report()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                metrics_logger.error(f"Error in periodic report loop: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes before retry
    
    async def _generate_report(self):
        """Generate and save periodic metrics report"""
        try:
            report_time = datetime.now()
            system_uptime = time.time() - self.metrics.system_start_time
            
            # Calculate derived metrics
            total_recording_time = self.metrics.audio.total_audio_duration_seconds
            total_voice_activity = self.metrics.audio.total_voice_activity_seconds
            
            # Service uptime percentages
            service_uptimes = {}
            for name, service in self.metrics.services.items():
                uptime_percentage = min(100.0, (service.total_uptime_seconds / system_uptime) * 100) if system_uptime > 0 else 0
                service_uptimes[name] = {
                    "uptime_seconds": service.total_uptime_seconds,
                    "uptime_percentage": round(uptime_percentage, 2),
                    "health_check_success_rate": round((service.health_check_successes / max(1, service.health_check_successes + service.health_check_failures)) * 100, 2),
                    "reconnection_attempts": service.reconnection_attempts,
                    "last_failure": service.last_failure_time,
                    "recent_failures": service.failure_reasons[-5:] if service.failure_reasons else []
                }
            
            # Client connection metrics
            client_stats = {
                "active_connections": self.metrics.active_client_count,
                "total_clients_seen": len(self.metrics.clients),
                "total_reconnections": sum(c.websocket_reconnections for c in self.metrics.clients.values()),
                "average_connection_duration_minutes": round(sum(c.total_connection_time for c in self.metrics.clients.values() if c.connection_end) / max(1, len([c for c in self.metrics.clients.values() if c.connection_end])) / 60, 2)
            }
            
            # Audio processing success rates
            audio_stats = {
                "total_recording_time_hours": round(total_recording_time / 3600, 2),
                "total_voice_activity_hours": round(total_voice_activity / 3600, 2),
                "voice_activity_percentage": round((total_voice_activity / max(1, total_recording_time)) * 100, 2),
                "chunk_processing_success_rate": round((self.metrics.audio.chunks_processed_successfully / max(1, self.metrics.audio.chunks_processed_successfully + self.metrics.audio.chunks_failed_processing)) * 100, 2),
                "transcription_success_rate": round((self.metrics.audio.transcription_successes / max(1, self.metrics.audio.transcription_requests)) * 100, 2),
                "memory_storage_success_rate": round((self.metrics.audio.memory_storage_successes / max(1, self.metrics.audio.memory_storage_requests)) * 100, 2),
                "average_transcription_latency_ms": round(self.metrics.audio.average_transcription_latency_ms, 2)
            }
            
            # Generate comprehensive report
            report = {
                "report_metadata": {
                    "generated_at": report_time.isoformat(),
                    "system_start_time": datetime.fromtimestamp(self.metrics.system_start_time).isoformat(),
                    "system_uptime_hours": round(system_uptime / 3600, 2),
                    "report_period_hours": round((time.time() - self.metrics.last_report_time) / 3600, 2) if self.metrics.last_report_time else round(system_uptime / 3600, 2)
                },
                "uptime_metrics": {
                    "system_uptime_vs_recording_time": {
                        "system_uptime_hours": round(system_uptime / 3600, 2),
                        "recording_time_hours": round(total_recording_time / 3600, 2),
                        "recording_efficiency_percentage": round((total_recording_time / max(1, system_uptime)) * 100, 2)
                    },
                    "service_uptimes": service_uptimes,
                    "client_connections": client_stats
                },
                "audio_processing_metrics": audio_stats,
                "raw_counters": {
                    "chunks_processed": self.metrics.audio.chunks_processed_successfully,
                    "chunks_failed": self.metrics.audio.chunks_failed_processing,
                    "transcription_requests": self.metrics.audio.transcription_requests,
                    "transcription_successes": self.metrics.audio.transcription_successes,
                    "memory_storage_requests": self.metrics.audio.memory_storage_requests,
                    "memory_storage_successes": self.metrics.audio.memory_storage_successes
                }
            }
            
            # Save report to file
            filename = f"metrics_report_{report_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.debug_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.metrics.last_report_time = time.time()
            
            metrics_logger.info(f"Metrics report saved: {filepath}")
            metrics_logger.info(f"System uptime: {system_uptime/3600:.1f}h, Recording: {total_recording_time/3600:.1f}h, Voice activity: {total_voice_activity/3600:.1f}h")
            
        except Exception as e:
            metrics_logger.error(f"Failed to generate metrics report: {e}")
    
    def get_current_metrics_summary(self) -> dict:
        """Get current metrics summary for API endpoints"""
        system_uptime = time.time() - self.metrics.system_start_time
        
        return {
            "system_uptime_hours": round(system_uptime / 3600, 2),
            "recording_time_hours": round(self.metrics.audio.total_audio_duration_seconds / 3600, 2),
            "active_clients": self.metrics.active_client_count,
            "chunks_processed": self.metrics.audio.chunks_processed_successfully,
            "transcription_success_rate": round((self.metrics.audio.transcription_successes / max(1, self.metrics.audio.transcription_requests)) * 100, 2),
            "voice_activity_hours": round(self.metrics.audio.total_voice_activity_seconds / 3600, 2),
            "services_status": {name: service.health_check_successes > service.health_check_failures for name, service in self.metrics.services.items()}
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector(debug_dir: str | Path = "/app/debug_dir") -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(debug_dir)
    return _metrics_collector

async def start_metrics_collection(debug_dir: str | Path):
    """Start metrics collection"""
    collector = get_metrics_collector(debug_dir)
    await collector.start()

async def stop_metrics_collection(debug_dir: str | Path):
    """Stop metrics collection"""
    collector = get_metrics_collector(debug_dir)
    await collector.stop()