"""
Health Monitoring and Service Recovery for Friend-Lite Backend

This module provides comprehensive health monitoring for all services and
automatic recovery mechanisms when services become unavailable.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import motor.motor_asyncio
from pymongo.errors import ServerSelectionTimeoutError

from .persistent_queue import PersistentQueue, get_persistent_queue
from .queue_tracker import QueueTracker, get_queue_tracker
from .recovery_manager import RecoveryManager, get_recovery_manager

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health information for a service"""

    name: str
    status: ServiceStatus
    last_check: float
    response_time: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    last_success: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check configuration"""

    name: str
    check_function: Callable
    timeout: float
    interval: float
    failure_threshold: int
    recovery_callback: Optional[Callable] = None


class HealthMonitor:
    """
    Comprehensive health monitoring system

    Features:
    - Service health monitoring with configurable checks
    - Automatic recovery triggers
    - Health history tracking
    - Circuit breaker integration
    - Health metrics and alerting
    """

    def __init__(
        self,
        recovery_manager: Optional[RecoveryManager] = None,
        queue_tracker: Optional[QueueTracker] = None,
        persistent_queue: Optional[PersistentQueue] = None,
    ):
        self.recovery_manager = recovery_manager or get_recovery_manager()
        self.queue_tracker = queue_tracker or get_queue_tracker()
        self.persistent_queue = persistent_queue or get_persistent_queue()

        self.health_checks: Dict[str, HealthCheck] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.running = False
        self.monitor_tasks: List[asyncio.Task] = []

        # Health monitoring stats
        self.stats = {
            "total_checks": 0,
            "failed_checks": 0,
            "services_recovered": 0,
            "uptime_start": time.time(),
        }

        # Initialize default health checks
        self._init_default_health_checks()

    def _init_default_health_checks(self):
        """Initialize default health checks for core services"""

        # MongoDB health check
        self.register_health_check(
            name="mongodb",
            check_function=self._check_mongodb,
            timeout=5.0,
            interval=30.0,
            failure_threshold=3,
            recovery_callback=self._recover_mongodb,
        )

        # Ollama health check
        self.register_health_check(
            name="ollama",
            check_function=self._check_ollama,
            timeout=10.0,
            interval=30.0,
            failure_threshold=3,
            recovery_callback=self._recover_ollama,
        )

        # Qdrant health check
        self.register_health_check(
            name="qdrant",
            check_function=self._check_qdrant,
            timeout=5.0,
            interval=30.0,
            failure_threshold=3,
            recovery_callback=self._recover_qdrant,
        )

        # ASR service health check
        self.register_health_check(
            name="asr_service",
            check_function=self._check_asr_service,
            timeout=5.0,
            interval=30.0,
            failure_threshold=2,
            recovery_callback=self._recover_asr_service,
        )

        # Queue health check
        self.register_health_check(
            name="processing_queues",
            check_function=self._check_processing_queues,
            timeout=2.0,
            interval=60.0,
            failure_threshold=2,
            recovery_callback=self._recover_processing_queues,
        )

    def register_health_check(
        self,
        name: str,
        check_function: Callable,
        timeout: float,
        interval: float,
        failure_threshold: int,
        recovery_callback: Optional[Callable] = None,
    ):
        """Register a health check"""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            timeout=timeout,
            interval=interval,
            failure_threshold=failure_threshold,
            recovery_callback=recovery_callback,
        )

        self.health_checks[name] = health_check
        self.service_health[name] = ServiceHealth(
            name=name, status=ServiceStatus.UNKNOWN, last_check=0, response_time=0
        )

        logger.info(f"Registered health check for {name}")

    async def start(self):
        """Start health monitoring"""
        if self.running:
            logger.warning("Health monitor already running")
            return

        self.running = True
        self.stats["uptime_start"] = time.time()

        # Start monitoring tasks for each health check
        for name, check in self.health_checks.items():
            task = asyncio.create_task(self._monitor_service(name, check))
            self.monitor_tasks.append(task)

        logger.info(f"Started health monitoring for {len(self.health_checks)} services")

    async def stop(self):
        """Stop health monitoring"""
        if not self.running:
            return

        self.running = False

        # Cancel all monitoring tasks
        for task in self.monitor_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.monitor_tasks:
            await asyncio.gather(*self.monitor_tasks, return_exceptions=True)

        self.monitor_tasks.clear()
        logger.info("Stopped health monitoring")

    async def _monitor_service(self, name: str, health_check: HealthCheck):
        """Monitor a single service"""
        while self.running:
            try:
                await self._run_health_check(name, health_check)
                await asyncio.sleep(health_check.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor for {name}: {e}")
                await asyncio.sleep(health_check.interval)

    async def _run_health_check(self, name: str, health_check: HealthCheck):
        """Run a single health check"""
        self.stats["total_checks"] += 1
        start_time = time.time()

        try:
            # Run the health check with timeout
            result = await asyncio.wait_for(
                health_check.check_function(), timeout=health_check.timeout
            )

            response_time = time.time() - start_time

            # Update service health
            service_health = self.service_health[name]
            service_health.status = ServiceStatus.HEALTHY if result else ServiceStatus.UNHEALTHY
            service_health.last_check = time.time()
            service_health.response_time = response_time
            service_health.error_message = None
            service_health.last_success = time.time()

            if result:
                service_health.consecutive_failures = 0
            else:
                service_health.consecutive_failures += 1
                self.stats["failed_checks"] += 1

                # Trigger recovery if threshold reached
                if (
                    service_health.consecutive_failures >= health_check.failure_threshold
                    and health_check.recovery_callback
                ):
                    await self._trigger_recovery(name, health_check)

        except asyncio.TimeoutError:
            self._handle_health_check_failure(name, "Health check timed out")
        except Exception as e:
            self._handle_health_check_failure(name, str(e))

    def _handle_health_check_failure(self, name: str, error_message: str):
        """Handle health check failure"""
        self.stats["failed_checks"] += 1

        service_health = self.service_health[name]
        service_health.status = ServiceStatus.UNHEALTHY
        service_health.last_check = time.time()
        service_health.error_message = error_message
        service_health.consecutive_failures += 1

        logger.warning(f"Health check failed for {name}: {error_message}")

        # Trigger recovery if threshold reached
        health_check = self.health_checks[name]
        if (
            service_health.consecutive_failures >= health_check.failure_threshold
            and health_check.recovery_callback
        ):
            asyncio.create_task(self._trigger_recovery(name, health_check))

    async def _trigger_recovery(self, name: str, health_check: HealthCheck):
        """Trigger recovery for a failed service"""
        try:
            logger.warning(f"Triggering recovery for {name}")

            if health_check.recovery_callback:
                await health_check.recovery_callback()
                self.stats["services_recovered"] += 1
                logger.info(f"Recovery triggered for {name}")

        except Exception as e:
            logger.error(f"Recovery failed for {name}: {e}")

    # Default health check implementations

    async def _check_mongodb(self) -> bool:
        """Check MongoDB health"""
        try:
            import os

            from motor.motor_asyncio import AsyncIOMotorClient

            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
            client = AsyncIOMotorClient(mongodb_uri)

            # Simple ping to check connection
            await client.admin.command("ping")
            client.close()
            return True

        except Exception as e:
            logger.debug(f"MongoDB health check failed: {e}")
            return False

    async def _check_ollama(self) -> bool:
        """Check Ollama health"""
        try:
            import os

            ollama_url = os.getenv("OLLAMA_URL", "http://192.168.0.110:11434")

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/version") as response:
                    return response.status == 200

        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    async def _check_qdrant(self) -> bool:
        """Check Qdrant health"""
        try:
            import os

            # Try internal Docker network first, then localhost
            qdrant_urls = ["http://qdrant:6333", "http://localhost:6333"]

            for url in qdrant_urls:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{url}/") as response:
                            if response.status == 200:
                                # Check if response contains Qdrant version info
                                data = await response.json()
                                if "title" in data and "qdrant" in data.get("title", "").lower():
                                    return True
                except:
                    continue

            return False

        except Exception as e:
            logger.debug(f"Qdrant health check failed: {e}")
            return False

    async def _check_asr_service(self) -> bool:
        """Check ASR service health"""
        try:
            import os

            # Check if using Deepgram or offline ASR
            deepgram_key = os.getenv("DEEPGRAM_API_KEY")
            if deepgram_key:
                # For Deepgram, we can't easily check without making a request
                # So we'll assume it's healthy if the key is present
                return True
            else:
                # Check offline ASR TCP connection
                asr_host = os.getenv("ASR_HOST", "192.168.0.110")
                asr_port = int(os.getenv("ASR_PORT", "8765"))

                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(asr_host, asr_port), timeout=3.0
                    )
                    writer.close()
                    await writer.wait_closed()
                    return True
                except:
                    return False

        except Exception as e:
            logger.debug(f"ASR service health check failed: {e}")
            return False

    async def _check_processing_queues(self) -> bool:
        """Check processing queues health"""
        try:
            # Check for stale processing items
            total_stale = 0
            for queue_type in ["chunk", "transcription", "memory", "action_item"]:
                try:
                    from .queue_tracker import QueueType

                    queue_enum = QueueType(queue_type.upper())
                    stale_items = self.queue_tracker.get_stale_processing_items(queue_enum, 300)
                    total_stale += len(stale_items)
                except:
                    pass

            # If too many stale items, consider unhealthy
            return total_stale < 10

        except Exception as e:
            logger.debug(f"Processing queues health check failed: {e}")
            return False

    # Recovery callbacks

    async def _recover_mongodb(self):
        """Recover MongoDB connection"""
        logger.info("Attempting MongoDB recovery - will rely on connection pooling")
        # MongoDB client should automatically reconnect
        pass

    async def _recover_ollama(self):
        """Recover Ollama connection"""
        logger.info("Attempting Ollama recovery - service may be restarting")
        # Ollama recovery would typically involve waiting for service restart
        pass

    async def _recover_qdrant(self):
        """Recover Qdrant connection"""
        logger.info("Attempting Qdrant recovery - checking service status")
        # Qdrant recovery would involve checking Docker container status
        pass

    async def _recover_asr_service(self):
        """Recover ASR service connection"""
        logger.info("Attempting ASR service recovery")
        # ASR service recovery would involve reconnecting websockets/TCP
        pass

    async def _recover_processing_queues(self):
        """Recover processing queues"""
        logger.info("Triggering processing queue recovery")
        try:
            # Trigger recovery manager to process stale items
            await self.recovery_manager.recover_from_startup()
        except Exception as e:
            logger.error(f"Processing queue recovery failed: {e}")

    # Public API methods

    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status for a specific service"""
        return self.service_health.get(service_name)

    def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status for all services"""
        return self.service_health.copy()

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        healthy_count = sum(
            1 for health in self.service_health.values() if health.status == ServiceStatus.HEALTHY
        )
        total_count = len(self.service_health)

        if healthy_count == total_count:
            overall_status = ServiceStatus.HEALTHY
        elif healthy_count > 0:
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.UNHEALTHY

        return {
            "overall_status": overall_status.value,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "uptime_seconds": time.time() - self.stats["uptime_start"],
            "services": {
                name: {
                    "status": health.status.value,
                    "last_check": health.last_check,
                    "response_time": health.response_time,
                    "consecutive_failures": health.consecutive_failures,
                    "error_message": health.error_message,
                }
                for name, health in self.service_health.items()
            },
            "stats": self.stats,
        }

    async def manual_health_check(self, service_name: str) -> Dict[str, Any]:
        """Manually trigger a health check for a service"""
        if service_name not in self.health_checks:
            return {"error": f"Service {service_name} not found"}

        health_check = self.health_checks[service_name]

        try:
            await self._run_health_check(service_name, health_check)
            service_health = self.service_health[service_name]

            return {
                "service": service_name,
                "status": service_health.status.value,
                "response_time": service_health.response_time,
                "error_message": service_health.error_message,
            }

        except Exception as e:
            return {"service": service_name, "error": str(e)}


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def init_health_monitor(
    recovery_manager: Optional[RecoveryManager] = None,
    queue_tracker: Optional[QueueTracker] = None,
    persistent_queue: Optional[PersistentQueue] = None,
):
    """Initialize the global health monitor"""
    global _health_monitor
    _health_monitor = HealthMonitor(recovery_manager, queue_tracker, persistent_queue)
    logger.info("Initialized health monitor")


def shutdown_health_monitor():
    """Shutdown the global health monitor"""
    global _health_monitor
    if _health_monitor:
        asyncio.create_task(_health_monitor.stop())
    _health_monitor = None
    logger.info("Shutdown health monitor")
