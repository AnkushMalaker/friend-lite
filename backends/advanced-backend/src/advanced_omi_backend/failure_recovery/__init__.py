"""
Failure Recovery System for Friend-Lite Backend

This package provides comprehensive failure recovery capabilities including:
- Persistent queue tracking
- Automatic retry mechanisms
- Health monitoring
- Circuit breaker protection
- Recovery management
- API endpoints for monitoring and control

Usage:
    from failure_recovery import init_failure_recovery_system, get_failure_recovery_router

    # Initialize the system
    await init_failure_recovery_system()

    # Get API router
    router = get_failure_recovery_router()
    app.include_router(router)
"""

import logging

from .api import get_failure_recovery_router
from .circuit_breaker import (  # Decorators
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerManager,
    CircuitState,
    asr_circuit_breaker,
    circuit_breaker,
    get_circuit_breaker_manager,
    init_circuit_breaker_manager,
    mongodb_circuit_breaker,
    ollama_circuit_breaker,
    qdrant_circuit_breaker,
    shutdown_circuit_breaker_manager,
)
from .health_monitor import (
    HealthMonitor,
    ServiceHealth,
    ServiceStatus,
    get_health_monitor,
    init_health_monitor,
    shutdown_health_monitor,
)
from .persistent_queue import (
    MessagePriority,
    PersistentMessage,
    PersistentQueue,
    get_persistent_queue,
    init_persistent_queue,
    shutdown_persistent_queue,
)
from .queue_tracker import (
    QueueItem,
    QueueStatus,
    QueueTracker,
    QueueType,
    get_queue_tracker,
    init_queue_tracker,
    shutdown_queue_tracker,
)
from .recovery_manager import (
    RecoveryAction,
    RecoveryManager,
    RecoveryRule,
    get_recovery_manager,
    init_recovery_manager,
    shutdown_recovery_manager,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Core classes
    "QueueTracker",
    "QueueItem",
    "QueueStatus",
    "QueueType",
    "PersistentQueue",
    "PersistentMessage",
    "MessagePriority",
    "RecoveryManager",
    "RecoveryRule",
    "RecoveryAction",
    "HealthMonitor",
    "ServiceHealth",
    "ServiceStatus",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitBreakerError",
    # Global getters
    "get_queue_tracker",
    "get_persistent_queue",
    "get_recovery_manager",
    "get_health_monitor",
    "get_circuit_breaker_manager",
    # Decorators
    "circuit_breaker",
    "mongodb_circuit_breaker",
    "ollama_circuit_breaker",
    "qdrant_circuit_breaker",
    "asr_circuit_breaker",
    # API
    "get_failure_recovery_router",
    # System management
    "init_failure_recovery_system",
    "shutdown_failure_recovery_system",
    "get_failure_recovery_status",
]

# Global system state
_system_initialized = False
_startup_recovery_completed = False


async def init_failure_recovery_system(
    queue_tracker_db: str = "queue_tracker.db",
    persistent_queue_db: str = "persistent_queues.db",
    start_monitoring: bool = True,
    start_recovery: bool = True,
    recovery_interval: int = 30,
):
    """
    Initialize the complete failure recovery system

    Args:
        queue_tracker_db: Path to queue tracker database
        persistent_queue_db: Path to persistent queue database
        start_monitoring: Whether to start health monitoring
        start_recovery: Whether to start recovery manager
        recovery_interval: Recovery check interval in seconds
    """
    global _system_initialized, _startup_recovery_completed

    if _system_initialized:
        logger.warning("Failure recovery system already initialized")
        return

    logger.info("Initializing failure recovery system...")

    try:
        # Initialize core components
        init_queue_tracker(queue_tracker_db)
        init_persistent_queue(persistent_queue_db)
        init_circuit_breaker_manager()

        # Get component instances
        queue_tracker = get_queue_tracker()
        persistent_queue = get_persistent_queue()
        circuit_manager = get_circuit_breaker_manager()

        # Initialize managers with dependencies
        init_recovery_manager(queue_tracker, persistent_queue)
        init_health_monitor(get_recovery_manager(), queue_tracker, persistent_queue)

        # Start monitoring and recovery if requested
        if start_monitoring:
            health_monitor = get_health_monitor()
            await health_monitor.start()
            logger.info("Health monitoring started")

        if start_recovery:
            recovery_manager = get_recovery_manager()
            await recovery_manager.start(recovery_interval)
            logger.info(f"Recovery manager started with {recovery_interval}s interval")

        _system_initialized = True
        logger.info("Failure recovery system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize failure recovery system: {e}")
        raise


async def perform_startup_recovery():
    """
    Perform startup recovery to handle items that were processing when service stopped
    """
    global _startup_recovery_completed

    if _startup_recovery_completed:
        logger.info("Startup recovery already completed")
        return

    if not _system_initialized:
        logger.error("Cannot perform startup recovery - system not initialized")
        return

    logger.info("Performing startup recovery...")

    try:
        recovery_manager = get_recovery_manager()
        await recovery_manager.recover_from_startup()

        _startup_recovery_completed = True
        logger.info("Startup recovery completed successfully")

    except Exception as e:
        logger.error(f"Startup recovery failed: {e}")
        raise


async def shutdown_failure_recovery_system():
    """
    Shutdown the complete failure recovery system
    """
    global _system_initialized, _startup_recovery_completed

    if not _system_initialized:
        logger.info("Failure recovery system not initialized, nothing to shutdown")
        return

    logger.info("Shutting down failure recovery system...")

    try:
        # Stop monitoring and recovery
        health_monitor = get_health_monitor()
        await health_monitor.stop()

        recovery_manager = get_recovery_manager()
        await recovery_manager.stop()

        # Shutdown components
        shutdown_health_monitor()
        shutdown_recovery_manager()
        shutdown_circuit_breaker_manager()
        shutdown_persistent_queue()
        shutdown_queue_tracker()

        _system_initialized = False
        _startup_recovery_completed = False

        logger.info("Failure recovery system shutdown complete")

    except Exception as e:
        logger.error(f"Error during failure recovery system shutdown: {e}")
        raise


def get_failure_recovery_status():
    """
    Get the current status of the failure recovery system
    """
    return {
        "system_initialized": _system_initialized,
        "startup_recovery_completed": _startup_recovery_completed,
        "components": {
            "queue_tracker": get_queue_tracker() is not None,
            "persistent_queue": get_persistent_queue() is not None,
            "recovery_manager": get_recovery_manager() is not None,
            "health_monitor": get_health_monitor() is not None,
            "circuit_breaker_manager": get_circuit_breaker_manager() is not None,
        },
    }


# Context manager for automatic system management
class FailureRecoverySystem:
    """
    Context manager for the failure recovery system

    Usage:
        async with FailureRecoverySystem() as system:
            # System is initialized and running
            pass
        # System is automatically shutdown
    """

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    async def __aenter__(self):
        await init_failure_recovery_system(**self.init_kwargs)
        await perform_startup_recovery()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await shutdown_failure_recovery_system()
        return False
