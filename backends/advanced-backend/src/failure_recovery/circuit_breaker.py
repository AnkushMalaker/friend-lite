"""
Circuit Breaker Implementation for Friend-Lite Backend

This module provides circuit breaker functionality to prevent cascading failures
and provide fast-fail behavior when services are unavailable.
"""

import asyncio
import logging
import time
from typing import Dict, Callable, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker is open (fast-fail)
    HALF_OPEN = "half_open"  # Testing if service is recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Number of failures before opening
    recovery_timeout: float = 60.0  # Time to wait before testing recovery
    success_threshold: int = 3      # Successes needed to close circuit
    timeout: float = 30.0          # Operation timeout
    slow_call_threshold: float = 5.0  # Slow call threshold in seconds

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """
    Circuit breaker implementation
    
    Provides protection against cascading failures by:
    - Monitoring failure rates
    - Opening circuit when failures exceed threshold
    - Providing fast-fail behavior when circuit is open
    - Testing service recovery periodically
    - Closing circuit when service is healthy again
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_request_time = 0
        
        # Statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "timeout_calls": 0,
            "slow_calls": 0,
            "circuit_opened_count": 0,
            "circuit_closed_count": 0
        }
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection"""
        async with self._lock:
            self.stats["total_calls"] += 1
            self.last_request_time = time.time()
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                else:
                    self.stats["failed_calls"] += 1
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
        
        # Execute the function
        start_time = time.time()
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Handle success
            await self._handle_success(execution_time)
            return result
            
        except asyncio.TimeoutError:
            await self._handle_timeout()
            raise
        except Exception as e:
            await self._handle_failure(e)
            raise
    
    async def _handle_success(self, execution_time: float):
        """Handle successful execution"""
        async with self._lock:
            self.stats["successful_calls"] += 1
            
            # Check for slow calls
            if execution_time > self.config.slow_call_threshold:
                self.stats["slow_calls"] += 1
                logger.warning(f"Slow call detected in {self.name}: {execution_time:.2f}s")
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    self._close_circuit()
                    
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    async def _handle_failure(self, error: Exception):
        """Handle failed execution"""
        async with self._lock:
            self.stats["failed_calls"] += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(f"Circuit breaker {self.name} failure: {error}")
            
            if self.state == CircuitState.HALF_OPEN:
                # Return to open state
                self._open_circuit()
            elif self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self.failure_count >= self.config.failure_threshold:
                    self._open_circuit()
    
    async def _handle_timeout(self):
        """Handle timeout"""
        async with self._lock:
            self.stats["timeout_calls"] += 1
            self.stats["failed_calls"] += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(f"Circuit breaker {self.name} timeout")
            
            if self.state == CircuitState.HALF_OPEN:
                self._open_circuit()
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._open_circuit()
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _open_circuit(self):
        """Open the circuit"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.stats["circuit_opened_count"] += 1
        logger.warning(f"Circuit breaker {self.name} OPENED")
    
    def _close_circuit(self):
        """Close the circuit"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.stats["circuit_closed_count"] += 1
        logger.info(f"Circuit breaker {self.name} CLOSED")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_request_time": self.last_request_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
                "slow_call_threshold": self.config.slow_call_threshold
            },
            "stats": self.stats.copy()
        }
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name} manually reset")

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()
    
    def get_circuit_breaker(self, 
                           name: str,
                           config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=config or self.default_config
            )
            logger.info(f"Created circuit breaker for {name}")
        
        return self.circuit_breakers[name]
    
    def set_default_config(self, config: CircuitBreakerConfig):
        """Set default configuration for new circuit breakers"""
        self.default_config = config
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: breaker.get_stats()
            for name, breaker in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        logger.info("Reset all circuit breakers")
    
    def reset_circuit_breaker(self, name: str) -> bool:
        """Reset a specific circuit breaker"""
        if name in self.circuit_breakers:
            self.circuit_breakers[name].reset()
            return True
        return False

# Decorator for circuit breaker protection
def circuit_breaker(name: str, 
                   config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to a function"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()
            breaker = manager.get_circuit_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

# Service-specific circuit breaker decorators
def mongodb_circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Circuit breaker for MongoDB operations"""
    mongodb_config = config or CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=10.0
    )
    return circuit_breaker("mongodb", mongodb_config)

def ollama_circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Circuit breaker for Ollama operations"""
    ollama_config = config or CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=2,
        timeout=45.0,
        slow_call_threshold=15.0
    )
    return circuit_breaker("ollama", ollama_config)

def qdrant_circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Circuit breaker for Qdrant operations"""
    qdrant_config = config or CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=10.0
    )
    return circuit_breaker("qdrant", qdrant_config)

def asr_circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Circuit breaker for ASR operations"""
    asr_config = config or CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=15.0
    )
    return circuit_breaker("asr", asr_config)

# Global circuit breaker manager
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager"""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager

def init_circuit_breaker_manager():
    """Initialize the global circuit breaker manager"""
    global _circuit_breaker_manager
    _circuit_breaker_manager = CircuitBreakerManager()
    logger.info("Initialized circuit breaker manager")

def shutdown_circuit_breaker_manager():
    """Shutdown the global circuit breaker manager"""
    global _circuit_breaker_manager
    _circuit_breaker_manager = None
    logger.info("Shutdown circuit breaker manager")