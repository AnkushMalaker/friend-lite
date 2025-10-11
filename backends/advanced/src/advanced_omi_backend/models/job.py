"""
Job models and base classes for RQ queue system.

This module provides:
- JobPriority enum for job priority levels
- BaseRQJob abstract class for common job setup and teardown
- async_job decorator for simplified job creation
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Callable
from functools import wraps

import redis.asyncio as redis_async

logger = logging.getLogger(__name__)


class JobPriority(str, Enum):
    """Priority levels for RQ job processing.

    Used to map priority to RQ job timeout values:
    - URGENT: 10 minutes timeout
    - HIGH: 8 minutes timeout
    - NORMAL: 5 minutes timeout (default)
    - LOW: 3 minutes timeout
    """
    URGENT = "urgent"      # 1 - Process immediately
    HIGH = "high"          # 2 - Process before normal
    NORMAL = "normal"      # 3 - Default priority
    LOW = "low"            # 4 - Process when idle


class BaseRQJob(ABC):
    """
    Base class for RQ job implementations.

    Handles common setup and teardown:
    - Event loop management
    - Beanie (MongoDB ODM) initialization
    - Redis client creation (optional)
    - Exception handling and logging

    Subclasses must implement the `execute()` method with job-specific logic.

    Example:
        class MyJob(BaseRQJob):
            async def execute(self) -> Dict[str, Any]:
                # Job-specific async logic here
                result = await some_async_operation()
                return {"success": True, "result": result}

        # RQ job function wrapper
        def my_job_function(arg1, arg2, redis_url=None):
            job = MyJob(redis_url=redis_url)
            return job.run(arg1=arg1, arg2=arg2)
    """

    def __init__(self, redis_url: Optional[str] = None, initialize_beanie: bool = True):
        """
        Initialize base job with common dependencies.

        Args:
            redis_url: Redis connection URL (optional, creates client if provided)
            initialize_beanie: Whether to initialize Beanie ODM (default True)
        """
        self.redis_url = redis_url
        self.initialize_beanie = initialize_beanie
        self.redis_client: Optional[redis_async.Redis] = None
        self.job_start_time = time.time()

    async def _setup(self):
        """Setup common dependencies before job execution."""
        # Initialize Beanie for MongoDB access
        if self.initialize_beanie:
            from advanced_omi_backend.controllers.queue_controller import _ensure_beanie_initialized
            await _ensure_beanie_initialized()
            logger.debug("Beanie initialized")

        # Create Redis client if URL provided
        if self.redis_url:
            self.redis_client = redis_async.from_url(self.redis_url)
            logger.debug(f"Redis client created: {self.redis_url}")

    async def _teardown(self):
        """Cleanup resources after job execution."""
        if self.redis_client:
            await self.redis_client.close()
            logger.debug("Redis client closed")

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute job-specific logic.

        This method must be implemented by subclasses.

        Args:
            **kwargs: Job-specific parameters passed from RQ

        Returns:
            Dict with job results
        """
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the job with common setup and teardown.

        This method:
        1. Creates a new event loop
        2. Calls _setup() for dependencies
        3. Calls execute() with job-specific logic
        4. Calls _teardown() for cleanup
        5. Handles exceptions and logging

        Args:
            **kwargs: Job-specific parameters to pass to execute()

        Returns:
            Dict with job results
        """
        job_name = self.__class__.__name__
        logger.info(f"🚀 Starting {job_name}")

        try:
            # Create new event loop for this job
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                async def process():
                    await self._setup()
                    try:
                        result = await self.execute(**kwargs)
                        return result
                    finally:
                        await self._teardown()

                result = loop.run_until_complete(process())

                elapsed = time.time() - self.job_start_time
                logger.info(f"✅ {job_name} completed in {elapsed:.2f}s")
                return result

            finally:
                loop.close()

        except Exception as e:
            elapsed = time.time() - self.job_start_time
            logger.error(f"❌ {job_name} failed after {elapsed:.2f}s: {e}", exc_info=True)
            raise


def async_job(redis: bool = True, beanie: bool = True):
    """
    Decorator to convert async functions into RQ-compatible job functions.

    Handles common setup/teardown:
    - Event loop management
    - Beanie (MongoDB ODM) initialization
    - Redis client creation (optional)
    - Exception handling and logging

    Args:
        redis: If True, creates Redis client and passes as 'redis_client' kwarg
        beanie: If True, initializes Beanie ODM (default True)

    Example:
        @async_job(redis=True, beanie=True)
        async def my_job(arg1, arg2, redis_client=None):
            # Job logic with redis_client available
            result = await some_async_operation()
            return {"success": True, "result": result}

        # Enqueue directly (no wrapper needed)
        queue.enqueue(my_job, arg1_value, arg2_value)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            job_name = func.__name__
            start_time = time.time()
            logger.info(f"🚀 Starting {job_name}")

            redis_client = None

            try:
                # Create new event loop for this job
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    async def process():
                        nonlocal redis_client

                        # Initialize Beanie for MongoDB access
                        if beanie:
                            from advanced_omi_backend.controllers.queue_controller import _ensure_beanie_initialized
                            await _ensure_beanie_initialized()
                            logger.debug("Beanie initialized")

                        # Create Redis client if requested
                        if redis:
                            from advanced_omi_backend.controllers.queue_controller import REDIS_URL
                            redis_client = redis_async.from_url(REDIS_URL)
                            kwargs['redis_client'] = redis_client
                            logger.debug(f"Redis client created")

                        try:
                            # Call the actual job function
                            result = await func(*args, **kwargs)
                            return result
                        finally:
                            # Cleanup Redis client
                            if redis_client:
                                await redis_client.close()
                                logger.debug("Redis client closed")

                    result = loop.run_until_complete(process())

                    elapsed = time.time() - start_time
                    logger.info(f"✅ {job_name} completed in {elapsed:.2f}s")
                    return result

                finally:
                    loop.close()

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ {job_name} failed after {elapsed:.2f}s: {e}", exc_info=True)
                raise

        return wrapper
    return decorator