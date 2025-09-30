"""
Job priority definitions for RQ queue system.

This module contains only the job priority enum used by RQ workers.
RQ handles all other job tracking (status, results, timing, etc.) natively.
"""

from enum import Enum


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