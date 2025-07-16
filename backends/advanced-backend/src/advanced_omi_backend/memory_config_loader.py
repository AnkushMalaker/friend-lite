"""
Memory Configuration Loader

This module loads and manages memory extraction configuration from YAML files.
"""

import logging
import os
from typing import Any, Dict

import yaml

# Logger for configuration
config_logger = logging.getLogger("memory_config")


class MemoryConfigLoader:
    """
    Loads and manages memory extraction configuration from YAML files.
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize the config loader.

        Args:
            config_path: Path to the configuration YAML file
        """
        if config_path is None:
            # Default to memory_config.yaml in the backend root
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "memory_config.yaml"
            )

        self.config_path = config_path
        self.config = self._load_config()

        # Set up logging level from config
        debug_config = self.config.get("debug", {})
        log_level = debug_config.get("log_level", "INFO")
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        config_logger.setLevel(numeric_level)

        config_logger.info(f"Loaded memory configuration from {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            config_logger.error(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            config_logger.error(f"Error parsing YAML configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        # Get model from environment or use a fallback
        default_model = os.getenv("OLLAMA_MODEL", "gemma3n:e4b")

        return {
            "memory_extraction": {
                "enabled": True,
                "prompt": "Extract anything relevant about this conversation.",
                "llm_settings": {"temperature": 0.1, "max_tokens": 2000, "model": default_model},
            },
            "fact_extraction": {
                "enabled": False,
                "prompt": "Extract specific facts from this conversation.",
                "llm_settings": {"temperature": 0.0, "max_tokens": 1500, "model": default_model},
            },
            "categorization": {
                "enabled": False,
                "categories": ["work", "personal", "meeting", "other"],
                "prompt": "Categorize this conversation.",
                "llm_settings": {"temperature": 0.2, "max_tokens": 100, "model": default_model},
            },
            "quality_control": {
                "min_conversation_length": 50,
                "max_conversation_length": 50000,
                "skip_low_content": True,
                "min_content_ratio": 0.3,
                "skip_patterns": ["^(um|uh|hmm|yeah|ok|okay)\\s*$"],
            },
            "processing": {
                "parallel_processing": True,
                "max_concurrent_tasks": 1,
                "processing_timeout": 600,
                "retry_failed": True,
                "max_retries": 2,
                "retry_delay": 5,
            },
            "storage": {
                "store_metadata": True,
                "store_prompts": True,
                "store_llm_responses": True,
                "store_timing": True,
            },
            "debug": {
                "enabled": True,
                "db_path": "/app/debug/memory_debug.db",
                "log_level": "INFO",
                "log_full_conversations": False,
                "log_extracted_memories": True,
            },
        }

    def reload_config(self) -> bool:
        """Reload configuration from file."""
        try:
            self.config = self._load_config()
            config_logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            config_logger.error(f"Failed to reload configuration: {e}")
            return False

    def get_memory_extraction_config(self) -> Dict[str, Any]:
        """Get memory extraction configuration."""
        return self.config.get("memory_extraction", {})

    def get_fact_extraction_config(self) -> Dict[str, Any]:
        """Get fact extraction configuration."""
        return self.config.get("fact_extraction", {})

    def get_categorization_config(self) -> Dict[str, Any]:
        """Get categorization configuration."""
        return self.config.get("categorization", {})

    def get_quality_control_config(self) -> Dict[str, Any]:
        """Get quality control configuration."""
        return self.config.get("quality_control", {})

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.config.get("processing", {})

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self.config.get("storage", {})

    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration."""
        return self.config.get("debug", {})

    def is_memory_extraction_enabled(self) -> bool:
        """Check if memory extraction is enabled."""
        return self.get_memory_extraction_config().get("enabled", True)

    def is_fact_extraction_enabled(self) -> bool:
        """Check if fact extraction is enabled."""
        return self.get_fact_extraction_config().get("enabled", False)

    def is_categorization_enabled(self) -> bool:
        """Check if categorization is enabled."""
        return self.get_categorization_config().get("enabled", False)

    def is_debug_enabled(self) -> bool:
        """Check if debug tracking is enabled."""
        return self.get_debug_config().get("enabled", True)

    def get_memory_prompt(self) -> str:
        """Get the memory extraction prompt."""
        return self.get_memory_extraction_config().get(
            "prompt", "Extract anything relevant about this conversation."
        )

    def get_fact_prompt(self) -> str:
        """Get the fact extraction prompt."""
        return self.get_fact_extraction_config().get(
            "prompt", "Extract specific facts from this conversation."
        )

    def get_categorization_prompt(self) -> str:
        """Get the categorization prompt."""
        return self.get_categorization_config().get("prompt", "Categorize this conversation.")

    def get_llm_settings(self, extraction_type: str) -> Dict[str, Any]:
        """
        Get LLM settings for a specific extraction type.

        Args:
            extraction_type: One of 'memory', 'fact', 'categorization'
        """
        config_key = f"{extraction_type}_extraction"
        if extraction_type == "memory":
            config_key = "memory_extraction"
        elif extraction_type == "fact":
            config_key = "fact_extraction"
        elif extraction_type == "categorization":
            config_key = "categorization"

        extraction_config = self.config.get(config_key, {})
        return extraction_config.get("llm_settings", {})

    def should_skip_conversation(self, conversation_text: str) -> bool:
        """
        Check if a conversation should be skipped based on quality control settings.

        Args:
            conversation_text: The full conversation text

        Returns:
            True if the conversation should be skipped
        """
        quality_config = self.get_quality_control_config()

        # Check length constraints
        min_length = quality_config.get("min_conversation_length", 50)
        max_length = quality_config.get("max_conversation_length", 50000)

        if len(conversation_text) < min_length:
            config_logger.debug(
                f"Skipping conversation: too short ({len(conversation_text)} < {min_length})"
            )
            return True

        if len(conversation_text) > max_length:
            config_logger.debug(
                f"Skipping conversation: too long ({len(conversation_text)} > {max_length})"
            )
            return True

        # Check skip patterns
        skip_patterns = quality_config.get("skip_patterns", [])
        if skip_patterns:
            import re

            for pattern in skip_patterns:
                if re.match(pattern, conversation_text.strip(), re.IGNORECASE):
                    config_logger.debug(f"Skipping conversation: matches skip pattern '{pattern}'")
                    return True

        # Check content ratio (if enabled)
        if quality_config.get("skip_low_content", False):
            min_content_ratio = quality_config.get("min_content_ratio", 0.3)

            # Simple heuristic: calculate ratio of meaningful words to total words
            words = conversation_text.split()
            if len(words) > 0:
                filler_words = {
                    "um",
                    "uh",
                    "hmm",
                    "yeah",
                    "ok",
                    "okay",
                    "like",
                    "you",
                    "know",
                    "so",
                    "well",
                }
                meaningful_words = [
                    word for word in words if word.lower() not in filler_words and len(word) > 2
                ]
                content_ratio = len(meaningful_words) / len(words)

                if content_ratio < min_content_ratio:
                    config_logger.debug(
                        f"Skipping conversation: low content ratio ({content_ratio:.2f} < {min_content_ratio})"
                    )
                    return True

        return False

    def get_categories(self) -> list[str]:
        """Get available categories for classification."""
        return self.get_categorization_config().get("categories", [])

    def get_debug_db_path(self) -> str:
        """Get the debug database path."""
        return self.get_debug_config().get("db_path", "/app/debug/memory_debug.db")

    def should_log_full_conversations(self) -> bool:
        """Check if full conversations should be logged."""
        return self.get_debug_config().get("log_full_conversations", False)

    def should_log_extracted_memories(self) -> bool:
        """Check if extracted memories should be logged."""
        return self.get_debug_config().get("log_extracted_memories", True)

    def get_processing_timeout(self) -> int:
        """Get the processing timeout in seconds."""
        return self.get_processing_config().get("processing_timeout", 600)

    def should_retry_failed(self) -> bool:
        """Check if failed extractions should be retried."""
        return self.get_processing_config().get("retry_failed", True)

    def get_max_retries(self) -> int:
        """Get the maximum number of retries."""
        return self.get_processing_config().get("max_retries", 2)

    def get_retry_delay(self) -> int:
        """Get the delay between retries in seconds."""
        return self.get_processing_config().get("retry_delay", 5)


# Global instance
_config_loader = None


def get_config_loader() -> MemoryConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = MemoryConfigLoader()
    return _config_loader
