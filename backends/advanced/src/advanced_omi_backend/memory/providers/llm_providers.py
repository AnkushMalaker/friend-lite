"""LLM provider implementations for memory service.

This module provides concrete implementations of LLM providers for:
- OpenAI (GPT models)
- Ollama (local models)

Each provider handles memory extraction, embedding generation, and
memory action proposals using their respective APIs.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from ..base import LLMProviderBase
from ..utils import extract_json_from_text
from ..prompts import FACT_RETRIEVAL_PROMPT, get_update_memory_messages

memory_logger = logging.getLogger("memory_service")


class OpenAIProvider(LLMProviderBase):
    """OpenAI LLM provider implementation.
    
    Provides memory extraction, embedding generation, and memory action
    proposals using OpenAI's GPT and embedding models.
    
    Attributes:
        api_key: OpenAI API key
        model: GPT model to use for text generation
        embedding_model: Model to use for embeddings
        base_url: API base URL (for custom endpoints)
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens in responses
    """

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["api_key"]
        self.model = config.get("model", "gpt-4")
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)

    async def extract_memories(self, text: str, prompt: str) -> List[str]:
        """Extract memories using OpenAI API with the enhanced fact retrieval prompt.
        
        Args:
            text: Input text to extract memories from
            prompt: System prompt to guide extraction (uses default if empty)
            
        Returns:
            List of extracted memory strings
        """
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Use the provided prompt or fall back to default
            system_prompt = prompt if prompt.strip() else FACT_RETRIEVAL_PROMPT
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            facts = (response.choices[0].message.content or "").strip()
            if not facts:
                return []

            cleaned_facts = _parse_memories_content(facts)
            memory_logger.info(f"Cleaned facts: {cleaned_facts}")
            return cleaned_facts
                
        except Exception as e:
            memory_logger.error(f"OpenAI memory extraction failed: {e}")
            return []

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors, one per input text
        """
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            memory_logger.error(f"OpenAI embedding generation failed: {e}")
            return []

    async def test_connection(self) -> bool:
        """Test OpenAI connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            await client.models.list()
            return True
            
        except Exception as e:
            memory_logger.error(f"OpenAI connection test failed: {e}")
            return False

    async def propose_memory_actions(
        self,
        retrieved_old_memory: List[Dict[str, str]],
        new_facts: List[str],
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Use OpenAI chat completion with enhanced prompt to propose memory actions.
        
        Args:
            retrieved_old_memory: List of existing memories for context
            new_facts: List of new facts to process
            custom_prompt: Optional custom prompt to override default
            
        Returns:
            Dictionary containing proposed memory actions
        """
        try:
            import openai

            # Generate the complete prompt using the helper function
            prompt_message = get_update_memory_messages(
                retrieved_old_memory, 
                new_facts, 
                custom_prompt
            )

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = (response.choices[0].message.content or "").strip()
            if not content:
                return {}
            return json.loads(content)
        except Exception as e:
            memory_logger.error(f"OpenAI propose_memory_actions failed: {e}")
            return {}


class OllamaProvider(LLMProviderBase):
    """Ollama LLM provider implementation.
    
    Provides memory extraction, embedding generation, and memory action
    proposals using Ollama's GPT and embedding models.
    
    
    Use the openai provider for ollama with different environment variables
    
    os.environ["OPENAI_API_KEY"] = "ollama"  
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["QDRANT_BASE_URL"] = "localhost"
    os.environ["OPENAI_EMBEDDER_MODEL"] = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
    
    """
    pass

def _parse_memories_content(content: str) -> List[str]:
    """
    Parse LLM content to extract memory strings.

    Handles cases where the model returns:
    - A JSON object after </think> with keys like "facts" and "preferences"
    - A plain JSON array of strings
    - Non-JSON text (fallback to single memory)
    """
    try:
        # Try robust extraction first (handles </think> and mixed output)
        parsed = extract_json_from_text(content)
        if isinstance(parsed, dict):
            collected: List[str] = []
            for key in ("facts", "preferences"):
                value = parsed.get(key)
                if isinstance(value, list):
                    collected.extend(
                        [str(item).strip() for item in value if str(item).strip()]
                    )
            # If the dict didn't contain expected keys, try to flatten any list values
            if not collected:
                for value in parsed.values():
                    if isinstance(value, list):
                        collected.extend(
                            [str(item).strip() for item in value if str(item).strip()]
                        )
            if collected:
                return collected
    except Exception:
        # Continue to other strategies
        pass

    # If content includes </think>, try parsing the post-think segment directly
    if "</think>" in content:
        post_think = content.split("</think>", 1)[1].strip()
        if post_think:
            parsed_list = _try_parse_list_or_object(post_think)
            if parsed_list is not None:
                return parsed_list

    # Try to parse the whole content as a JSON list or object
    parsed_list = _try_parse_list_or_object(content)
    if parsed_list is not None:
        return parsed_list

    # Fallback: treat as a single memory string
    return [content] if content else []


def _try_parse_list_or_object(text: str) -> List[str] | None:
    """Try to parse text as JSON list or object and extract strings."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict):
            collected: List[str] = []
            for key in ("facts", "preferences"):
                value = data.get(key)
                if isinstance(value, list):
                    collected.extend(
                        [str(item).strip() for item in value if str(item).strip()]
                    )
            if collected:
                return collected
            # As a last attempt, flatten any list values
            for value in data.values():
                if isinstance(value, list):
                    collected.extend(
                        [str(item).strip() for item in value if str(item).strip()]
                    )
            return collected if collected else None
    except Exception:
        return None
