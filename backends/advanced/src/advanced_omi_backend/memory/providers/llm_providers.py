"""LLM provider implementations for memory service.

This module provides concrete implementations of LLM providers for:
- OpenAI (GPT models)
- Ollama (local models)

Each provider handles memory extraction, embedding generation, and
memory action proposals using their respective APIs.
"""

import json
import logging
from typing import Any, Dict, List, Optional

# TODO: Re-enable spacy when Docker build is fixed
# import spacy

from ..base import LLMProviderBase
from ..prompts import (
    FACT_RETRIEVAL_PROMPT,
    build_update_memory_messages,
    get_update_memory_messages,
)
from ..update_memory_utils import (
    extract_assistant_xml_from_openai_response,
    items_to_json,
    parse_memory_xml,
)
from ..utils import extract_json_from_text

memory_logger = logging.getLogger("memory_service")

# TODO: Re-enable spacy when Docker build is fixed
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     # Model not installed, fallback to None
#     memory_logger.warning("spacy model 'en_core_web_sm' not found. Using fallback text chunking.")
#     nlp = None
nlp = None  # Temporarily disabled

def chunk_text_with_spacy(text: str, max_tokens: int = 100) -> List[str]:
    """Split text into chunks using spaCy sentence segmentation.
    max_tokens is the maximum number of words in a chunk.
    """
    # Fallback chunking when spacy is not available
    if nlp is None:
        # Simple sentence-based chunking
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    # Original spacy implementation when available
    doc = nlp(text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_tokens = len(sent_text.split())  # Simple word count
        
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent_text
            current_tokens = sent_tokens
        else:
            current_chunk += " " + sent_text if current_chunk else sent_text
            current_tokens += sent_tokens
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

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
            import langfuse.openai as openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Use the provided prompt or fall back to default
            system_prompt = prompt if prompt.strip() else FACT_RETRIEVAL_PROMPT
            
            # local models can only handle small chunks of input text
            text_chunks = chunk_text_with_spacy(text)
            
            # Process all chunks in sequence, not concurrently
            results = [
                await self._process_chunk(client, system_prompt, chunk, i) 
                for i, chunk in enumerate(text_chunks)
            ]
            
            # Spread list of list of facts into a single list of facts
            cleaned_facts = []
            for result in results:
                memory_logger.info(f"Cleaned facts: {result}")
                cleaned_facts.extend(result)
            
            return cleaned_facts
                
        except Exception as e:
            memory_logger.error(f"OpenAI memory extraction failed: {e}")
            return []
        
    async def _process_chunk(self, client, system_prompt: str, chunk: str, index: int) -> List[str]:
        """Process a single text chunk to extract memories using OpenAI API.
        
        This private method handles the LLM interaction for a single chunk of text,
        sending it to OpenAI's chat completion API with the specified system prompt
        to extract structured memory facts.
        
        Args:
            client: OpenAI async client instance for API communication
            system_prompt: System prompt that guides the memory extraction behavior
            chunk: Individual text chunk to process for memory extraction
            index: Index of the chunk for logging and error tracking purposes
            
        Returns:
            List of extracted memory fact strings from the chunk. Returns empty list
            if no facts are found or if an error occurs during processing.
            
        Note:
            Errors are logged but don't propagate to avoid failing the entire
            memory extraction process.
        """
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            facts = (response.choices[0].message.content or "").strip()
            if not facts:
                return []

            return _parse_memories_content(facts)
            
        except Exception as e:
            memory_logger.error(f"Error processing chunk {index}: {e}")
            return []

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors, one per input text
        """
        try:
            import langfuse.openai as openai
            
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
            raise e

    async def test_connection(self) -> bool:
        """Test OpenAI connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            import langfuse.openai as openai
            
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
        retrieved_old_memory: List[Dict[str, str]] | List[str],
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
            import langfuse.openai as openai

            # Generate the complete prompt using the helper function
            memory_logger.debug(f"🧠 Facts passed to prompt builder: {new_facts}")
            update_memory_messages = build_update_memory_messages(
                retrieved_old_memory, 
                new_facts, 
                custom_prompt
            )
            memory_logger.debug(f"🧠 Generated prompt user content: {update_memory_messages[1]['content'][:200]}...")

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            response = await client.chat.completions.create(
                model=self.model,
                messages=update_memory_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = (response.choices[0].message.content or "").strip()
            if not content:
                return {}

            xml = extract_assistant_xml_from_openai_response(response)
            memory_logger.info(f"OpenAI propose_memory_actions xml: {xml}")
            items = parse_memory_xml(xml)
            memory_logger.info(f"OpenAI propose_memory_actions items: {items}")
            result = items_to_json(items)
            # example {'memory': [{'id': '0', 'event': 'UPDATE', 'text': 'My name is John', 'old_memory': None}}
            memory_logger.info(f"OpenAI propose_memory_actions result: {result}")
            return result

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
