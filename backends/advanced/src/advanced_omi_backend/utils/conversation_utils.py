"""
Conversation utilities - speech detection, title/summary generation.

Extracted from legacy TranscriptionService to be reusable across V2 architecture.
"""

import logging
from typing import Optional

from advanced_omi_backend.config import get_speech_detection_settings
from advanced_omi_backend.llm_client import async_generate

logger = logging.getLogger(__name__)


def analyze_speech(transcript_data: dict) -> dict:
    """
    Analyze transcript for meaningful speech to determine if conversation should be created.

    Uses configurable thresholds from environment:
    - SPEECH_DETECTION_MIN_WORDS (default: 5)
    - SPEECH_DETECTION_MIN_CONFIDENCE (default: 0.5)

    Args:
        transcript_data: Dictionary with:
            - "text": str - Full transcript text
            - "words": list - Word-level data with confidence and timing (optional)
                [{"text": str, "confidence": float, "start": float, "end": float}, ...]

    Returns:
        dict: {
            "has_speech": bool,
            "reason": str,
            "word_count": int,
            "duration": float (seconds, 0.0 if no timing data),
            "speech_start": float (optional),
            "speech_end": float (optional),
            "fallback": bool (optional, true if text-only analysis)
        }

    Example:
        >>> result = analyze_speech({"text": "Hello world", "words": [...]})
        >>> if result["has_speech"]:
        >>>     print(f"Speech detected: {result['word_count']} words, {result['duration']}s")
    """
    settings = get_speech_detection_settings()
    words = transcript_data.get("words", [])

    # Method 1: Word-level analysis (preferred - has confidence scores and timing)
    if words:
        # Filter by confidence threshold
        valid_words = [
            w for w in words
            if w.get("confidence", 0) >= settings["min_confidence"]
        ]

        if len(valid_words) < settings["min_words"]:
            return {
                "has_speech": False,
                "reason": f"Not enough valid words ({len(valid_words)} < {settings['min_words']})",
                "word_count": len(valid_words),
                "duration": 0.0
            }

        # Calculate speech duration from word timing
        if valid_words:
            speech_start = valid_words[0].get("start", 0)
            speech_end = valid_words[-1].get("end", 0)
            speech_duration = speech_end - speech_start

            return {
                "has_speech": True,
                "word_count": len(valid_words),
                "speech_start": speech_start,
                "speech_end": speech_end,
                "duration": speech_duration,
                "reason": f"Valid speech detected ({len(valid_words)} words, {speech_duration:.1f}s)"
            }

    # Method 2: Text-only fallback (when no word-level data available)
    text = transcript_data.get("text", "").strip()
    if text:
        word_count = len(text.split())
        if word_count >= settings["min_words"]:
            return {
                "has_speech": True,
                "word_count": word_count,
                "speech_start": 0.0,
                "speech_end": 0.0,
                "duration": 0.0,
                "reason": f"Valid speech detected ({word_count} words, no timing data)",
                "fallback": True
            }

    # No speech detected
    return {
        "has_speech": False,
        "reason": "No meaningful speech content detected",
        "word_count": 0,
        "duration": 0.0
    }


async def generate_title(text: str) -> str:
    """
    Generate an LLM-powered title from conversation text.

    Args:
        text: Conversation transcript

    Returns:
        str: Generated title (3-6 words) or fallback
    """
    if not text or len(text.strip()) < 10:
        return "Conversation"

    try:
        prompt = f"""Generate a concise, descriptive title (3-6 words) for this conversation transcript:

"{text[:500]}"

Rules:
- Maximum 6 words
- Capture the main topic or theme
- No quotes or special characters
- Examples: "Planning Weekend Trip", "Work Project Discussion", "Medical Appointment"

Title:"""

        title = await async_generate(prompt, temperature=0.3)
        return title.strip().strip('"').strip("'") or "Conversation"

    except Exception as e:
        logger.warning(f"Failed to generate LLM title: {e}")
        # Fallback to simple title generation
        words = text.split()[:6]
        title = " ".join(words)
        return title[:40] + "..." if len(title) > 40 else title or "Conversation"


async def generate_summary(text: str) -> str:
    """
    Generate an LLM-powered summary from conversation text.

    Args:
        text: Conversation transcript

    Returns:
        str: Generated summary (1-2 sentences, max 120 chars) or fallback
    """
    if not text or len(text.strip()) < 10:
        return "No content"

    try:
        prompt = f"""Generate a brief, informative summary (1-2 sentences, max 120 characters) for this conversation:

"{text[:1000]}"

Rules:
- Maximum 120 characters
- 1-2 complete sentences
- Capture key topics and outcomes
- Use present tense
- Be specific and informative

Summary:"""

        summary = await async_generate(prompt, temperature=0.3)
        return summary.strip().strip('"').strip("'") or "No content"

    except Exception as e:
        logger.warning(f"Failed to generate LLM summary: {e}")
        # Fallback to simple summary generation
        return text[:120] + "..." if len(text) > 120 else text or "No content"


async def generate_title_with_speakers(segments: list) -> str:
    """
    Generate an LLM-powered title from conversation segments with speaker information.

    Args:
        segments: List of dicts with:
            [{"speaker": str, "text": str, "start": float, "end": float}, ...]

    Returns:
        str: Generated title (max 40 chars) or fallback
    """
    if not segments:
        return "Conversation"

    # Format conversation with speaker names
    conversation_text = ""
    for segment in segments[:10]:  # Use first 10 segments for title generation
        speaker = segment.get("speaker", "")
        text = segment.get("text", "").strip()
        if text:
            if speaker:
                conversation_text += f"{speaker}: {text}\n"
            else:
                conversation_text += f"{text}\n"

    if not conversation_text.strip():
        return "Conversation"

    try:
        prompt = f"""Generate a concise title (max 40 characters) for this conversation:

"{conversation_text[:500]}"

Rules:
- Maximum 40 characters
- Include speaker names if relevant
- Capture the main topic
- Be specific and informative

Title:"""

        title = await async_generate(prompt, temperature=0.3)
        title = title.strip().strip('"').strip("'")
        return title[:40] + "..." if len(title) > 40 else title or "Conversation"

    except Exception as e:
        logger.warning(f"Failed to generate LLM title with speakers: {e}")
        # Fallback to simple title generation
        words = conversation_text.split()[:6]
        title = " ".join(words)
        return title[:40] + "..." if len(title) > 40 else title or "Conversation"


async def generate_summary_with_speakers(segments: list) -> str:
    """
    Generate an LLM-powered summary from conversation segments with speaker information.

    Args:
        segments: List of dicts with:
            [{"speaker": str, "text": str, "start": float, "end": float}, ...]

    Returns:
        str: Generated summary (1-2 sentences, max 120 chars) or fallback
    """
    if not segments:
        return "No content"

    # Format conversation with speaker names
    conversation_text = ""
    speakers_in_conv = set()
    for segment in segments:
        speaker = segment.get("speaker", "")
        text = segment.get("text", "").strip()
        if text:
            if speaker:
                conversation_text += f"{speaker}: {text}\n"
                speakers_in_conv.add(speaker)
            else:
                conversation_text += f"{text}\n"

    if not conversation_text.strip():
        return "No content"

    try:
        prompt = f"""Generate a brief, informative summary (1-2 sentences, max 120 characters) for this conversation with speakers:

"{conversation_text[:1000]}"

Rules:
- Maximum 120 characters
- 1-2 complete sentences
- Include speaker names when relevant (e.g., "John discusses X with Sarah")
- Capture key topics and outcomes
- Use present tense
- Be specific and informative

Summary:"""

        summary = await async_generate(prompt, temperature=0.3)
        return summary.strip().strip('"').strip("'") or "No content"

    except Exception as e:
        logger.warning(f"Failed to generate LLM summary with speakers: {e}")
        # Fallback to simple summary generation
        return conversation_text[:120] + "..." if len(conversation_text) > 120 else conversation_text or "No content"
