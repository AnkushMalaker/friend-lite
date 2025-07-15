from pydantic import BaseModel
from typing import List, Optional

class TranscriptSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float

class SpeechSegment(BaseModel):
    start: float
    end: float

class AudioChunk(BaseModel):
    audio_uuid: str
    audio_path: str
    client_id: str
    timestamp: int
    transcript: List[TranscriptSegment] = []
    speakers_identified: List[str] = []
    cropped_audio_path: Optional[str] = None
    speech_segments: List[SpeechSegment] = []
    cropped_duration: Optional[float] = None
    cropped_at: Optional[float] = None