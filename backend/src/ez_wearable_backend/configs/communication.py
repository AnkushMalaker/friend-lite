from pydantic import BaseModel, Field

TRANSCRIBE_INTERVAL = 2
FRAME_SIZE = 360
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE / 10)


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=SAMPLE_RATE, description="Audio sample rate")
    channels: int = Field(default=CHANNELS, description="Number of audio channels")
    frame_size: int = Field(
        default=FRAME_SIZE, description="Frame size for Opus decoding"
    )


class ASRConfig(BaseModel):
    language: str = Field(default="en", description="Language for ASR")
    model_size: str = Field(default="distil-large-v3", description="Model size for ASR")
    use_vad: bool = Field(default=True, description="Use Voice Activity Detection")
    use_vac: bool = Field(default=False, description="Use Voice Activity Confirmation")


# TODO: Use a more generic config for VAD
# from faster_whisper.vad import VadOptions
class VADConfig(BaseModel):
    sampling_rate: int = Field(default=16000, description="Sampling rate for VAD")
    threshold: float = Field(default=0.5, description="Threshold for VAD")
    min_silence_duration: float = Field(
        default=0.1, description="Minimum silence duration for VAD"
    )
    speech_pad_ms: float = Field(default=100, description="Speech pad duration in ms")

    @property
    def chunk_size(self) -> int:
        return 512 if self.sampling_rate == 16000 else 256
