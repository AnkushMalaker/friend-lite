"""Audio processing backend using PyAnnote and SpeechBrain."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pyannote.audio import Audio, Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment

logger = logging.getLogger(__name__)


class AudioBackend:
    """Wrapper around PyAnnote & SpeechBrain components."""

    def __init__(self, hf_token: str, device: torch.device):
        self.device = device
        self.diar = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        ).to(device)
        
        # Configure pipeline with proper segmentation parameters to reduce over-segmentation
        # Note: embedding model is fixed in pre-trained pipeline and cannot be changed at instantiation
        pipeline_params = {
            'segmentation': {
                'min_duration_off': 1.5  # Fill gaps shorter than 1.5 seconds
            }
            # embedding_exclude_overlap is also fixed in the pre-trained pipeline
        }
        self.diar.instantiate(pipeline_params)
        
        # Use the EXACT same embedding model that the diarization pipeline uses internally
        self.embedder = PretrainedSpeakerEmbedding(
            "pyannote/wespeaker-voxceleb-resnet34-LM", device=device
        )
        self.loader = Audio(sample_rate=16_000, mono="downmix")

    def embed(self, wave: torch.Tensor) -> np.ndarray:  # (1, T)
        with torch.inference_mode():
            emb = self.embedder(wave.to(self.device))
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    async def async_embed(self, wave: torch.Tensor) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, wave)

    def diarize(self, path: Path, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> List[Dict]:
        """Perform speaker diarization on an audio file."""
        with torch.inference_mode():
            # Pass speaker count parameters to pyannote
            kwargs = {}
            if min_speakers is not None:
                kwargs['min_speakers'] = min_speakers
            if max_speakers is not None:
                kwargs['max_speakers'] = max_speakers
            
            diarization = self.diar(str(path), **kwargs)
            logger.info(f"Diarization: {diarization}")
            
            # Apply PyAnnote's built-in gap filling using support() method
            # This fills gaps shorter than 2 seconds between segments from same speaker
            diarization = diarization.support(collar=2.0)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
                "duration": float(turn.end - turn.start)
            })
        
        return segments

    async def async_diarize(self, path: Path, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> List[Dict]:
        """Async wrapper for diarization."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.diarize, path, min_speakers, max_speakers)

    def load_wave(self, path: Path, start: Optional[float] = None, end: Optional[float] = None) -> torch.Tensor:
        if start is not None and end is not None:
            seg = Segment(start, end)
            wav, _ = self.loader.crop(str(path), seg)
        else:
            wav, _ = self.loader(str(path))
        return wav.unsqueeze(0)  # (1, 1, T)