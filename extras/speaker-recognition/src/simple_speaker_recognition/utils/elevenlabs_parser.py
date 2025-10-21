"""Parser for ElevenLabs JSON transcription output."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ElevenLabsParser:
    """Parse ElevenLabs JSON transcription output and extract speaker segments."""

    def __init__(self, min_segment_duration: float = 0.5):
        """Initialize parser.

        Args:
            min_segment_duration: Minimum duration for a segment in seconds
        """
        self.min_segment_duration = min_segment_duration

    def parse_elevenlabs_json(self, json_path: str) -> Dict[str, Any]:
        """Parse ElevenLabs JSON file and extract transcript data.

        Args:
            json_path: Path to ElevenLabs JSON file

        Returns:
            Parsed data with metadata and segments
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract text and language info
        transcript = data.get('text', '')
        language_code = data.get('language_code', '')
        language_probability = data.get('language_probability', 0.0)

        # Extract words with speaker information
        words = data.get('words', [])

        # Filter only actual words (skip spacing and audio events)
        filtered_words = [w for w in words if w.get('type') == 'word']

        # Group words into speaker segments
        segments = self._group_words_by_speaker(filtered_words)

        # Extract unique speakers
        unique_speakers = self._extract_unique_speakers(segments)

        # Calculate total duration from last word end time
        total_duration = 0.0
        if filtered_words:
            total_duration = filtered_words[-1].get('end', 0.0)

        return {
            'metadata': {
                'language_code': language_code,
                'language_probability': language_probability,
                'duration': total_duration
            },
            'transcript': transcript,
            'confidence': self._calculate_avg_confidence(filtered_words),
            'segments': segments,
            'unique_speakers': unique_speakers,
            'total_duration': total_duration
        }

    def _group_words_by_speaker(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group consecutive words by speaker into segments.

        Args:
            words: List of word dictionaries from ElevenLabs

        Returns:
            List of speaker segments
        """
        if not words:
            return []

        segments = []
        current_segment = None

        for word in words:
            speaker_id = word.get('speaker_id')
            if speaker_id is None:
                continue

            speaker_label = f"speaker_{speaker_id}"

            # Check if we need to start a new segment
            if (current_segment is None or
                current_segment['elevenlabs_speaker_label'] != speaker_label):

                # Save previous segment if it exists and meets minimum duration
                if current_segment is not None:
                    duration = current_segment['end_time'] - current_segment['start_time']
                    if duration >= self.min_segment_duration:
                        segments.append(current_segment)

                # Start new segment
                current_segment = {
                    'elevenlabs_speaker_label': speaker_label,
                    'start_time': word.get('start', 0.0),
                    'end_time': word.get('end', 0.0),
                    'text': word.get('text', ''),
                    'words': [word],
                    'confidence': self._logprob_to_confidence(word.get('logprob', 0.0))
                }
            else:
                # Extend current segment
                current_segment['end_time'] = word.get('end', 0.0)
                current_segment['text'] += ' ' + word.get('text', '')
                current_segment['words'].append(word)
                # Update average confidence
                word_confidence = self._logprob_to_confidence(word.get('logprob', 0.0))
                current_segment['confidence'] = (
                    (current_segment['confidence'] * (len(current_segment['words']) - 1) + word_confidence) /
                    len(current_segment['words'])
                )

        # Don't forget the last segment
        if current_segment is not None:
            duration = current_segment['end_time'] - current_segment['start_time']
            if duration >= self.min_segment_duration:
                segments.append(current_segment)

        return segments

    def _extract_unique_speakers(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique speakers and their statistics from segments.

        Args:
            segments: List of speaker segments

        Returns:
            List of unique speakers with statistics
        """
        speaker_stats = defaultdict(lambda: {
            'total_duration': 0.0,
            'segment_count': 0,
            'word_count': 0
        })

        for segment in segments:
            speaker = segment['elevenlabs_speaker_label']
            duration = segment['end_time'] - segment['start_time']
            speaker_stats[speaker]['total_duration'] += duration
            speaker_stats[speaker]['segment_count'] += 1
            speaker_stats[speaker]['word_count'] += len(segment['words'])

        unique_speakers = []
        for speaker, stats in speaker_stats.items():
            unique_speakers.append({
                'speaker': speaker,
                **stats
            })

        # Sort by total duration (most active speaker first)
        unique_speakers.sort(key=lambda x: x['total_duration'], reverse=True)

        return unique_speakers

    def _logprob_to_confidence(self, logprob: float) -> float:
        """Convert ElevenLabs logprob to confidence score (0-1).

        Args:
            logprob: Log probability from ElevenLabs

        Returns:
            Confidence score between 0 and 1
        """
        # ElevenLabs returns log probability (negative values closer to 0 are more confident)
        # Convert to confidence: closer to 0 = higher confidence
        return 1.0 - min(abs(logprob), 1.0)

    def _calculate_avg_confidence(self, words: List[Dict[str, Any]]) -> float:
        """Calculate average confidence from word list.

        Args:
            words: List of word dictionaries

        Returns:
            Average confidence score
        """
        if not words:
            return 0.0

        total_confidence = sum(
            self._logprob_to_confidence(w.get('logprob', 0.0))
            for w in words
        )
        return total_confidence / len(words)

    def extract_speaker_segments_for_identification(
        self,
        segments: List[Dict[str, Any]],
        audio_path: str
    ) -> List[Dict[str, Any]]:
        """Extract speaker segment info for identification.

        Args:
            segments: Parsed segments from parse_elevenlabs_json
            audio_path: Path to the audio file

        Returns:
            List of segment info dicts for speaker identification
        """
        segment_info = []
        for i, segment in enumerate(segments):
            segment_info.append({
                'segment_id': i,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'text': segment['text'],
                'audio_path': audio_path,
                'elevenlabs_speaker_label': segment['elevenlabs_speaker_label'],
                'confidence': segment['confidence']
            })
        return segment_info
