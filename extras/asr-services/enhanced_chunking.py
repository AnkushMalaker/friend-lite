"""
Enhanced chunking implementation for long audio processing.

This module provides a timestamp-preserving chunking solution using NeMo's
FrameBatchChunkedRNNT with proper timestamp joining following NeMo patterns.
"""

import logging
import torch
import time
from typing import List, Dict, Any, Optional
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchChunkedRNNT
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

logger = logging.getLogger(__name__)


class TimestampedFrameBatchChunkedRNNT(FrameBatchChunkedRNNT):
    """
    Enhanced FrameBatchChunkedRNNT that preserves word-level timestamps.

    Follows NeMo's FrameBatchMultiTaskAED pattern for proper timestamp handling
    with chunk_offsets and hypothesis merging.
    """

    def __init__(self, asr_model, frame_len=4, total_buffer=4, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size)
        self.all_hypotheses = []
        self.chunk_offsets = [0]  # Track chunk offsets like NeMo's FrameBatchMultiTaskAED
        self.merged_hypothesis = None

        # Get subsampling factor for timestamp calculations (following FrameBatchMultiTaskAED)
        self.subsampling_factor = getattr(asr_model._cfg.encoder, 'subsampling_factor', 4)

        # Ensure model is in eval mode and timestamps enabled
        self.asr_model.eval()
        if hasattr(self.asr_model, 'decoding') and hasattr(self.asr_model.decoding, 'compute_timestamps'):
            original_value = self.asr_model.decoding.compute_timestamps
            self.asr_model.decoding.compute_timestamps = True
            logger.info(f"🔧 TIMESTAMP CONFIG: Set compute_timestamps=True (was: {original_value})")
        else:
            logger.warning("🚨 TIMESTAMP CONFIG: Model does not have compute_timestamps attribute!")

    def reset(self):
        """Reset the chunked inference state and clear accumulated hypotheses."""
        super().reset()
        self.all_hypotheses = []
        self.chunk_offsets = [0]
        self.merged_hypothesis = None

    @torch.no_grad()
    def _get_batch_preds(self, keep_logits=False):
        """
        Override parent method to capture Hypothesis objects with timestamps.

        The parent class calls rnnt_decoder_predictions_tensor with return_hypotheses=False,
        which discards timestamp information. We override to capture the full Hypothesis objects.
        """
        # Ensure model is in eval mode to avoid batch norm issues
        self.asr_model.eval()
        device = self.asr_model.device

        for batch in iter(self.data_loader):
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)

            # Get encoder outputs - following parent class pattern
            encoded, encoded_len = self.asr_model(
                processed_signal=feat_signal, processed_signal_length=feat_signal_len
            )

            # KEY CHANGE: Get full Hypothesis objects instead of just text
            hypotheses = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len,
                return_hypotheses=True  # CRITICAL: Get timestamps and confidence
            )

            # Store hypotheses with chunk offset tracking
            self.all_hypotheses.extend(hypotheses)

            # Update chunk offsets for timestamp joining (following FrameBatchMultiTaskAED pattern)
            if hypotheses:
                # Calculate frame-based offset for proper timestamp alignment
                # Each chunk processes audio frames, track cumulative frame offset
                if len(self.chunk_offsets) > 1:
                    # Add frame offset based on processed frames
                    frame_offset = feat_signal.shape[1]  # Number of frames in current chunk
                    self.chunk_offsets.append(self.chunk_offsets[-1] + frame_offset)
                else:
                    # First chunk beyond initialization
                    self.chunk_offsets.append(feat_signal.shape[1])

            # Extract text for parent class compatibility
            best_hyp_text = [hyp.text if hyp.text else "" for hyp in hypotheses]
            self.all_preds.extend(best_hyp_text)

            # Cleanup - following parent class pattern
            del encoded, encoded_len

    def get_timestamped_results(self):
        """Return results with timestamp information using NeMo's timestamp joining."""
        if not self.all_hypotheses:
            return []

        # Join hypotheses using NeMo's FrameBatchMultiTaskAED pattern
        try:
            self.merged_hypothesis = self._join_hypotheses(self.all_hypotheses)
            return [self.merged_hypothesis] if self.merged_hypothesis else self.all_hypotheses
        except Exception as e:
            logger.warning(f"Hypothesis joining failed: {e}, returning raw hypotheses")
            return self.all_hypotheses

    def _join_hypotheses(self, hypotheses):
        """Join multiple hypotheses with proper timestamp alignment following NeMo's FrameBatchMultiTaskAED."""
        if len(hypotheses) == 1:
            return hypotheses[0]

        # Initialize a new hypothesis following NeMo's pattern
        merged_hypothesis = rnnt_utils.Hypothesis(
            score=0.0,
            y_sequence=torch.tensor([]),
            timestamp={
                'char': [],
                'word': [],
                'segment': [],
            },
        )

        # Join components following NeMo's FrameBatchMultiTaskAED pattern
        merged_hypothesis = self._join_text(merged_hypothesis, hypotheses)
        merged_hypothesis = self._join_y_sequence(merged_hypothesis, hypotheses)
        merged_hypothesis = self._join_timestamp(merged_hypothesis, hypotheses)

        return merged_hypothesis

    def _join_text(self, merged_hypothesis, hypotheses):
        """Join text from multiple hypotheses."""
        merged_hypothesis.text = " ".join([h.text for h in hypotheses if h.text])
        return merged_hypothesis

    def _join_y_sequence(self, merged_hypothesis, hypotheses):
        """Join y_sequence from multiple hypotheses."""
        y_sequences = [h.y_sequence for h in hypotheses if hasattr(h, 'y_sequence') and h.y_sequence is not None]
        if y_sequences:
            merged_hypothesis.y_sequence = torch.cat(y_sequences)
        return merged_hypothesis

    def _join_timestamp(self, merged_hypothesis, hypotheses):
        """Join timestamps from multiple hypotheses with proper offset handling."""
        cumulative_offset = 0

        for i, h in enumerate(hypotheses):
            if i < len(self.chunk_offsets):
                cumulative_offset = self.chunk_offsets[i]

            # Process word-level timestamps if available
            if hasattr(h, 'timestamp') and h.timestamp and 'word' in h.timestamp:
                word_timestamps = h.timestamp['word']
                updated_timestamps = []

                for word in word_timestamps:
                    if isinstance(word, dict):
                        updated_word = word.copy()
                        # Apply frame offset with subsampling factor
                        if 'start_offset' in word:
                            updated_word['start_offset'] = word['start_offset'] + cumulative_offset // self.subsampling_factor
                        if 'end_offset' in word:
                            updated_word['end_offset'] = word['end_offset'] + cumulative_offset // self.subsampling_factor
                        updated_timestamps.append(updated_word)

                merged_hypothesis.timestamp['word'].extend(updated_timestamps)

            # Process segment-level timestamps if available
            if hasattr(h, 'timestamp') and h.timestamp and 'segment' in h.timestamp:
                segment_timestamps = h.timestamp['segment']
                updated_timestamps = []

                for segment in segment_timestamps:
                    if isinstance(segment, dict):
                        updated_segment = segment.copy()
                        # Apply frame offset with subsampling factor
                        if 'start_offset' in segment:
                            updated_segment['start_offset'] = segment['start_offset'] + cumulative_offset // self.subsampling_factor
                        if 'end_offset' in segment:
                            updated_segment['end_offset'] = segment['end_offset'] + cumulative_offset // self.subsampling_factor
                        updated_timestamps.append(updated_segment)

                merged_hypothesis.timestamp['segment'].extend(updated_timestamps)

        return merged_hypothesis




def extract_timestamps_from_hypotheses(hypotheses: List[Hypothesis], chunk_start_time: float = 0.0, model=None) -> List[Dict[str, Any]]:
    """
    Extract word-level timestamps from NeMo Hypothesis objects.

    Args:
        hypotheses: List of NeMo Hypothesis objects
        chunk_start_time: Start time of the chunk for timestamp adjustment
        model: NeMo ASR model for accessing configuration

    Returns:
        List of word dictionaries with 'start', 'end', 'word', 'confidence'
    """
    try:
        words = []

        print(f"🔍 LEN CHECK 1: hypotheses type: {type(hypotheses)}")
        if hypotheses is None:
            print("🔍 LEN CHECK 1: hypotheses is None - returning empty list")
            return []

        print(f"🔍 LEN CHECK 1: About to call len(hypotheses)")
        logger.info(f"Processing {len(hypotheses)} hypotheses for timestamp extraction")
        print(f"🔍 LEN CHECK 1: Successfully called len(hypotheses) = {len(hypotheses)}")

        for i, hyp in enumerate(hypotheses):
            print(f"🔍 LEN CHECK 2: Processing hypothesis {i}, hyp type: {type(hyp)}")
            try:
                # Extract timestamps from NeMo Hypothesis structure:
                # timestamp={'timestep': [], 'char': [], 'word': [], 'segment': []}
                if hasattr(hyp, 'timestamp') and isinstance(hyp.timestamp, dict):
                    timestamp_dict = hyp.timestamp
                    print(f"🔍 LEN CHECK 3: timestamp_dict type: {type(timestamp_dict)}")

                    # Get word-level timestamps
                    word_timestamps = timestamp_dict.get('word', [])
                    print(f"🔍 DEBUG: word_timestamps type: {type(word_timestamps)}")
                    if word_timestamps is None:
                        logger.warning(f"Hypothesis {i}: word_timestamps is None, using empty list")
                        word_timestamps = []

                    word_confidence = getattr(hyp, 'word_confidence', [])
                    print(f"🔍 DEBUG: word_confidence type: {type(word_confidence)}")
                    if word_confidence is None:
                        logger.warning(f"Hypothesis {i}: word_confidence is None, using empty list")
                        word_confidence = []

                    print(f"🔍 DEBUG: word_timestamps length: {len(word_timestamps)}")
                    logger.info(f"Hypothesis {i}: Found {len(word_timestamps)} word timestamps")

                    # 🔍 CRITICAL DEBUG: Log structure of first word timestamp
                    if word_timestamps:
                        print(f"🔍 TIMESTAMP STRUCTURE: First word_timestamp: {word_timestamps[0]}")
                        print(f"🔍 TIMESTAMP STRUCTURE: Type: {type(word_timestamps[0])}")
                        if isinstance(word_timestamps[0], dict):
                            print(f"🔍 TIMESTAMP STRUCTURE: Keys: {list(word_timestamps[0].keys())}")
                        logger.info(f"🔍 TIMESTAMP STRUCTURE: {word_timestamps[0]}")

                    # Process word-level timing data
                    for j, word_timing in enumerate(word_timestamps):
                        print(f"🔍 DEBUG: Processing word {j}, word_timing: {word_timing}")
                        try:
                            logger.info(f"🔍 WORD PROCESSING: word {j}: {word_timing}")
                            if isinstance(word_timing, dict):
                                # Word timing is a dictionary with timing info
                                word_text = word_timing.get('word', '')

                                # Extract timestamps using NeMo's offset fields
                                start_offset = word_timing.get('start_offset', 0)
                                end_offset = word_timing.get('end_offset', 0)

                                # Convert frame offsets to time using NeMo's formula
                                window_stride = 0.01  # NeMo default
                                subsampling_factor = 4  # NeMo default
                                start_time = start_offset * window_stride * subsampling_factor
                                end_time = end_offset * window_stride * subsampling_factor

                                print(f"🔧 TIMESTAMP: {word_text} [{start_offset},{end_offset}] -> [{start_time:.3f}s,{end_time:.3f}s]")

                                print(f"🔍 LEN CHECK 9: word_confidence before len check: {type(word_confidence)}")
                                if word_confidence is not None:
                                    print(f"🔍 LEN CHECK 10: About to call len(word_confidence)")
                                    confidence_len = len(word_confidence)
                                    print(f"🔍 LEN CHECK 10: len(word_confidence) = {confidence_len}")
                                    logger.info(f"🔍 ISOLATE: word_confidence type: {type(word_confidence)}, len: {confidence_len}")
                                    confidence = word_confidence[j] if j < confidence_len else 1.0
                                else:
                                    print(f"🔍 LEN CHECK 10: word_confidence is None, using default confidence")
                                    confidence = 1.0

                                if word_text:  # Only add non-empty words
                                    word_dict = {
                                        'word': word_text,
                                        'start': float(start_time) + chunk_start_time,
                                        'end': float(end_time) + chunk_start_time,
                                        'confidence': float(confidence)
                                    }
                                    words.append(word_dict)
                                    print(f"🔍 LEN CHECK 11: Added word {j}: {word_text}")
                        except Exception as word_error:
                            print(f"🔍 LEN CHECK ERROR: Error processing word {j}: {word_error}")
                            logger.error(f"🔍 ISOLATE: Error processing word {j}: {word_error}", exc_info=True)
                            raise

                # Fallback: if no word timestamps but we have text, create words from text
                elif hasattr(hyp, 'text') and hyp.text:
                    print(f"🔍 LEN CHECK 12: Using text fallback for hyp {i}")
                    try:
                        # Split text into words and estimate timing
                        text_words = hyp.text.split()
                        print(f"🔍 LEN CHECK 13: text_words type: {type(text_words)}")
                        if text_words:
                            print(f"🔍 LEN CHECK 14: About to call len(text_words)")
                            # Estimate timing: spread words evenly across a reasonable duration
                            estimated_duration = max(len(text_words) * 0.5, 1.0)  # 0.5s per word minimum
                            print(f"🔍 LEN CHECK 14: len(text_words) = {len(text_words)}")
                            word_duration = estimated_duration / len(text_words)

                            for j, word in enumerate(text_words):
                                start_time = chunk_start_time + (j * word_duration)
                                end_time = chunk_start_time + ((j + 1) * word_duration)

                                word_dict = {
                                    'word': word,
                                    'start': start_time,
                                    'end': end_time,
                                    'confidence': getattr(hyp, 'score', 1.0)
                                }
                                words.append(word_dict)

                            logger.info(f"Hypothesis {i}: Created {len(text_words)} estimated word timings from text")
                    except Exception as text_error:
                        print(f"🔍 LEN CHECK ERROR: Text fallback error for hyp {i}: {text_error}")
                        logger.error(f"Error processing text fallback for hypothesis {i}: {text_error}", exc_info=True)

                # Create empty word entries for silence/non-speech if hypothesis is empty
                else:
                    print(f"🔍 LEN CHECK 15: Creating empty word entry for hyp {i}")
                    # This represents silence or non-speech audio
                    words.append({
                        'word': '',
                        'start': chunk_start_time,
                        'end': chunk_start_time + 1.0,
                        'confidence': 0.0
                    })

            except Exception as hyp_error:
                print(f"🔍 LEN CHECK ERROR: Error processing hypothesis {i}: {hyp_error}")
                logger.error(f"Error processing hypothesis {i}: {hyp_error}", exc_info=True)

        print(f"🔍 LEN CHECK 16: About to call len(words) at end")
        logger.info(f"Extracted {len(words)} total words with timestamps")
        print(f"🔍 LEN CHECK 16: Successfully called len(words) = {len(words)}")
        print(f"🔍 LEN CHECK 17: About to return words")
        return words

    except Exception as e:
        print(f"🔍 LEN CHECK ERROR: Critical error in extract_timestamps_from_hypotheses: {e}")
        logger.error(f"Critical error in extract_timestamps_from_hypotheses: {e}", exc_info=True)
        return []


async def transcribe_with_enhanced_chunking(model, audio_file_path: str,
                                          frame_len: float = 4.0,
                                          total_buffer: float = 8.0) -> Dict[str, Any]:
    """
    Transcribe long audio using enhanced chunking with NeMo's built-in timestamp preservation.

    Args:
        model: Loaded NeMo ASR model
        audio_file_path: Path to audio file
        frame_len: Duration of each frame in seconds (NeMo FrameBatchChunkedRNNT parameter)
        total_buffer: Total buffer duration in seconds (NeMo FrameBatchChunkedRNNT parameter)

    Returns:
        Dictionary with transcription results including word-level timestamps
    """
    # 📊 TIMING: Start overall timing
    overall_start = time.time()
    logger.info(f"📊 TIMING: Starting enhanced chunking transcription for {audio_file_path}")
    print(f"📊 TIMING: 🎯 PHASE START: Audio Upload/Processing at {time.strftime('%H:%M:%S')}")

    # Ensure model is in eval mode for inference
    model.eval()

    try:
        with torch.no_grad():
            # 📊 TIMING: Initialization phase
            init_start = time.time()
            logger.info(f"📊 TIMING: Initializing NeMo chunked processor...")
            print(f"📊 TIMING: 🔧 PHASE START: Initialization at {time.strftime('%H:%M:%S')}")

            # Initialize NeMo's chunked processor with timestamp preservation
            chunker = TimestampedFrameBatchChunkedRNNT(
                asr_model=model,
                frame_len=frame_len,      # Frame duration in seconds
                total_buffer=total_buffer, # Total buffer duration
                batch_size=1              # Process one chunk at a time
            )

            init_end = time.time()
            init_duration = init_end - init_start
            logger.info(f"📊 TIMING: ✅ Initialization completed in {init_duration:.3f}s")
            print(f"📊 TIMING: ✅ PHASE END: Initialization completed in {init_duration:.3f}s")

            # 📊 TIMING: Audio loading phase
            loading_start = time.time()
            logger.info(f"📊 TIMING: Loading audio file into chunker...")
            print(f"📊 TIMING: 📁 PHASE START: Audio Loading at {time.strftime('%H:%M:%S')}")

            # Process the audio file using NeMo's built-in chunking
            chunker.read_audio_file(
                audio_filepath=audio_file_path,
                delay=0,  # No delay
                model_stride_in_secs=0.02  # 20ms stride typical for ASR
            )

            loading_end = time.time()
            loading_duration = loading_end - loading_start
            logger.info(f"📊 TIMING: ✅ Audio loading completed in {loading_duration:.3f}s")
            print(f"📊 TIMING: ✅ PHASE END: Audio Loading completed in {loading_duration:.3f}s")

            # 📊 TIMING: Processing phase
            processing_start = time.time()
            logger.info(f"📊 TIMING: Running enhanced chunking inference...")
            print(f"📊 TIMING: 🚀 PHASE START: Processing at {time.strftime('%H:%M:%S')}")

            # Run inference with NeMo's chunked processing
            result_text = chunker.transcribe()

            processing_end = time.time()
            processing_duration = processing_end - processing_start
            logger.info(f"📊 TIMING: ✅ Processing completed in {processing_duration:.3f}s")
            print(f"📊 TIMING: ✅ PHASE END: Processing completed in {processing_duration:.3f}s")

            # 📊 TIMING: Reconcile phase (hypothesis extraction and merging)
            reconcile_start = time.time()
            logger.info(f"📊 TIMING: Extracting and reconciling hypotheses...")
            print(f"📊 TIMING: 🔀 PHASE START: Reconcile at {time.strftime('%H:%M:%S')}")

            hypotheses = chunker.get_timestamped_results()

            reconcile_end = time.time()
            reconcile_duration = reconcile_end - reconcile_start
            logger.info(f"📊 TIMING: ✅ Reconcile completed in {reconcile_duration:.3f}s")
            print(f"📊 TIMING: ✅ PHASE END: Reconcile completed in {reconcile_duration:.3f}s")

            # 📊 TIMING: Timestamp extraction phase
            timestamp_start = time.time()
            logger.info(f"📊 TIMING: Extracting word-level timestamps...")
            print(f"📊 TIMING: 📝 PHASE START: Timestamp Extraction at {time.strftime('%H:%M:%S')}")

            words = extract_timestamps_from_hypotheses(hypotheses, chunk_start_time=0.0, model=model)

            timestamp_end = time.time()
            timestamp_duration = timestamp_end - timestamp_start
            logger.info(f"📊 TIMING: ✅ Timestamp extraction completed in {timestamp_duration:.3f}s")
            print(f"📊 TIMING: ✅ PHASE END: Timestamp Extraction completed in {timestamp_duration:.3f}s")

            # 📊 TIMING: Final formatting phase
            format_start = time.time()
            logger.info(f"📊 TIMING: Formatting final response...")
            print(f"📊 TIMING: 📄 PHASE START: Final Formatting at {time.strftime('%H:%M:%S')}")

            if words is None:
                logger.warning("Words extraction returned None, using empty list")
                words = []

            words_count = len(words) if words else 0

            # Safe response creation
            if words and len(words) > 0:
                end_time = words[-1]['end']
                logger.info(f"Using end time from last word: {end_time}")
            else:
                end_time = 0.0
                logger.info("Using default end time: 0.0")

            response = {
                'text': result_text,
                'words': words,
                'segments': [{'start': 0.0, 'end': end_time, 'text': result_text}]
            }

            format_end = time.time()
            format_duration = format_end - format_start
            logger.info(f"📊 TIMING: ✅ Final formatting completed in {format_duration:.3f}s")
            print(f"📊 TIMING: ✅ PHASE END: Final Formatting completed in {format_duration:.3f}s")

        # 📊 TIMING: Overall completion summary
        overall_end = time.time()
        overall_duration = overall_end - overall_start

        logger.info(f"📊 TIMING: =================== COMPLETE TIMING SUMMARY ===================")
        logger.info(f"📊 TIMING: 🔧 Initialization: {init_duration:.3f}s")
        logger.info(f"📊 TIMING: 📁 Audio Loading: {loading_duration:.3f}s")
        logger.info(f"📊 TIMING: 🚀 Processing: {processing_duration:.3f}s")
        logger.info(f"📊 TIMING: 🔀 Reconcile: {reconcile_duration:.3f}s")
        logger.info(f"📊 TIMING: 📝 Timestamp Extraction: {timestamp_duration:.3f}s")
        logger.info(f"📊 TIMING: 📄 Final Formatting: {format_duration:.3f}s")
        logger.info(f"📊 TIMING: 🎯 TOTAL END-TO-END: {overall_duration:.3f}s")
        logger.info(f"📊 TIMING: Enhanced chunking completed. Transcribed {words_count} words")
        logger.info(f"📊 TIMING: ================================================================")

        print(f"📊 TIMING: =================== COMPLETE TIMING SUMMARY ===================")
        print(f"📊 TIMING: 🔧 Initialization: {init_duration:.3f}s")
        print(f"📊 TIMING: 📁 Audio Loading: {loading_duration:.3f}s")
        print(f"📊 TIMING: 🚀 Processing: {processing_duration:.3f}s")
        print(f"📊 TIMING: 🔀 Reconcile: {reconcile_duration:.3f}s")
        print(f"📊 TIMING: 📝 Timestamp Extraction: {timestamp_duration:.3f}s")
        print(f"📊 TIMING: 📄 Final Formatting: {format_duration:.3f}s")
        print(f"📊 TIMING: 🎯 TOTAL END-TO-END: {overall_duration:.3f}s")
        print(f"📊 TIMING: Enhanced chunking completed. Transcribed {words_count} words")
        print(f"📊 TIMING: ================================================================")

        return response

    except Exception as e:
        logger.error(f"Enhanced chunking failed: {e}")
        raise