#!/usr/bin/env python3
"""
Voice PE Audio Processor for XMOS XU316 Enhanced Audio
Handles 48kHz XMOS-processed audio with beamforming, noise suppression,
and intelligent resampling for optimal Deepgram transcription quality.
"""

import numpy as np
import logging
from typing import Optional
from scipy import signal
from scipy.signal import resample_poly
from wyoming.audio import AudioChunk

logger = logging.getLogger(__name__)


class VoicePEAudioProcessor:
    """
    Enhanced audio processor for Voice PE with XMOS XU316 + AIC3204.
    
    Processes XMOS-enhanced audio features:
    - 48kHz sample rate (XMOS native)
    - 32-bit containers with 24-bit effective resolution
    - Stereo beamformed output from dual microphones
    - Hardware noise suppression and echo cancellation
    """
    
    def __init__(
        self,
        input_sample_rate: int = 16000,  # Voice PE actually uses 16kHz
        output_sample_rate: int = 16000,
        input_channels: int = 2,
        output_channels: int = 1,
        bits_per_sample: int = 32,
        enable_quality_metrics: bool = True
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bits_per_sample = bits_per_sample
        self.enable_quality_metrics = enable_quality_metrics
        
        # Calculate resampling ratio
        self.resample_ratio = output_sample_rate / input_sample_rate
        logger.info(f"XMOS Audio Processor initialized: {input_sample_rate}Hz→{output_sample_rate}Hz, "
                   f"{input_channels}→{output_channels} channels")
        
        # Audio processing parameters
        self.noise_gate_threshold = -50.0  # dB
        self.voice_frequency_range = (80, 8000)  # Hz - human voice range
        self.quality_window_size = 1000  # samples for quality analysis
        
        # Quality metrics
        self.processed_chunks = 0
        self.total_input_samples = 0
        self.total_output_samples = 0
        self.average_snr = 0.0
        self.clipping_events = 0
        
        # Design anti-aliasing filter for downsampling
        self._setup_anti_aliasing_filter()
        
        logger.info(f"Voice PE Audio Processor ready - optimized for XMOS XU316 beamformed audio")
    
    def _setup_anti_aliasing_filter(self):
        """Setup anti-aliasing filter for high-quality downsampling."""
        nyquist_input = self.input_sample_rate / 2
        nyquist_output = self.output_sample_rate / 2
        
        # Cutoff at 95% of output Nyquist to prevent aliasing
        cutoff_freq = nyquist_output * 0.95
        normalized_cutoff = cutoff_freq / nyquist_input
        
        # Design Butterworth filter for smooth response in voice range
        self.aa_filter_order = 8
        self.aa_filter_b, self.aa_filter_a = signal.butter(
            self.aa_filter_order, 
            normalized_cutoff, 
            btype='low',
            analog=False
        )
        
        logger.debug(f"Anti-aliasing filter: order={self.aa_filter_order}, "
                    f"cutoff={cutoff_freq:.1f}Hz ({normalized_cutoff:.3f} normalized)")
    
    def _extract_effective_audio_from_32bit(self, audio_32bit: np.ndarray) -> np.ndarray:
        """
        Extract effective audio from 32-bit I2S containers.
        
        Based on ESPHome I2S documentation, this processes 32-bit samples 
        where the actual audio data may use the full 32-bit range or be 
        left-justified in the container.
        """
        # Convert to float64 for processing
        audio_float = audio_32bit.astype(np.float64)
        
        # Normalize to [-1, 1] range using full 32-bit signed range
        # This handles the standard I2S 32-bit format correctly
        audio_normalized = audio_float / (2**31)  # 32-bit signed max value
        
        # Clamp to prevent overflow (shouldn't be needed but safety first)
        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)
        
        return audio_normalized
    
    def _apply_stereo_beamforming_enhancement(self, stereo_audio: np.ndarray) -> np.ndarray:
        """
        Enhanced processing of XMOS beamformed stereo output.
        The XMOS has already done beamforming, but we can optimize the channels.
        """
        if stereo_audio.shape[1] != 2:
            logger.warning(f"Expected stereo input, got {stereo_audio.shape[1]} channels")
            return stereo_audio[:, 0] if stereo_audio.shape[1] > 0 else stereo_audio
        
        left_channel = stereo_audio[:, 0]
        right_channel = stereo_audio[:, 1]
        
        # Check if channels have different content (true beamformed output)
        if len(left_channel) > 1 and len(right_channel) > 1:
            correlation = np.corrcoef(left_channel, right_channel)[0, 1]
        else:
            correlation = 1.0  # Default to high correlation for short segments
        
        if correlation > 0.95:
            # Channels are very similar, use left channel (XMOS primary output)
            logger.debug("High channel correlation (%.3f), using left channel", correlation)
            return left_channel
        else:
            # Channels have different content, combine intelligently
            logger.debug("Low channel correlation (%.3f), combining channels", correlation)
            
            # Weighted combination favoring the channel with higher energy in voice range
            left_energy = self._calculate_voice_band_energy(left_channel)
            right_energy = self._calculate_voice_band_energy(right_channel)
            
            if left_energy > right_energy:
                # Left channel has more voice energy
                weight_left = 0.7
                weight_right = 0.3
            else:
                # Right channel has more voice energy  
                weight_left = 0.3
                weight_right = 0.7
            
            combined = weight_left * left_channel + weight_right * right_channel
            logger.debug("Combined channels with weights L:%.1f R:%.1f", weight_left, weight_right)
            return combined
    
    def _calculate_voice_band_energy(self, audio: np.ndarray) -> float:
        """Calculate energy in human voice frequency range."""
        # Simple energy calculation in voice band (300-3400 Hz)
        # More sophisticated methods could use FFT, but this is efficient for real-time
        
        # High-pass filter to remove low-frequency noise
        sos_hp = signal.butter(4, 300 / (self.input_sample_rate / 2), btype='high', output='sos')
        audio_hp = signal.sosfilt(sos_hp, audio)
        
        # Low-pass filter to remove high-frequency noise
        sos_lp = signal.butter(4, 3400 / (self.input_sample_rate / 2), btype='low', output='sos')
        audio_voice_band = signal.sosfilt(sos_lp, audio_hp)
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_voice_band**2))
        return energy
    
    def _intelligent_resample(self, audio: np.ndarray) -> np.ndarray:
        """
        High-quality resampling optimized for speech.
        For Voice PE, typically 16kHz→16kHz (no resampling needed).
        """
        if self.input_sample_rate == self.output_sample_rate:
            return audio
        
        # Apply anti-aliasing filter before downsampling
        audio_filtered = signal.filtfilt(self.aa_filter_b, self.aa_filter_a, audio)
        
        # Use polyphase resampling for high quality
        # Calculate integer factors for efficient resampling
        gcd = np.gcd(self.output_sample_rate, self.input_sample_rate)
        up_factor = self.output_sample_rate // gcd
        down_factor = self.input_sample_rate // gcd
        
        logger.debug(f"Resampling with up={up_factor}, down={down_factor}")
        
        # Resample using polyphase filters
        audio_resampled = resample_poly(audio_filtered, up_factor, down_factor)
        
        return audio_resampled
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to remove low-level noise between speech."""
        # Calculate RMS in overlapping windows
        window_size = min(1024, len(audio) // 4)
        if window_size < 64:
            return audio  # Too short for gating
        
        hop_size = window_size // 2
        rms_values = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(20 * np.log10(rms + 1e-10))  # Convert to dB
        
        # Apply gate based on threshold
        gate_mask = np.array(rms_values) > self.noise_gate_threshold
        
        # Smooth the gate to avoid artifacts
        from scipy.ndimage import binary_dilation
        gate_mask = binary_dilation(gate_mask, iterations=2)
        
        # Apply gate with smooth transitions
        gated_audio = audio.copy()
        for i, should_gate in enumerate(gate_mask):
            start_idx = i * hop_size
            end_idx = min(start_idx + window_size, len(audio))
            if not should_gate:
                # Apply gentle attenuation instead of hard gate
                gated_audio[start_idx:end_idx] *= 0.1
        
        return gated_audio
    
    def _calculate_quality_metrics(self, input_audio: np.ndarray, output_audio: np.ndarray):
        """Calculate audio quality metrics for monitoring."""
        if not self.enable_quality_metrics:
            return
        
        # Signal-to-noise ratio estimation
        signal_power = np.mean(output_audio**2)
        if signal_power > 0:
            # Estimate noise from quiet segments (bottom 10% of RMS values)
            window_size = 1024
            rms_values = []
            for i in range(0, len(output_audio) - window_size, window_size//2):
                window = output_audio[i:i + window_size]
                rms_values.append(np.sqrt(np.mean(window**2)))
            
            if rms_values:
                noise_estimate = np.percentile(rms_values, 10)**2
                if noise_estimate > 0:
                    snr_db = 10 * np.log10(signal_power / noise_estimate)
                    self.average_snr = (self.average_snr * self.processed_chunks + snr_db) / (self.processed_chunks + 1)
        
        # Check for clipping
        if np.any(np.abs(output_audio) > 0.95):
            self.clipping_events += 1
        
        # Update counters
        self.total_input_samples += len(input_audio)
        self.total_output_samples += len(output_audio)
        self.processed_chunks += 1
    
    def process_xmos_audio_chunk(self, chunk: AudioChunk) -> Optional[AudioChunk]:
        """
        Process a chunk of XMOS-enhanced audio.
        
        Pipeline:
        1. Extract 24-bit effective audio from 32-bit containers
        2. Process stereo beamformed channels
        3. Apply intelligent resampling (48kHz → 16kHz)
        4. Apply noise gating and enhancement
        5. Convert to target format
        
        Args:
            chunk: AudioChunk with XMOS audio data
            
        Returns:
            Enhanced AudioChunk ready for Deepgram transcription
        """
        try:
            # Convert bytes to numpy array
            if self.bits_per_sample == 32:
                audio_data = np.frombuffer(chunk.audio, dtype=np.int32)
            elif self.bits_per_sample == 16:
                audio_data = np.frombuffer(chunk.audio, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported bits per sample: {self.bits_per_sample}")
            
            # Handle empty data
            if len(audio_data) == 0:
                return None
            
            # Ensure we have complete samples for stereo
            if self.input_channels == 2:
                if len(audio_data) % 2 != 0:
                    audio_data = audio_data[:-1]  # Remove incomplete sample
                
                # Reshape to stereo
                audio_data = audio_data.reshape(-1, 2)
            
            # Step 1: Extract effective audio from 32-bit I2S containers
            if self.bits_per_sample == 32:
                audio_float = self._extract_effective_audio_from_32bit(audio_data)
            else:
                # Convert 16-bit to float
                audio_float = audio_data.astype(np.float64) / (2**(self.bits_per_sample-1))
            
            # Step 2: Process beamformed stereo channels
            if self.input_channels == 2 and self.output_channels == 1:
                audio_mono = self._apply_stereo_beamforming_enhancement(audio_float)
            elif self.input_channels == 1:
                audio_mono = audio_float.flatten() if audio_float.ndim > 1 else audio_float
            else:
                audio_mono = audio_float[:, 0]  # Take first channel
            
            # Step 3: Intelligent resampling for speech preservation
            audio_resampled = self._intelligent_resample(audio_mono)
            
            # Step 4: Apply noise gating (light touch - XMOS already did heavy lifting)
            audio_gated = self._apply_noise_gate(audio_resampled)
            
            # Step 5: Convert to 16-bit PCM for Deepgram
            audio_16bit = np.clip(audio_gated * 32767, -32768, 32767).astype(np.int16)
            
            # Calculate quality metrics
            self._calculate_quality_metrics(audio_data.flatten() if audio_data.ndim > 1 else audio_data, 
                                          audio_gated)
            
            # Create output chunk
            output_chunk = AudioChunk(
                audio=audio_16bit.tobytes(),
                rate=self.output_sample_rate,
                width=2,  # 16-bit = 2 bytes
                channels=self.output_channels,
                timestamp=chunk.timestamp if hasattr(chunk, 'timestamp') else None
            )
            
            # Periodic quality reporting
            if self.processed_chunks % 1000 == 0 and self.processed_chunks > 0:
                logger.info(f"XMOS Audio Quality: {self.processed_chunks} chunks, "
                           f"SNR: {self.average_snr:.1f}dB, "
                           f"Clipping: {self.clipping_events}, "
                           f"Samples: {self.total_input_samples}→{self.total_output_samples}")
            
            return output_chunk
            
        except Exception as e:
            logger.error(f"Error processing XMOS audio chunk: {e}")
            return None
    
    def get_quality_stats(self) -> dict:
        """Get current audio quality statistics."""
        return {
            'processed_chunks': self.processed_chunks,
            'total_input_samples': self.total_input_samples,
            'total_output_samples': self.total_output_samples,
            'average_snr_db': self.average_snr,
            'clipping_events': self.clipping_events,
            'processing_ratio': self.total_output_samples / max(1, self.total_input_samples)
        }