# Voice PE Audio Quality Analysis

## Hardware Pipeline Overview

### Voice PE Hardware Components

**Home Assistant Voice Preview Edition** features:
- **XMOS XU316**: AI sound processing chip with advanced voice capabilities
- **AIC3204**: Texas Instruments DAC operating at 48kHz sampling rate
- **Dual Microphone Array**: Advanced microphone setup for beamforming
- **ESP32-S3**: Main controller handling I2S communication and networking

### Audio Processing Pipeline

```
[Microphone 1] ──┐
                 ├─► [XMOS XU316] ──► [ESP32-S3] ──► [TCP Stream] ──► [Backend]
[Microphone 2] ──┘      DSP              I2S           Network        Processing
```

#### Stage 1: Microphone Capture
- **Dual microphones** capture ambient audio
- Raw analog audio signals from microphone diaphragms
- Spatial separation enables beamforming capabilities

#### Stage 2: XMOS XU316 Processing
The XMOS XU316 chip performs hardware-level audio enhancement:

**Confirmed Features:**
- **Echo Cancellation**: Hardware-level echo removal
- **Stationary Noise Removal**: Background noise suppression
- **Auto Gain Control (AGC)**: Automatic level normalization
- **Beamforming**: Directional audio processing from dual microphones

**Unknown Specifications:**
- Internal sample rate during processing
- Bit depth of internal processing
- Channel output configuration
- I2S output format to ESP32

#### Stage 3: ESP32-S3 I2S Interface
- Receives processed audio from XMOS via I2S
- Configuration in `voice-pe.yaml`:
  ```yaml
  microphone:
    platform: i2s_audio
    sample_rate: 16000
    bits_per_sample: 32bit
    channel: stereo
    i2s_din_pin: GPIO15
  ```

#### Stage 4: TCP Network Streaming
- ESP32 streams I2S audio data via TCP
- Current processing in `ESP32TCPServer.read()`:
  ```python
  pcm32 = np.frombuffer(raw_data, dtype="<i4")  # 32-bit little-endian
  pcm32 = pcm32.reshape(-1, 2)[:, 0]           # Take LEFT channel only
  pcm16 = (pcm32 >> 16).astype(np.int16)       # Right shift 16 bits
  ```

## Current Audio Processing Issues

### Problem: Low Audio Levels Causing Poor Transcription

Despite having premium XMOS hardware processing, audio levels are very low, causing poor transcription quality:

**Symptoms**:
- Speech is audible but very quiet
- Deepgram may return empty or poor quality transcripts due to low signal levels
- Audio levels insufficient for reliable speech recognition

### Analysis of Current TCPServer Processing

**ESP32TCPServer class** (`main.py:160-237`) handles audio with these assumptions:

1. **Channel Selection**: Always takes left channel only (`[:, 0]`)
2. **Bit Format**: Assumes upper 16 bits contain audio (`>> 16`)
3. **Mono Output**: Discards right channel completely

**Potential Issues:**

1. **Incorrect Bit Handling/Gain**
   - 16-bit right shift (`>> 16`) may be discarding significant audio data
   - XMOS might use full 32-bit dynamic range
   - Audio levels too low after bit shifting

2. **Channel Selection with Low Gain**
   - Channel 0 (ASR) may have conservative AGC settings  
   - Channel 1 (Communications) confirmed to have very low levels
   - May need different channel or gain adjustment

3. **Missing Amplification**
   - XMOS processing may output at lower levels than expected
   - Need proper gain staging for Deepgram optimal levels
   - 32-bit to 16-bit conversion may need amplification

## Official XMOS Documentation Research

### ✅ CONFIRMED: XMOS XU316 Channel Configuration (Official Docs)

**Source**: [XMOS XVF3610 Voice Processor Audio Pipeline Documentation](https://www.xmos.com/documentation/XM-014600-PC/html/doc/product_description/audio_pipeline.html)

**OFFICIAL CHANNEL PURPOSE**:
- **Channel [0] (Left)**: Optimized for **Automatic Speech Recognition (ASR)**
- **Channel [1] (Right)**: Designed for **communications applications**

```yaml
voice_kit:
  # Voice Kit Firmware v1.3.1 Configuration  
  channel_0_stage: AGC    # Final AGC optimization for ASR
  channel_1_stage: NS     # Final NS optimization for communications
```

### Complete XMOS Processing Pipeline

**All audio goes through the full pipeline**:
1. PDM microphone input conversion
2. Acoustic Echo Cancellation (AEC)  
3. Automatic Delay Estimation & Control (ADEC)
4. Interference Cancellation (IC)
5. Voice Activity Detection (VAD)
6. **Noise Suppression (NS)** - applied to both channels
7. **Automatic Gain Control (AGC)** - applied to both channels

**Key Insight**: Both channels receive **both NS and AGC processing**. The `channel_X_stage` parameters refer to the **final optimization stage** for each channel's intended use case.

### Root Cause Analysis

**The real problem is channel optimization mismatch, not missing processing**:

1. **Channel 0 (ASR-optimized)**: 
   - Designed for speech recognition systems
   - May have aggressive AGC that distorts audio for transcription
   - Current code uses this channel: `pcm32.reshape(-1, 2)[:, 0]`

2. **Channel 1 (Communications-optimized)**:
   - Designed for voice communications (calls, conferencing)
   - May have aggressive NS that removes too much speech content
   - **User testing confirmed**: Channel 1 contains mostly silence

**Neither channel is optimized for Deepgram transcription specifically**

### XMOS XU316 Output Configuration

1. **Channel Mapping**: ✅ **CONFIRMED** - Both channels fully processed but optimized differently:
   - **Channel 0 (Left)**: ASR-optimized with final AGC tuning
   - **Channel 1 (Right)**: Communications-optimized with final NS tuning
   - **Real issue**: Optimization mismatch with Deepgram requirements

2. **32-bit Audio Format**: ✅ **CONFIRMED** - I2S 32-bit containers at 16kHz
   - Configuration shows: `sample_rate: 16000`, `bits_per_sample: 32bit`
   - Need to verify if upper 16 bits or full 32-bit range is used

3. **Sample Rate**: ✅ **CONFIRMED** - 16kHz throughout the pipeline
   - Voice PE uses 16kHz (not 48kHz as initially assumed)
   - No resampling needed in VoicePEAudioProcessor
   - AIC3204 likely configured for 16kHz operation

4. **Beamforming Output**: ✅ **CONFIRMED** (Official XMOS Documentation)
   - **Channel 0**: ASR-optimized with final AGC tuning
   - **Channel 1**: Communications-optimized with final NS tuning  
   - **Both channels**: Receive full XMOS pipeline (AEC, IC, VAD, NS, AGC)
   - **Key insight**: Neither channel optimized specifically for Deepgram transcription

### Remaining Questions for Voice PE Developer

**With XMOS pipeline now documented, key remaining questions:**

5. **Channel Optimization Details**: Can XMOS firmware be configured for Deepgram-specific optimization?
   - Are there alternative firmware builds available?
   - Can channel optimization parameters be adjusted?
   - Any "raw" or less-processed output modes?

6. **32-bit Audio Format**: Exact bit packing in I2S containers?
   - Full 32-bit dynamic range or padded 16/24-bit?
   - Signed integer format specifications?
   - Any endianness considerations?

7. **Quality Metrics for Transcription**: Expected characteristics for optimal Deepgram performance?
   - Target SNR levels after XMOS processing?
   - Dynamic range specifications?
   - Frequency response optimizations for speech?

8. **Alternative Access Methods**: Can we access audio before final channel optimization?
   - Raw post-beamforming but pre-channel-optimization?
   - Direct microphone array access?
   - Custom firmware compilation options?

## Debug Recommendations

### 1. Channel Comparison Test
Create test code to record both left and right channels separately:
```python
left_channel = pcm32.reshape(-1, 2)[:, 0]
right_channel = pcm32.reshape(-1, 2)[:, 1]
# Compare energy, correlation, frequency content
```

### 2. Bit Format Analysis
Test different bit extraction methods:
```python
# Current method (upper 16 bits)
method1 = (pcm32 >> 16).astype(np.int16)

# Full 32-bit range
method2 = (pcm32 / (2**15)).clip(-1, 1)

# Left-justified 24-bit
method3 = (pcm32 >> 8).astype(np.int32) >> 8
```

### 3. Audio Quality Metrics
Implement quality measurement:
- RMS energy levels
- Frequency spectrum analysis
- SNR estimation
- Clipping detection

### 4. Deepgram Format Testing
Test different audio formats with Deepgram:
- Mono vs stereo input
- 16-bit vs 32-bit samples
- Different sample rates

## SOLUTION BASED ON OFFICIAL XMOS DOCUMENTATION

### Root Cause: Channel Optimization Mismatch

**The low audio levels are caused by channel optimization mismatch and improper gain staging!**

**Channel 0 (ASR-optimized)**: May have conservative AGC settings resulting in low output levels
**Channel 1 (Communications-optimized)**: Confirmed to have very low audio levels, nearly silent

### Solution Approaches

#### 1. Intelligent Channel Combining (Recommended)

Use the `VoicePEAudioProcessor` class which:
- Analyzes both channels for voice content
- Combines channels based on correlation and voice energy
- Applies proper 32-bit audio handling
- Provides quality monitoring

```python
# In main.py, replace ESP32TCPServer with VoicePETCPServer
server = VoicePETCPServer(host="0.0.0.0", port=8989, enable_voice_pe=True)
```

#### 2. Channel Testing Protocol

Test both channels systematically:

```python
# Test Channel 0 (current default - ASR-optimized)
pcm32 = pcm32.reshape(-1, 2)[:, 0]  

# Test Channel 1 (communications-optimized) 
pcm32 = pcm32.reshape(-1, 2)[:, 1]  

# Test intelligent combination (VoicePEAudioProcessor)
# Automatically selects best channel or combines based on content
```

#### 3. Audio Level Boost Testing

Since audio is audible but very low, test gain adjustments:

```python
# Current processing (potentially too low gain)
pcm16 = (pcm32 >> 16).astype(np.int16)

# Test different bit extraction methods:
# Method 1: Use full 32-bit range (may boost levels)
audio_float = pcm32.astype(np.float64) / (2**31)  
pcm16 = (audio_float * 32767).astype(np.int16)

# Method 2: Apply additional gain
pcm16 = (pcm32 >> 16).astype(np.int16)
pcm16 = np.clip(pcm16 * 4, -32768, 32767)  # 4x gain boost

# Method 3: Automatic gain normalization
rms = np.sqrt(np.mean(pcm32.astype(np.float64)**2))
target_rms = 0.1 * (2**31)  # Target 10% of full scale
if rms > 0:
    gain = target_rms / rms
    pcm16 = np.clip(pcm32 * gain / (2**16), -32768, 32767).astype(np.int16)
```

#### 4. VoicePEAudioProcessor Advanced Gain Control

The `VoicePEAudioProcessor` includes automatic gain normalization specifically for this issue:
- Analyzes both channels for optimal levels
- Applies intelligent gain boosting
- Normalizes audio for Deepgram optimal range
- Prevents clipping while maximizing signal level

## Proposed Solutions

### Enhanced Audio Processor
The `VoicePEAudioProcessor` class addresses these issues:

1. **Intelligent Channel Combining**:
   ```python
   correlation = np.corrcoef(left_channel, right_channel)[0, 1]
   if correlation < 0.95:
       # Combine channels based on voice energy
   ```

2. **Proper 32-bit Handling**:
   ```python
   audio_normalized = audio_float / (2**31)  # Full 32-bit range
   ```

3. **Quality Monitoring**:
   - SNR tracking
   - Clipping detection
   - Processing metrics

### Testing Protocol

1. **Baseline Test**: Current ESP32TCPServer with problematic audio
2. **Enhanced Test**: VoicePEAudioProcessor with intelligent processing
3. **Comparison**: Deepgram transcription quality between methods
4. **Debugging**: Audio file capture and analysis

## Expected Outcomes (Updated Based on XMOS Documentation)

**If Audio Level Boost Resolves Low Volume Issue**: 
- Problem was insufficient gain staging after XMOS processing
- Simple gain adjustment or proper 32-bit handling resolves transcription issues
- **Most likely outcome** since speech is audible but quiet

**If VoicePEAudioProcessor Intelligent Processing Helps**: 
- Channel optimization mismatch combined with gain issues
- Intelligent combining + automatic gain normalization provides optimal audio
- Proper 32-bit handling with adaptive gain control

**If Channel 0 Works Better with Gain Boost**:
- ASR optimization works with Deepgram when levels are corrected
- Solution: continue using Channel 0 with proper gain staging

**If Audio Levels Remain Too Low**:
- May need XMOS firmware parameter adjustments
- Consider Voice PE developer consultation for AGC/gain settings
- Possible hardware gain adjustment requirements

---

*This document serves as a technical reference for debugging Voice PE audio quality issues and provides specific questions for hardware developers to resolve empty transcript problems.*