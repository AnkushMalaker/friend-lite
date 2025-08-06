The speaker recognition service will provide the following functionality:
- Enroll a speaker
- List enrolled speakers
- Remove a speaker
- Health check
- Identify speakers in audio segment

The speaker recognition service will be used to identify speakers in chunks of audio
The service will make use of SpeechBrain, FAISS, pytorch and pyannote to do speaker recognition.

## Processing Modes

The speaker recognition service supports multiple processing modes for different use cases:

### Mode Overview
1. **Diarization Only**: Pure speaker separation without identification
2. **Speaker Identification**: Diarization + identify enrolled speakers
3. **Deepgram Enhanced**: Deepgram transcription + diarization + enhanced speaker ID
4. **Deepgram + Internal Speakers**: Deepgram transcription + internal diarization + speaker ID
5. **Plain**: Legacy mode (same as Speaker Identification)

## Flow

### 1. Service Initialization Flow
1. **Environment Setup**
   - Load environment variables (HF_TOKEN, SIMILARITY_THRESHOLD)

2. **Model Loading**
   - Load pyannote speaker diarization pipeline (`pyannote/speaker-diarization-3.1`)
   - Load SpeechBrain speaker embedding model (`speechbrain/spkrec-ecapa-voxceleb`)
   - Initialize audio loader with 16kHz sample rate

3. **Database Initialization**
   - Create FAISS index for vector similarity search (IndexFlatIP with embedding dimension)
   - Initialize empty enrolled speakers list
   - Set up FastAPI application with health endpoints

### 2. Speaker Enrollment Flow
1. **Audio Input**
   - Receive audio file path or uploaded audio file
   - Optional: specify time segment (start_time, end_time)

2. **Audio Processing**
   - Load audio using pyannote Audio loader
   - Crop to specified segment if time bounds provided
   - Ensure proper format (16kHz, mono)

3. **Embedding Extraction**
   - Pass audio waveform through SpeechBrain embedding model
   - Apply L2 normalization for cosine similarity compatibility
   - Extract 512-dimensional speaker embedding vector

4. **Speaker Registration**
   - Check if speaker ID already exists (update if found)
   - Add new speaker entry with ID, name, and embedding
   - Add embedding to FAISS index for fast similarity search
   - Rebuild FAISS index if updating existing speaker

### 3. Speaker Identification Flow
1. **Audio Input**
   - Receive audio file path and optional time segment
   - Load and preprocess audio (same as enrollment)

2. **Embedding Extraction**
   - Extract speaker embedding using same model as enrollment
   - Apply L2 normalization

3. **Similarity Search**
   - Query FAISS index with extracted embedding
   - Find closest match using inner product (cosine similarity)
   - Compare similarity score against threshold (default: 0.85)

4. **Identity Resolution**
   - Return speaker ID and info if similarity > threshold
   - Return "not identified" if no match found

### 4. Processing Mode Flows

#### 4a. Diarization Only Flow (`/v1/diarize-only`)
1. **Audio Processing**
   - Run pyannote diarization pipeline on entire audio file
   - Extract speaker segments with timestamps
   - Apply minimum duration filter

2. **Result Formatting**
   - Return segments with generic speaker labels (SPEAKER_00, SPEAKER_01, etc.)
   - No speaker identification performed
   - No transcription provided

3. **Output**
   - Timestamped segments with generic speaker IDs
   - Summary statistics (duration, speaker count)
   - Processing metadata

#### 4b. Speaker Identification Flow (`/diarize-and-identify`)
1. **Audio Processing**
   - Run pyannote diarization pipeline on entire audio file
   - Extract speaker segments with timestamps
   - Apply minimum duration filter

2. **Speaker Identification**
   - For each detected speaker segment:
     - Extract audio segment from timestamps
     - Generate speaker embedding using SpeechBrain model
     - Query FAISS index for closest enrolled speaker match
     - Apply similarity threshold to determine identification

3. **Result Compilation**
   - Return segments with both diarization labels and identified speaker names
   - Include confidence scores for identification
   - Filter results based on identification requirements

#### 4c. Deepgram Enhanced Flow (`/v1/listen`)
1. **Deepgram Processing**
   - Forward audio to Deepgram API with diarization enabled
   - Receive transcription with speaker diarization
   - Extract word-level speaker assignments

2. **Speaker Enhancement**
   - Group consecutive words by Deepgram speaker labels
   - For each speaker segment:
     - Extract audio segment from timestamps
     - Generate embedding using internal model
     - Identify against enrolled speakers
     - Replace Deepgram speaker labels with identified names

3. **Response Enhancement**
   - Modify Deepgram response with identified speaker information
   - Add speaker enhancement metadata
   - Preserve original transcription quality

#### 4d. Deepgram + Internal Speakers Flow (`/v1/transcribe-and-diarize`)
1. **Deepgram Transcription**
   - Forward audio to Deepgram API for transcription only (diarization disabled)
   - Receive high-quality transcript without speaker labels

2. **Internal Diarization**
   - Run pyannote diarization pipeline on audio file
   - Extract speaker segments with timestamps
   - Apply minimum duration filter

3. **Speaker Identification**
   - For each diarized segment:
     - Extract audio segment
     - Generate embedding
     - Identify against enrolled speakers

4. **Hybrid Response**
   - Combine Deepgram transcript with internal speaker mapping
   - Provide both transcription and speaker identification
   - Return enhanced response with multiple data sources

### 5. Speaker Management Flow
1. **List Speakers**
   - Return all enrolled speakers with IDs and names
   - Provide count of enrolled speakers

2. **Remove Speaker**
   - Find speaker by ID in enrolled speakers list
   - Remove from speakers list
   - Rebuild FAISS index without removed speaker's embedding
   - Return success/failure status

### 6. Health Check Flow
1. **Service Status**
   - Check if all required models are loaded
   - Verify device availability (CPU/CUDA)
   - Report number of enrolled speakers
   - Return overall service health status

### Data Flow Architecture

#### Diarization Only Mode
```
Audio Input → Pyannote Diarization → Segment Extraction → Generic Labels → API Response
```

#### Speaker Identification Mode  
```
Audio Input → Pyannote Diarization → Segment Extraction → Embedding Model → FAISS Index
                                                                              ↓
                                         Enrolled Speakers Database ← Speaker Registration
                                                                              ↓
                                                        Similarity Search → Identity Resolution → API Response
```

#### Deepgram Enhanced Mode
```
Audio Input → Deepgram API (Transcription + Diarization) → Word Grouping → Embedding Model → FAISS Index
                                                                                              ↓
                                                             Enrolled Speakers Database ← Speaker Registration
                                                                                              ↓
                                                                        Similarity Search → Response Enhancement → API Response
```

#### Deepgram + Internal Speakers Mode
```
Audio Input → Deepgram API (Transcription Only) → Transcript
               ↓
           Pyannote Diarization → Segment Extraction → Embedding Model → FAISS Index
                                                                          ↓
                                        Enrolled Speakers Database ← Speaker Registration
                                                                          ↓
                                                Similarity Search → Hybrid Response → API Response
```

### Key Components
- **FAISS Index**: Fast similarity search for speaker embeddings
- **SpeechBrain Model**: Speaker embedding extraction
- **Pyannote Pipeline**: Speaker diarization and audio processing
- **Enrolled Speakers DB**: In-memory storage of registered speakers
- **Similarity Threshold**: Configurable threshold for speaker matching (default: 0.85)


