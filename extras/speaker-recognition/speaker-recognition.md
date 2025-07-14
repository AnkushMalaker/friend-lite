The speaker recognition service will provide the following functionality:
- Enroll a speaker
- List enrolled speakers
- Remove a speaker
- Health check
- Identify speakers in audio segment

The speaker recognition service will be used to identify speakers in chunks of audio
The service will make use of SpeechBrain, FAISS, pytorch and pyannote to do speaker recognition.

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

### 4. Speaker Diarization Flow
1. **Audio Processing**
   - Run pyannote diarization pipeline on entire audio file
   - Extract speaker segments with timestamps

2. **Speaker Segmentation**
   - Identify distinct speakers (SPEAKER_00, SPEAKER_01, etc.)
   - Get temporal boundaries for each speaker's speech

3. **Speaker Verification**
   - For each detected speaker:
     - Find longest speech segment for that speaker
     - Extract embedding from longest segment
     - Attempt to identify against enrolled speakers
     - Assign verified speaker ID or generate unknown speaker ID

4. **Result Compilation**
   - Return segments with timestamps and speaker assignments
   - Include both diarization labels and verified speaker IDs
   - Provide speaker embeddings for further processing

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
```
Audio Input → Audio Loader → Embedding Model → FAISS Index
                                ↓
Enrolled Speakers Database ← Speaker Registration
                                ↓
Similarity Search → Identity Resolution → API Response
```

### Key Components
- **FAISS Index**: Fast similarity search for speaker embeddings
- **SpeechBrain Model**: Speaker embedding extraction
- **Pyannote Pipeline**: Speaker diarization and audio processing
- **Enrolled Speakers DB**: In-memory storage of registered speakers
- **Similarity Threshold**: Configurable threshold for speaker matching (default: 0.85)


