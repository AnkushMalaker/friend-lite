"""
Test data for Robot Framework tests
"""


# Test Data
SAMPLE_CONVERSATIONS = [
    {
        "id": "conv_001",
        "transcript": "This is a test conversation about AI development.",
        "created_at": "2025-01-15T10:00:00Z"
    },
    {
        "id": "conv_002",
        "transcript": "Another test conversation discussing machine learning.",
        "created_at": "2025-01-15T11:00:00Z"
    }
]

SAMPLE_MEMORIES = [
    {
        "text": "User prefers AI discussions in the morning",
        "importance": 0.8
    },
    {
        "text": "User is interested in machine learning applications",
        "importance": 0.7
    }
]

TEST_AUDIO_FILE = "tests/test_assets/DIY_Experts_Glass_Blowing_16khz_mono_1min.wav"
TEST_DEVICE_NAME = "Robot-test-device"

# Expected content for transcript quality verification
EXPECTED_TRANSCRIPT = "glass blowing"
