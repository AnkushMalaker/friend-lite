#!/usr/bin/env python3
"""
Integration tests for the standalone Speaker Recognition service.

This test suite:
- Starts the speaker service via docker compose (service-only)
- Enrolls a speaker from a sample WAV
- Verifies the identify endpoint with the same audio (round-trip)
- Calls diarize-and-identify with a conversation WAV to ensure basic functionality

Requirements:
- HF_TOKEN must be set in the environment (pyannote models)
- Docker must be available

Run:
  uv run pytest extras/speaker-recognition/tests/test_speaker_service_integration.py -v -s
"""

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEAKER_DIR = REPO_ROOT / "extras" / "speaker-recognition"

# Prefer test compose variant/port; fall back to default if env forces it
COMPOSE_FILE = SPEAKER_DIR / os.environ.get(
    "SPEAKER_COMPOSE_FILE",
    "docker-compose-test.yml",
)

SPEAKER_SERVICE_PORT = int(os.environ.get("SPEAKER_SERVICE_TEST_PORT", os.environ.get("SPEAKER_SERVICE_PORT", "8086")))
SPEAKER_SERVICE_URL = f"http://localhost:{SPEAKER_SERVICE_PORT}"


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _wait_for_health(url: str, timeout_seconds: int = 300) -> None:
    start = time.time()
    last_error = None
    while time.time() - start < timeout_seconds:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return
            last_error = f"status={r.status_code} body={r.text[:200]}"
        except Exception as e:
            last_error = str(e)
        time.sleep(3)
    raise TimeoutError(f"Speaker service not healthy at {url} within {timeout_seconds}s: {last_error}")


def _compose_up():
    env = os.environ.copy()
    # Ensure port is exported so compose mapping aligns with localhost probing
    # For test compose, we export SPEAKER_SERVICE_TEST_PORT (host mapping)
    if COMPOSE_FILE.name.endswith("docker-compose-test.yml"):
        env.setdefault("SPEAKER_SERVICE_TEST_PORT", str(SPEAKER_SERVICE_PORT))
    else:
        env.setdefault("SPEAKER_SERVICE_PORT", str(SPEAKER_SERVICE_PORT))
    # Pass through HF_TOKEN and optional DEEPGRAM_API_KEY
    if "HF_TOKEN" not in env:
        pytest.skip("HF_TOKEN not set; skipping speaker recognition integration tests")
    if not _command_exists("docker"):
        pytest.skip("Docker not available; skipping speaker recognition integration tests")

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "up",
            "-d",
            "--build",
            # Service name varies by compose file
            "speaker-service-test" if COMPOSE_FILE.name.endswith("docker-compose-test.yml") else "speaker-service",
        ],
        cwd=str(SPEAKER_DIR),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _compose_down():
    if not _command_exists("docker"):
        return
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
            cwd=str(SPEAKER_DIR),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=False)
def speaker_service():
    try:
        _compose_up()
        _wait_for_health(SPEAKER_SERVICE_URL, timeout_seconds=600)
        yield
    finally:
        # Keep containers up if SPEAKER_TEST_KEEP_CONTAINERS=1
        if os.environ.get("SPEAKER_TEST_KEEP_CONTAINERS", "0") != "1":
            _compose_down()


def _find_sample(path_candidates: list[Path]) -> Path:
    for p in path_candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError("No suitable sample audio file found in expected locations")


def test_identify_roundtrip(speaker_service):
    sample = _find_sample(
        [
            SPEAKER_DIR / "sample-voice-ankush-anushpa.wav",
            SPEAKER_DIR / "test.wav",
        ]
    )

    files = {"file": (sample.name, open(sample, "rb"), "audio/wav")}

    # Enroll speaker for user 1
    enroll_data = {
        "speaker_id": "user_1_test_speaker",
        "speaker_name": "Test Speaker",
    }
    r = requests.post(f"{SPEAKER_SERVICE_URL}/enroll/upload", files=files, data=enroll_data, timeout=300)
    assert r.status_code == 200, f"Enroll failed: {r.status_code} {r.text[:500]}"
    resp = r.json()
    assert resp.get("speaker_id") == "user_1_test_speaker"

    # Identify using the same audio; expect a positive match
    files = {"file": (sample.name, open(sample, "rb"), "audio/wav")}
    identify_data = {
        "similarity_threshold": "0.10",
        "user_id": "1",
    }
    r = requests.post(f"{SPEAKER_SERVICE_URL}/identify", files=files, data=identify_data, timeout=300)
    assert r.status_code == 200, f"Identify failed: {r.status_code} {r.text[:500]}"
    result = r.json()
    assert result.get("found") is True
    assert result.get("speaker_id") == "user_1_test_speaker"
    assert isinstance(result.get("confidence"), (int, float))


def test_diarize_and_identify_basic(speaker_service):
    # Use a conversation-style sample if present; fall back to the same sample
    convo = _find_sample(
        [
            SPEAKER_DIR / "phone-hangout-group-recording-1-compressed.wav",
            SPEAKER_DIR / "sample-voice-ankush-anushpa.wav",
            SPEAKER_DIR / "test.wav",
        ]
    )

    files = {"file": (convo.name, open(convo, "rb"), "audio/wav")}
    params = {
        "user_id": "1",
        "min_duration": "0.5",
        "similarity_threshold": "0.10",
        "min_speakers": "1",
        "max_speakers": "4",
    }
    r = requests.post(f"{SPEAKER_SERVICE_URL}/diarize-and-identify", files=files, params=params, timeout=600)
    assert r.status_code == 200, f"diarize-and-identify failed: {r.status_code} {r.text[:500]}"
    result = r.json()
    # Basic structure assertions
    assert "segments" in result
    assert isinstance(result["segments"], list)
    # If segments are present, fields should exist
    if result["segments"]:
        seg = result["segments"][0]
        assert "start" in seg and "end" in seg and "speaker" in seg
        # Identified speaker fields may or may not be present depending on audio; that's fine


