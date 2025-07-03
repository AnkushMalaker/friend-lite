import argparse
import sys
import os
from diarization_service import enroll_speaker, load_enrolled_speakers
import traceback

def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker from an audio file.")
    parser.add_argument('--audio', required=True, help='Path to the audio file')
    parser.add_argument('--label', required=True, help='Speaker label/name')
    parser.add_argument('--id', help='Speaker ID (defaults to label)')
    parser.add_argument('--start', type=float, default=0, help='Start time (seconds) for enrollment segment')
    parser.add_argument('--end', type=float, default=30, help='End time (seconds) for enrollment segment')
    args = parser.parse_args()

    speaker_id = args.id if args.id else args.label
    speaker_name = args.label
    audio_file = args.audio
    start_time = args.start
    end_time = args.end

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)

    load_enrolled_speakers()  # Ensure we have the latest data
    try:
        success = enroll_speaker(speaker_id, speaker_name, audio_file, start_time, end_time)
        if success:
            print(f"Successfully enrolled speaker '{speaker_name}' (ID: {speaker_id}) from {audio_file}")
        else:
            print(f"Failed to enroll speaker '{speaker_name}' (ID: {speaker_id}) from {audio_file}")
            sys.exit(2)
    except Exception as e:
        print(f"Exception during enrollment: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main() 