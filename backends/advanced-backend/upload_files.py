#!/usr/bin/env python3
"""
Upload audio files to the Friend-Lite backend for processing.
"""

import os
import sys
import requests
from pathlib import Path
from typing import Optional


def load_env_variables() -> Optional[str]:
    """Load ADMIN_PASSWORD from .env file."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Please create it with ADMIN_PASSWORD.")
        return None
    
    admin_password = None
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ADMIN_PASSWORD='):
                admin_password = line.split('=', 1)[1].strip('"\'')
                break
    
    if not admin_password:
        print("❌ ADMIN_PASSWORD not found in .env file.")
        return None
    
    return admin_password


def get_admin_token(password: str, base_url: str = "http://localhost:8000") -> Optional[str]:
    """Authenticate and get admin token."""
    print("🔑 Requesting admin token...")
    
    auth_url = f"{base_url}/auth/jwt/login"
    
    try:
        response = requests.post(
            auth_url,
            data={
                'username': 'admin@example.com',
                'password': password
            },
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            timeout=10
        )
        
        print(f"🔍 Auth response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            if token:
                print("✅ Admin token obtained.")
                return token
            else:
                print("❌ No access token in response.")
                print(f"Available fields: {list(data.keys())}")
                return None
        else:
            print(f"❌ Authentication failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def collect_wav_files(audio_dir: str, filter_list: Optional[list[str]] = None) -> list[str]:
    """Collect all .wav files from the specified directory."""
    print(f"📂 Collecting .wav files from {audio_dir} ...")
    
    audio_path = Path(audio_dir).expanduser()
    if not audio_path.exists():
        print(f"❌ Directory {audio_path} does not exist.")
        return []
    
    wav_files = list(audio_path.glob("*.wav"))
    
    if not wav_files:
        print(f"⚠️  No .wav files found in {audio_path}")
        return []
    
    # Filter files if filter_list is provided, otherwise accept all
    if filter_list is None:
        selected_files = wav_files
    else:
        selected_files = []
        for f in wav_files:
            if f.name in filter_list:
                selected_files.append(f)
            else:
                print(f"  ⏭️  Skipping file (not in filter): {f.name}")
    
    print(f"📦 Total files to upload: {len(selected_files)}")
    for file_path in selected_files:
        print(f"  ➕ Added file: {file_path}")
    
    return [str(f) for f in selected_files]


def upload_files(files: list[str], token: str, base_url: str = "http://localhost:8000") -> bool:
    """Upload files to the backend for processing."""
    if not files:
        print("❌ No files to upload.")
        return False
    
    print(f"🚀 Uploading files to {base_url}/api/process-audio-files ...")
    
    # Prepare files for upload
    files_data = []
    for file_path in files:
        try:
            files_data.append(('files', (os.path.basename(file_path), open(file_path, 'rb'), 'audio/wav')))
        except IOError as e:
            print(f"❌ Error opening file {file_path}: {e}")
            continue
    
    if not files_data:
        print("❌ No files could be opened for upload.")
        return False
    
    try:
        response = requests.post(
            f"{base_url}/api/process-audio-files",
            files=files_data,
            data={'device_name': 'file_upload_batch'},
            headers={
                'Authorization': f'Bearer {token}'
            },
            timeout=300  # 5 minutes timeout for large uploads
        )
        
        # Close all file handles
        for _, file_tuple in files_data:
            file_tuple[1].close()
        
        print(f"📤 Upload response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ File upload completed successfully.")
            try:
                result = response.json()
                print(f"📊 Response: {result}")
            except:
                print(f"📊 Response: {response.text}")
            return True
        else:
            print(f"❌ File upload failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Upload request timed out.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload request failed: {e}")
        return False
    finally:
        # Ensure all file handles are closed
        for _, file_tuple in files_data:
            try:
                file_tuple[1].close()
            except:
                pass


def main():
    """Main function to orchestrate the upload process."""
    print("🎵 Friend-Lite Audio File Upload Tool")
    print("=" * 40)
    
    # Load environment variables
    admin_password = load_env_variables()
    if not admin_password:
        sys.exit(1)
    
    # Get admin token
    token = get_admin_token(admin_password)
    if not token:
        sys.exit(1)
    # Test with the specific file the user mentioned
    specific_file = "none"
    
    # Check backends/advanced-backend/audio_chunks/ first
    backend_audio_dir = "./audio_chunks/"
    audio_dir_path = Path(backend_audio_dir)
    specific_file_path = audio_dir_path / specific_file
    
    if specific_file_path.exists():
        wav_files = [str(specific_file_path)]
        print(f"📦 Found specific test file: {specific_file_path}")
    else:
        # Fallback to original directory
        audio_dir = os.path.expanduser("~/Some dir/")
        # You can specify some test_files list if you want here
        wav_files = collect_wav_files(audio_dir, filter_list=None)
        if not wav_files:
            sys.exit(1)
    
    if not wav_files:
        print("❌ None of the test files were found")
        sys.exit(1)
    
    print(f"🧪 Testing with {len(wav_files)} files:")
    for f in wav_files:
        print(f"  - {os.path.basename(f)}")
    
    success = upload_files(wav_files, token)
    
    if success:
        print("\n🎉 Upload process completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Upload process failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 