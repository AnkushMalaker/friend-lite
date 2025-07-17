#!/usr/bin/env python3
"""
Upload audio files to the Friend-Lite backend for processing.
"""

import logging
import os
import sys
import requests
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_env_variables() -> Optional[str]:
    """Load ADMIN_PASSWORD from .env file."""
    env_file = Path(".env")
    if not env_file.exists():
        logger.error(".env file not found. Please create it with ADMIN_PASSWORD.")
        return None
    
    admin_password = None
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ADMIN_PASSWORD='):
                admin_password = line.split('=', 1)[1].strip('"\'')
                break
    
    if not admin_password:
        logger.error("ADMIN_PASSWORD not found in .env file.")
        return None
    
    return admin_password


def get_admin_token(password: str, base_url: str = "http://localhost:8000") -> Optional[str]:
    """Authenticate and get admin token."""
    logger.info("Requesting admin token...")
    
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
        
        logger.info(f"Auth response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            if token:
                logger.info("Admin token obtained.")
                return token
            else:
                logger.error("No access token in response.")
                logger.error(f"Available fields: {list(data.keys())}")
                return None
        else:
            logger.error(f"Authentication failed with status {response.status_code}")
            try:
                error_data = response.json()
                logger.error(f"Error details: {error_data}")
            except Exception as json_error:
                logger.error(f"Failed to parse error response as JSON: {json_error}")
                logger.error(f"Response text: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


def collect_wav_files(audio_dir: str, filter_list: Optional[list[str]] = None) -> list[str]:
    """Collect all .wav files from the specified directory."""
    logger.info(f"Collecting .wav files from {audio_dir} ...")
    
    audio_path = Path(audio_dir).expanduser()
    if not audio_path.exists():
        logger.error(f"Directory {audio_path} does not exist.")
        return []
    
    wav_files = list(audio_path.glob("*.wav"))
    
    if not wav_files:
        logger.warning(f"No .wav files found in {audio_path}")
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
                logger.info(f"Skipping file (not in filter): {f.name}")
    
    logger.info(f"Total files to upload: {len(selected_files)}")
    for file_path in selected_files:
        logger.info(f"Added file: {file_path}")
    
    return [str(f) for f in selected_files]


def upload_files(files: list[str], token: str, base_url: str = "http://localhost:8000") -> bool:
    """Upload files to the backend for processing."""
    if not files:
        logger.error("No files to upload.")
        return False
    
    logger.info(f"Uploading files to {base_url}/api/process-audio-files ...")
    
    # Prepare files for upload
    files_data = []
    for file_path in files:
        try:
            files_data.append(('files', (os.path.basename(file_path), open(file_path, 'rb'), 'audio/wav')))
        except IOError as e:
            logger.error(f"Error opening file {file_path}: {e}")
            continue
    
    if not files_data:
        logger.error("No files could be opened for upload.")
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
        
        logger.info(f"Upload response status: {response.status_code}")
        
        if response.status_code == 200:
            logger.info("File upload completed successfully.")
            try:
                result = response.json()
                logger.info(f"Response: {result}")
            except Exception as json_error:
                logger.error(f"Failed to parse success response as JSON: {json_error}")
                logger.info(f"Response: {response.text}")
            return True
        else:
            logger.error(f"File upload failed with status {response.status_code}")
            try:
                error_data = response.json()
                logger.error(f"Error details: {error_data}")
            except Exception as json_error:
                logger.error(f"Failed to parse error response as JSON: {json_error}")
                logger.error(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Upload request timed out.")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Upload request failed: {e}")
        return False
    finally:
        # Ensure all file handles are closed
        for _, file_tuple in files_data:
            try:
                file_tuple[1].close()
            except Exception as close_error:
                logger.warning(f"Failed to close file handle: {close_error}")


def main():
    """Main function to orchestrate the upload process."""
    logger.info("Friend-Lite Audio File Upload Tool")
    logger.info("=" * 40)
    
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
        logger.info(f"Found specific test file: {specific_file_path}")
    else:
        # Fallback to original directory
        audio_dir = os.path.expanduser("~/Some dir/")
        # You can specify some test_files list if you want here
        wav_files = collect_wav_files(audio_dir, filter_list=None)
        if not wav_files:
            sys.exit(1)
    
    if not wav_files:
        logger.error("None of the test files were found")
        sys.exit(1)
    
    logger.info(f"Testing with {len(wav_files)} files:")
    for f in wav_files:
        logger.info(f"- {os.path.basename(f)}")
    
    success = upload_files(wav_files, token)
    
    if success:
        logger.info("Upload process completed successfully!")
        sys.exit(0)
    else:
        logger.error("Upload process failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 