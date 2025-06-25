# Laptop Client Usage Guide

## Basic Usage (without User ID)
```bash
python laptop_client.py
```
This connects to `ws://localhost:8000/ws_pcm` using a random client ID.

## Usage with User ID
```bash
python laptop_client.py --user-id john_doe
```
This connects to `ws://localhost:8000/ws_pcm?user_id=john_doe` and associates audio with the user "john_doe".

## Advanced Options
```bash
python laptop_client.py --host 192.168.1.100 --port 8001 --endpoint /ws --user-id alice
```
This connects to `ws://192.168.1.100:8001/ws?user_id=alice`.

## Command Line Arguments
- `--host`: WebSocket server host (default: localhost)
- `--port`: WebSocket server port (default: 8000)  
- `--endpoint`: WebSocket endpoint (default: /ws_pcm)
- `--user-id`: User ID for audio session (optional)

## Examples

### Test with different users:
```bash
# Terminal 1 - User Alice
python laptop_client.py --user-id alice

# Terminal 2 - User Bob  
python laptop_client.py --user-id bob
```

### Connect to remote server:
```bash
python laptop_client.py --host your-server.com --user-id remote_user
```

## Backward Compatibility
The client works exactly as before when no `--user-id` is provided, maintaining full backward compatibility with existing setups. 