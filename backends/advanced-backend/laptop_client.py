import argparse
import asyncio
import logging

import websockets
import websockets.exceptions
from easy_audio_interfaces.extras.local_audio import InputMicStream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default WebSocket settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_ENDPOINT = "/ws_pcm"


def build_websocket_uri(host: str, port: int, endpoint: str, user_id: str | None = None) -> str:
    """Build WebSocket URI with optional user_id parameter."""
    base_uri = f"ws://{host}:{port}{endpoint}"
    if user_id:
        base_uri += f"?user_id={user_id}"
    return base_uri


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Laptop audio client for OMI backend")
    parser.add_argument("--host", default=DEFAULT_HOST, help="WebSocket server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="WebSocket server port")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="WebSocket endpoint")
    parser.add_argument("--user-id", help="User ID for audio session (optional)")
    args = parser.parse_args()
    
    # Build WebSocket URI
    ws_uri = build_websocket_uri(args.host, args.port, args.endpoint, args.user_id)
    print(f"Connecting to {ws_uri}")
    if args.user_id:
        print(f"Using User ID: {args.user_id}")
    
    try:
        async with websockets.connect(ws_uri) as websocket:
            print("Connected to WebSocket")
            
            async def send_audio():
                """Capture audio from microphone and send raw PCM bytes over WebSocket"""
                async with InputMicStream(chunk_size=512) as stream:
                    while True:
                        try:
                            data = await stream.read()
                            if data and data.audio:
                                # Send raw PCM bytes directly to WebSocket
                                await websocket.send(data.audio)
                                logger.debug(f"Sent audio chunk: {len(data.audio)} bytes")
                            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                        except websockets.exceptions.ConnectionClosed:
                            logger.info("WebSocket connection closed during audio sending")
                            break
                        except Exception as e:
                            logger.error(f"Error sending audio: {e}")
                            break

            async def receive_messages():
                """Receive any messages from the WebSocket server"""
                try:
                    async for message in websocket:
                        print(f"Received message: {message}")
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed during message receiving")
                except Exception as e:
                    logger.error(f"Error receiving messages: {e}")

            # Run both audio sending and message receiving concurrently
            await asyncio.gather(send_audio(), receive_messages())
            
    except ConnectionRefusedError:
        logger.error(f"Could not connect to {ws_uri}. Make sure the server is running.")
    except Exception as e:
        logger.error(f"Error connecting to WebSocket: {e}")


if __name__ == "__main__":
    asyncio.run(main())