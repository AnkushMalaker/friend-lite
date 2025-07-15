import argparse
import asyncio
import json
import logging

import aiohttp
import websockets
import websockets.exceptions
from easy_audio_interfaces.extras.local_audio import InputMicStream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default WebSocket settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_ENDPOINT = "/ws_pcm"


def build_websocket_uri(
    host: str, port: int, endpoint: str, token: str | None = None, device_name: str = "laptop"
) -> str:
    """Build WebSocket URI with JWT token authentication."""
    base_uri = f"ws://{host}:{port}{endpoint}"
    params = []
    if token:
        params.append(f"token={token}")
    if device_name:
        params.append(f"device_name={device_name}")

    if params:
        base_uri += "?" + "&".join(params)
    return base_uri


async def authenticate_with_credentials(host: str, port: int, username: str, password: str) -> str:
    """Authenticate with username/password and return JWT token."""
    auth_url = f"http://{host}:{port}/auth/jwt/login"

    # Prepare form data for authentication
    form_data = aiohttp.FormData()
    form_data.add_field("username", username)
    form_data.add_field("password", password)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    token = result.get("access_token")
                    if token:
                        logger.info(f"Successfully authenticated user '{username}'")
                        return token
                    else:
                        raise Exception("No access token received from server")
                elif response.status == 400:
                    error_detail = await response.text()
                    raise Exception(f"Authentication failed: Invalid credentials - {error_detail}")
                else:
                    error_detail = await response.text()
                    raise Exception(
                        f"Authentication failed with status {response.status}: {error_detail}"
                    )
    except aiohttp.ClientError as e:
        raise Exception(f"Failed to connect to authentication server: {e}")


def validate_auth_args(args):
    """Validate that exactly one authentication method is provided."""
    has_token = bool(args.token)
    has_credentials = bool(args.username and args.password)

    if not has_token and not has_credentials:
        raise ValueError(
            "Authentication required: Please provide either --token OR both --username and --password"
        )

    if has_token and has_credentials:
        raise ValueError(
            "Conflicting authentication methods: Please provide either --token OR --username/--password, not both"
        )

    if args.username and not args.password:
        raise ValueError(
            "Username provided but password missing: Both --username and --password are required"
        )

    if args.password and not args.username:
        raise ValueError(
            "Password provided but username missing: Both --username and --password are required"
        )


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Laptop audio client for OMI backend with dual authentication modes"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="WebSocket server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="WebSocket server port")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="WebSocket endpoint")

    # Authentication options (mutually exclusive)
    auth_group = parser.add_argument_group("authentication", "Choose one authentication method")
    auth_group.add_argument("--token", help="JWT authentication token")
    auth_group.add_argument("--username", help="Username for login authentication")
    auth_group.add_argument("--password", help="Password for login authentication")

    parser.add_argument(
        "--device-name", default="laptop", help="Device name for client identification"
    )
    args = parser.parse_args()

    # Validate authentication arguments
    try:
        validate_auth_args(args)
    except ValueError as e:
        logger.error(f"Authentication error: {e}")
        parser.print_help()
        return

    # Get or obtain authentication token
    token = None

    if args.token:
        # Use provided token directly
        token = args.token
        print(
            f"Using provided JWT token: {token[:20]}...{token[-10:] if len(token) > 30 else token}"
        )

    elif args.username and args.password:
        # Authenticate with username/password to get token
        print(f"Authenticating with username: {args.username}")
        try:
            token = await authenticate_with_credentials(
                args.host, args.port, args.username, args.password
            )
            print(
                f"Authentication successful! Received token: {token[:20]}...{token[-10:] if len(token) > 30 else token}"
            )
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return

    # Build WebSocket URI
    ws_uri = build_websocket_uri(args.host, args.port, args.endpoint, token, args.device_name)
    print(f"Connecting to {ws_uri}")
    print(f"Using device name: {args.device_name}")

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
