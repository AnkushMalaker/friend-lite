import asyncio
import logging
import websockets
import websockets.exceptions

from easy_audio_interfaces.extras.local_audio import InputMicStream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# WebSocket URI for the /ws_pcm endpoint
WS_URI = "ws://localhost:8000/ws_pcm"


async def main():
    print(f"Connecting to {WS_URI}")
    
    try:
        async with websockets.connect(WS_URI) as websocket:
            print("Connected to WebSocket")
            
            async def send_audio():
                """Capture audio from microphone and send raw PCM bytes over WebSocket"""
                async with InputMicStream(chunk_size=512) as stream:
                    while True:
                        try:
                            data = await stream.read()
                            if data and data.raw_data:
                                # Send raw PCM bytes directly to WebSocket
                                await websocket.send(data.raw_data)
                                logger.debug(f"Sent audio chunk: {len(data.raw_data)} bytes")
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
        logger.error(f"Could not connect to {WS_URI}. Make sure the server is running.")
    except Exception as e:
        logger.error(f"Error connecting to WebSocket: {e}")


if __name__ == "__main__":
    asyncio.run(main())