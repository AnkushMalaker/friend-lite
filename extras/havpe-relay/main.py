#!/usr/bin/env python3
"""
TCP-to-WebSocket Relay for ESPHome Voice-PE

• Listens on TCP port 8989 for ESP32 connections
• Converts 32-bit PCM to 16-bit PCM audio format using easy-audio-interfaces
• Forwards audio data to WebSocket /ws_pcm endpoint on port 8000
• Handles reconnections and audio format parameters
"""

import argparse
import asyncio
import logging
import signal
import time
from typing import Optional

import websockets
from easy_audio_interfaces import ResamplingBlock
from wyoming.audio import AudioChunk

DEFAULT_TCP_PORT = 8989
DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws_pcm"

# ESP32 audio format (from Voice-PE)
ESP32_SAMPLE_RATE = 16000
ESP32_CHANNELS = 2
ESP32_SAMPLE_WIDTH = 4  # 32-bit (4 bytes per sample)

# Target format for backend
TARGET_SAMPLE_WIDTH = 2  # 16-bit (2 bytes per sample)


class AudioConverter:
    """Handles audio format conversion from 32-bit to 16-bit PCM."""
    
    def __init__(self):
        # We'll use ResamplingBlock for format conversion
        # Even though sample rate stays the same, it handles bit depth conversion
        self.resampler = ResamplingBlock(resample_rate=ESP32_SAMPLE_RATE)
        
    async def convert_audio_chunk(self, audio_data: bytes) -> bytes:
        """Convert 32-bit PCM audio data to 16-bit PCM."""
        try:
            # Create AudioChunk from incoming 32-bit data
            input_chunk = AudioChunk(
                audio=audio_data,
                rate=ESP32_SAMPLE_RATE,
                width=ESP32_SAMPLE_WIDTH,
                channels=ESP32_CHANNELS,
            )
            
            # Process through resampler to convert format
            async def single_chunk_generator():
                yield input_chunk
            
            converted_chunks = []
            async for converted_chunk in self.resampler.process(single_chunk_generator()):
                converted_chunks.append(converted_chunk)
            
            if converted_chunks:
                # The resampler maintains 32-bit format, so we need to manually convert to 16-bit
                converted_chunk = converted_chunks[0]
                return self._convert_32bit_to_16bit(converted_chunk.audio)
            else:
                logging.warning("No converted chunks returned from resampler")
                return b""
                
        except Exception as e:
            logging.error(f"Error converting audio: {e}")
            return b""
    
    def _convert_32bit_to_16bit(self, audio_32bit: bytes) -> bytes:
        """Convert 32-bit PCM data to 16-bit PCM data."""
        import struct

        # Unpack 32-bit samples (little-endian float32)
        num_samples = len(audio_32bit) // 4
        samples_32bit = struct.unpack(f'<{num_samples}f', audio_32bit)
        
        # Convert to 16-bit integers (clamp to [-1, 1] range first)
        samples_16bit = []
        for sample in samples_32bit:
            # Clamp to [-1, 1] range
            clamped = max(-1.0, min(1.0, sample))
            # Convert to 16-bit integer
            sample_16bit = int(clamped * 32767)
            samples_16bit.append(sample_16bit)
        
        # Pack as 16-bit integers (little-endian)
        return struct.pack(f'<{len(samples_16bit)}h', *samples_16bit)
    
    async def open(self):
        """Initialize the audio converter."""
        await self.resampler.open()
    
    async def close(self):
        """Clean up the audio converter."""
        await self.resampler.close()


class TCPToWSRelay:
    def __init__(self, tcp_port: int, ws_url: str):
        self.tcp_port = tcp_port
        self.ws_url = ws_url
        self.running = True
        self.shutdown_event = asyncio.Event()
        self.audio_converter = AudioConverter()

    async def handle_tcp_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        client_addr = writer.get_extra_info("peername")
        logging.info(f"TCP client connected from {client_addr}")

        try:
            # Initialize audio converter
            await self.audio_converter.open()
            
            # Add audio format parameters to WebSocket URL for the backend
            ws_url_with_params = (
                f"{self.ws_url}?user_id=esp32_voice_pe"
                f"&rate={ESP32_SAMPLE_RATE}"
                f"&width={TARGET_SAMPLE_WIDTH}"  # Tell backend we're sending 16-bit
                f"&channels={ESP32_CHANNELS}"
                f"&src=voice_pe"
            )

            async with websockets.connect(
                ws_url_with_params, max_size=None
            ) as websocket:
                logging.info(f"WebSocket connected to {ws_url_with_params}")

                try:
                    while self.running and not self.shutdown_event.is_set():
                        # Read data from TCP client (ESP32) with timeout
                        try:
                            data = await asyncio.wait_for(
                                reader.read(4096), timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            continue  # Check shutdown event and running flag

                        if not data:
                            logging.info("TCP client disconnected")
                            break

                        # Convert audio format from 32-bit to 16-bit
                        converted_data = await self.audio_converter.convert_audio_chunk(data)
                        
                        if converted_data:
                            # Forward converted data to WebSocket
                            await websocket.send(converted_data)
                            logging.debug(
                                f"Relayed {len(data)} bytes (32-bit) -> {len(converted_data)} bytes (16-bit) from TCP to WebSocket"
                            )
                        else:
                            logging.warning(f"Failed to convert {len(data)} bytes of audio data")

                except websockets.exceptions.ConnectionClosed:
                    logging.warning("WebSocket connection closed")
                except asyncio.CancelledError:
                    logging.info("TCP client handler cancelled")
                    return
                except Exception as e:
                    logging.error(f"Error in relay loop: {e}")

        except websockets.exceptions.InvalidURI:
            logging.error(f"Invalid WebSocket URL: {ws_url_with_params}")
        except ConnectionRefusedError:
            logging.error(f"Could not connect to WebSocket server at {self.ws_url}")
        except Exception as e:
            logging.error(f"Error connecting to WebSocket: {e}")
        finally:
            # Clean up audio converter
            await self.audio_converter.close()
            
            writer.close()
            await writer.wait_closed()
            logging.info(f"TCP client {client_addr} disconnected")

    async def start_server(self):
        server = await asyncio.start_server(
            self.handle_tcp_client, "0.0.0.0", self.tcp_port
        )

        addr = server.sockets[0].getsockname()
        logging.info(f"TCP-to-WebSocket relay listening on {addr[0]}:{addr[1]}")
        logging.info(f"Will forward to WebSocket: {self.ws_url}")
        logging.info(f"Audio conversion: {ESP32_SAMPLE_WIDTH*8}-bit -> {TARGET_SAMPLE_WIDTH*8}-bit PCM")

        try:
            async with server:
                # Wait for shutdown event instead of serve_forever
                await self.shutdown_event.wait()
        finally:
            logging.info("Server shutting down...")

    def stop(self):
        self.running = False
        self.shutdown_event.set()


async def main():
    parser = argparse.ArgumentParser(
        description="TCP-to-WebSocket relay for ESP32 audio with format conversion"
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=DEFAULT_TCP_PORT,
        help=f"TCP port to listen on (default: {DEFAULT_TCP_PORT})",
    )
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        help=f"WebSocket URL to forward to (default: {DEFAULT_WS_URL})",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG"
    )
    args = parser.parse_args()

    # Setup logging
    loglevel = logging.WARNING - (10 * min(args.verbose, 2))
    logging.basicConfig(
        format="%(asctime)s  %(levelname)s  %(message)s", level=loglevel
    )

    # Create relay
    relay = TCPToWSRelay(args.tcp_port, args.ws_url)

    # Handle graceful shutdown
    def signal_handler():
        logging.info("Received shutdown signal")
        relay.stop()

    # Use asyncio signal handling for better integration
    loop = asyncio.get_running_loop()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, signal_handler)

    try:
        await relay.start_server()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down...")
        relay.stop()
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        logging.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
