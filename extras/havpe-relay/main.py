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
import os
import signal
import time
from datetime import datetime

import numpy as np
import websockets
from easy_audio_interfaces import ResamplingBlock
from pathlib import Path
from wyoming.audio import AudioChunk
from easy_audio_interfaces.filesystem import LocalFileSink

DEFAULT_TCP_PORT = 8989
DEFAULT_WS_URL = "ws://host.docker.internal:8000/ws_pcm"

DEBUG_DIR = Path("./audio_chunks")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = os.getenv("DEBUG", "0") == "1"

# Chunk duration in seconds
CHUNK_DURATION_SECONDS = 10

# ESP32 audio format (from Voice-PE)
ESP32_SAMPLE_RATE = 16000
ESP32_CHANNELS = 2
ESP32_SAMPLE_WIDTH = 4  # 32-bit (4 bytes per sample)

# Target format for backend
TARGET_SAMPLE_WIDTH = 2  # 16-bit (2 bytes per sample)
TARGET_CHANNELS = 1  # Convert from stereo to mono


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

            converted_chunks = []
            async for converted_chunk in self.resampler.process_chunk(input_chunk):
                converted_chunks.append(converted_chunk)

            if converted_chunks:
                # The resampler maintains 32-bit format, so we need to manually convert to 16-bit
                if len(converted_chunks) > 1:
                    logging.warning(
                        f"WARNING: {len(converted_chunks)} converted chunks returned from resampler"
                    )
                converted_chunk = converted_chunks[0]
                return self._convert_32bit_to_16bit(converted_chunk.audio)
            else:
                logging.warning("No converted chunks returned from resampler")
                return b""

        except Exception as e:
            logging.error(f"Error converting audio: {e}")
            return b""

    def _convert_32bit_to_16bit(self, audio_32bit: bytes) -> bytes:
        """Convert 32-bit PCM data to 16-bit PCM data and stereo to mono using NumPy."""

        # Convert bytes to numpy array of int32 (little-endian)
        samples_32bit = np.frombuffer(audio_32bit, dtype=np.int32)

        # Convert int32 to int16 by right-shifting by 16 bits
        # This preserves the most significant bits while reducing bit depth
        samples_16bit = (samples_32bit >> 16).astype(np.int16)

        # Convert stereo to mono by averaging left and right channels
        if ESP32_CHANNELS == 2:
            # Reshape to separate channels (samples, channels)
            stereo_samples = samples_16bit.reshape(-1, 2)
            # Average left and right channels to create mono
            mono_samples = np.mean(stereo_samples, axis=1, dtype=np.int16)
            return mono_samples.tobytes()
        else:
            # Already mono or other channel configuration
            return samples_16bit.tobytes()

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
        
        # Audio sink management
        self.sink = None
        self.sink_converted = None
        self.chunk_start_time = None
        self.current_chunk_samples = 0
        self.chunk_counter = 0

    async def _create_new_sinks(self):
        """Create new audio sinks with timestamped filenames."""
        if self.sink:
            await self.sink.close()
        if self.sink_converted:
            await self.sink_converted.close()
            
        if DEBUG:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.chunk_counter += 1
            
            # Create new input sink (32-bit)
            input_filename = f"relay_input_{timestamp}_chunk{self.chunk_counter:03d}.wav"
            self.sink = LocalFileSink(
                DEBUG_DIR / input_filename,
                sample_rate=ESP32_SAMPLE_RATE,
                channels=ESP32_CHANNELS,
                sample_width=ESP32_SAMPLE_WIDTH,
            )
            await self.sink.open()
            
            # Create new converted sink (16-bit)
            output_filename = f"relay_output_{timestamp}_chunk{self.chunk_counter:03d}.wav"
            self.sink_converted = LocalFileSink(
                DEBUG_DIR / output_filename,
                sample_rate=ESP32_SAMPLE_RATE,
                channels=TARGET_CHANNELS,
                sample_width=TARGET_SAMPLE_WIDTH,
            )
            await self.sink_converted.open()
            
            logging.info(f"Created new audio chunk files: {input_filename}, {output_filename}")
        else:
            self.sink = None
            self.sink_converted = None
            
        # Reset chunk timing
        self.chunk_start_time = time.time()
        self.current_chunk_samples = 0

    async def _check_chunk_rotation(self, samples_written: int):
        """Check if we need to rotate to a new chunk file based on 10-second duration."""
        if not DEBUG:
            return
            
        self.current_chunk_samples += samples_written
        
        # Calculate elapsed time based on samples written
        elapsed_samples = self.current_chunk_samples
        elapsed_seconds = elapsed_samples / ESP32_SAMPLE_RATE
        
        if elapsed_seconds >= CHUNK_DURATION_SECONDS:
            logging.info(f"Rotating audio chunk after {elapsed_seconds:.2f} seconds")
            await self._create_new_sinks()

    async def handle_tcp_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        client_addr = writer.get_extra_info("peername")
        logging.info(f"TCP client connected from {client_addr}")

        # Initialize first chunk
        await self._create_new_sinks()

        try:
            # Initialize audio converter
            await self.audio_converter.open()

            # Add audio format parameters to WebSocket URL for the backend
            ws_url_with_params = (
                f"{self.ws_url}?user_id=havpe"
                # f"&rate={ESP32_SAMPLE_RATE}"
                # f"&width={TARGET_SAMPLE_WIDTH}"  # Tell backend we're sending 16-bit
                # f"&channels={ESP32_CHANNELS}"
                # f"&src=voice_pe"
            )
            logging.info(f"WebSocket URL with params: {ws_url_with_params}")

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
                            if self.sink:
                                chunk = AudioChunk(
                                    audio=data,
                                    rate=ESP32_SAMPLE_RATE,
                                    channels=ESP32_CHANNELS,
                                    width=ESP32_SAMPLE_WIDTH,
                                )
                                await self.sink.write(chunk)
                                
                                # Calculate samples written for rotation check
                                samples_in_chunk = len(data) // (ESP32_SAMPLE_WIDTH * ESP32_CHANNELS)
                                await self._check_chunk_rotation(samples_in_chunk)
                        except asyncio.TimeoutError:
                            continue  # Check shutdown event and running flag

                        if not data:
                            logging.info("TCP client disconnected")
                            break

                        # Convert audio format from 32-bit to 16-bit
                        converted_data = await self.audio_converter.convert_audio_chunk(
                            data
                        )

                        if converted_data:
                            # Forward converted data to WebSocket
                            await websocket.send(converted_data)
                            if self.sink_converted:
                                await self.sink_converted.write(
                                    AudioChunk(
                                        audio=converted_data,
                                        rate=ESP32_SAMPLE_RATE,
                                        channels=TARGET_CHANNELS,
                                        width=TARGET_SAMPLE_WIDTH,
                                    )
                                )
                            logging.debug(
                                f"Relayed {len(data)} bytes (32-bit) -> {len(converted_data)} bytes (16-bit) from TCP to WebSocket"
                            )
                        else:
                            logging.warning(
                                f"Failed to convert {len(data)} bytes of audio data"
                            )

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
            
            # Clean up audio sinks
            if self.sink:
                await self.sink.close()
            if self.sink_converted:
                await self.sink_converted.close()

            writer.close()
            await writer.wait_closed()
            logging.info(f"TCP client {client_addr} disconnected")

    async def start_server(self):
        logging.info(f"Starting TCP-to-WebSocket relay on port {self.tcp_port}")
        server = await asyncio.start_server(
            self.handle_tcp_client, "0.0.0.0", self.tcp_port
        )

        addr = server.sockets[0].getsockname()
        logging.info(f"TCP-to-WebSocket relay listening on {addr[0]}:{addr[1]}")
        logging.info(f"Will forward to WebSocket: {self.ws_url}")
        logging.info(
            f"Audio conversion: {ESP32_CHANNELS}-channel {ESP32_SAMPLE_WIDTH*8}-bit -> {TARGET_CHANNELS}-channel {TARGET_SAMPLE_WIDTH*8}-bit PCM"
        )

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
    # Get defaults from environment variables
    tcp_port_default = int(os.getenv("TCP_PORT", DEFAULT_TCP_PORT))
    ws_url_default = os.getenv("WS_URL", DEFAULT_WS_URL)

    parser = argparse.ArgumentParser(
        description="TCP-to-WebSocket relay for ESP32 audio with format conversion"
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=tcp_port_default,
        help=f"TCP port to listen on (default: {tcp_port_default}, env: TCP_PORT)",
    )
    parser.add_argument(
        "--ws-url",
        default=ws_url_default,
        help=f"WebSocket URL to forward to (default: {ws_url_default}, env: WS_URL)",
    )
    args = parser.parse_args()

    # Setup logging
    loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)s  %(message)s", level=loglevel
    )

    logging.info(f"TCP_PORT: {tcp_port_default}")
    logging.info(f"WS_URL: {ws_url_default}")

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
