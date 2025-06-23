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

import websockets
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
ESP32_CHANNELS = 1
ESP32_SAMPLE_WIDTH = 2  # 16-bit (2 bytes per sample)


class TCPToWSRelay:
    def __init__(self, tcp_port: int, ws_url: str):
        self.tcp_port = tcp_port
        self.ws_url = ws_url
        self.running = True
        self.shutdown_event = asyncio.Event()
        
        # Audio sink management
        self.sink = None
        self.sink_converted = None  # No longer used but kept for compatibility
        self.chunk_start_time = None
        self.current_chunk_samples = 0
        self.chunk_counter = 0

    async def _create_new_sinks(self):
        """Create new audio sinks with timestamped filenames."""
        if self.sink:
            await self.sink.close()
        # if self.sink_converted:
        #     await self.sink_converted.close()
            
        if DEBUG:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.chunk_counter += 1
            
            # Create new input sink (16-bit - no conversion needed)
            input_filename = f"relay_input_{timestamp}_chunk{self.chunk_counter:03d}.wav"
            self.sink = LocalFileSink(
                DEBUG_DIR / input_filename,
                sample_rate=ESP32_SAMPLE_RATE,
                channels=ESP32_CHANNELS,
                sample_width=ESP32_SAMPLE_WIDTH,
            )
            await self.sink.open()
            
            logging.info(f"Created new audio chunk file: {input_filename}")
        else:
            self.sink = None
            self.sink_converted = None  # No longer used
            
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
            ws_url_with_params = f"{self.ws_url}?user_id=havpe"
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

                        # Forward audio data directly to WebSocket (no conversion needed)
                        await websocket.send(data)
                        logging.debug(f"Relayed {len(data)} bytes directly from TCP to WebSocket")


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
            # Clean up audio sinks
            if self.sink:
                await self.sink.close()

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

        logging.info(f"Audio format: {ESP32_CHANNELS}-channel {ESP32_SAMPLE_WIDTH*8}-bit PCM (direct forwarding)")

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
