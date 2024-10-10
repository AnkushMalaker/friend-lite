import opuslib
from easy_audio_interfaces.types import NumpyFrame


def decode_friend_message(message: bytes, decoder: opuslib.Decoder) -> NumpyFrame:
    opus_data = message[3:]
    pcm_data = decoder.decode(bytes(opus_data), frame_size=360)  # type: ignore
    return NumpyFrame.frombuffer(pcm_data)
