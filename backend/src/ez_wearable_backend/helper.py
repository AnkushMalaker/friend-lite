import logging
from functools import partial

import opuslib
from easy_audio_interfaces.audio_interfaces import SocketReceiver
from ez_wearable_backend.utils import decode_friend_message

logger = logging.getLogger(__name__)


decoder = opuslib.Decoder(fs=16000, channels=1)
_post_process_callback = partial(decode_friend_message, decoder=decoder)


class FriendSocketReceiver(SocketReceiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, post_process_bytes_fn=_post_process_callback, **kwargs)
