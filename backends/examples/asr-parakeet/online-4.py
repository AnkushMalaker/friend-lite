#!/usr/bin/env python3
"""
Streaming ASR over WebSockets with NeMo-toolkit ≥ 2.3

* Model  : nvidia/parakeet-rnnt-1.1b  (BPE-RNNT, English)
* Server : ws://<host>:<port>/asr   (binary messages = OPUS frames)
* Output : JSON text messages
           {"type":"partial","text":"..."}  – interim
           {"type":"final",  "text":"..."}  – segment completed
           
Key 2.3 API differences handled
--------------------------------
1.  ``rnnt_greedy_decoding`` module path unchanged, but **stateful
    streaming helpers were promoted**; import them directly:
        >>> from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import (
                GreedyRNNTInfer)
2.  ``EncDecRNNTBPEModel.from_pretrained`` now **auto-downloads HF weights**
    and returns a model with *frozen* weights and
    ``model.preprocessor.featurizer`` moved under ``preprocessor``.
3.  Audio tensors **must be float32, normalised to ±1.0** – we convert
    the int16 PCM accordingly.

References: NeMo ASR API docs on *partial_hypothesis* for RNNT
streaming :contentReference[oaicite:0]{index=0}, NeMo real-time transcription
intro :contentReference[oaicite:1]{index=1}
"""
import asyncio
import json
from typing import Optional, Tuple

import numpy as np
import torch
import websockets

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import (
    GreedyRNNTInfer,
)

from omi.decoder import OmiOpusDecoder

opus_decoder = OmiOpusDecoder()

# ---------- configuration ----------
MODEL_NAME = "nvidia/parakeet-rnnt-1.1b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000          # parakeet-rnnt is 16 kHz
CHUNK_SEC = 2             # 320 ms chunks give <400 ms latency
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)
PORT = 8080
# -----------------------------------

# ----- load model once when script starts -----
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=MODEL_NAME)
asr_model.to(DEVICE)
asr_model.eval()
featurizer = asr_model.preprocessor
greedy_decoder = GreedyRNNTInfer(
    decoder_model=asr_model.decoder,
    joint_model=asr_model.joint,
    blank_index=asr_model.decoder.blank_idx,
    max_symbols_per_step=4,
)#.to(DEVICE)

# ----- helper: decode one chunk, keep decoder state between calls -----
@torch.no_grad()
def rnnt_stream_step(
    pcm_float: np.ndarray,
    prev_hyp: Optional[torch.Tensor] = None,
) -> Tuple[str, torch.Tensor]:
    """
    pcm_float : mono float32 [-1,1], shape (n,)
    Returns   : (partial_text, new_state, new_hyp)
    """
    # (1) feature extraction
    pcm_tensor = torch.from_numpy(pcm_float).unsqueeze(0).to(DEVICE)  # (1, T)
    feat, feat_len = featurizer(input_signal=pcm_tensor, length=torch.tensor([pcm_tensor.size(1)]).to(DEVICE))

    # (2) encoder
    enc_out, enc_len = asr_model.encoder(audio_signal=feat, length=feat_len)

    # (3) RNNT greedy step
    hyp_list = greedy_decoder(
        encoder_output=enc_out, encoded_lengths=enc_len,
        partial_hypotheses=[prev_hyp] if prev_hyp is not None else None,
    )
    new_hyp = hyp_list[0][0]                 # current partial Hypothesis object
    partial_text = asr_model.tokenizer.ids_to_text(new_hyp.y_sequence)
    print(f"partial_text: {partial_text}")

    return partial_text, new_hyp

# ------------- WebSocket handler ----------------
class ASRSession:
    """One client connection = one ASR stream."""
    def __init__(self, ws):
        self.ws = ws
        self.buffer = bytearray()
        self.state = None          # decoder state
        self.hyp = None            # partial hypothesis
        self.closed = False

    async def run(self):
        try:
            async for opus_bytes in self.ws:
                if isinstance(opus_bytes, str):
                    # ignore stray text frames
                    continue
                pcm_i16 = opus_decoder.decode_packet(opus_bytes, strip_header=False)
                self.buffer.extend(pcm_i16)

                while len(self.buffer) >= CHUNK_SAMPLES * 2:    # int16 → 2 bytes
                    chunk = np.frombuffer(self.buffer[:CHUNK_SAMPLES*2], dtype=np.int16)
                    del self.buffer[:CHUNK_SAMPLES*2]

                    pcm_float = (chunk.astype(np.float32) / 32768.0)
                    txt, self.hyp = rnnt_stream_step(pcm_float, self.hyp)
                    await self.ws.send(json.dumps({"type": "partial", "text": txt}))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # flush final hypothesis on close
            if self.hyp is not None:
                final_txt = asr_model.tokenizer.ids_to_text(self.hyp.y_sequence)
                try:
                    await self.ws.send(json.dumps({"type": "final", "text": final_txt}))
                except Exception:
                    pass
            self.closed = True

# ------------- server main ----------------------
async def handler(ws):
    session = ASRSession(ws)
    await session.run()

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8080,
                                max_size=2**23, ping_interval=30):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
