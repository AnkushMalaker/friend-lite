Use pyenv to install python 3.12.5
activate that python version
pip install --upgrade pip
pip install poetry
poetry install
poetry shell
poetry run python examples/streaming_vad.py


Cool, so now there are a few ways to improve what we have. 

Latency.
ASR Quality.
Reliability.

Number 1 is to improve reliability
Number 2 is to fix the stupid noise bug somewhere in the code. Try to save the audio file at different places to find it. 
Number 3 is to finetune the asr model with lora and recording audio etc.
Number 4 is to improve latency

I think Reliability is the most important one, without which others won't matter.
