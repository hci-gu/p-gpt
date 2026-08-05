# Realtime speaker mode

Speaker mode connects the browser to `GET /speaker/v1` with the
`p-gpt-speaker.v1` WebSocket subprotocol. Control events are JSON and audio is
raw mono PCM16: 16 kHz input frames and 24 kHz output chunks.

The backend eagerly loads `nvidia/parakeet-tdt-0.6b-v3` through the
NeMo-free `nano-parakeet` PyTorch runtime at startup. Override
the checkpoint with `P_GPT_SPEAKER_ASR_MODEL`. CUDA is selected when available,
with a float32 CPU fallback. Startup also transcribes `backend/assets/default-voice.mp3`
as a real inference warmup, so the first speaker session does not pay model
initialization cost and load/inference failures surface before the app is ready.

Parakeet TDT 0.6B v3 is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its
[model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) documents the
supported languages, intended use, and training data. The realtime architecture
is inspired by Hugging Face's
[speech-to-speech project](https://github.com/huggingface/speech-to-speech), but
P-GPT uses its own versioned protocol and does not depend on that package.

Inference is connection-safe but intentionally serialized: all speaker sessions
share one Parakeet lock and the existing OmniVoice lock. A stale request waiting
for a lock is discarded before output; a PyTorch call already running cannot be
safely interrupted and retains its lock until it returns.

For validation, run:

```powershell
cd backend
uv run --extra cpu python -m unittest discover -s tests

cd ..\frontend
pnpm.cmd test
pnpm.cmd build
pnpm.cmd lint
```
