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

The reopen grace gate defaults to 1.5 seconds. Change it with

```dotenv
P_GPT_SPEAKER_REOPEN_GRACE_SECONDS=1.5
```

Values from 0.25 through 5 seconds are accepted. Increasing it makes short
pauses more likely to remain one user turn, but delays the earliest possible
TTS playback by the same amount.

## Persistent diagnostics

Backend application and speaker diagnostics are written to the terminal and to
`backend/logs/p-gpt.log`. The file rotates at 10 MiB and retains five backups by
default. These settings can be changed in `backend/.env`:

```dotenv
P_GPT_LOG_LEVEL=DEBUG
P_GPT_LOG_PATH=C:/path/to/p-gpt.log
P_GPT_LOG_MAX_BYTES=10485760
P_GPT_LOG_BACKUP_COUNT=5
```

`INFO` records pipeline timing and lifecycle milestones. `DEBUG` additionally
records control-state transitions, stale-generation drops, capture progress,
and one-second client VAD probability summaries. VAD diagnostics contain no
audio or transcript text. Restart the backend after changing the level.

The browser also writes speaker transport and VAD transitions to its developer
console with the `[speaker-session]` and `[speaker-vad]` prefixes. Persistent
copies of the privacy-safe VAD summaries are forwarded over the speaker socket
to the backend log when `DEBUG` is enabled.

For validation, run:

```powershell
cd backend
uv run --extra cpu python -m unittest discover -s tests

cd ..\frontend
pnpm.cmd test
pnpm.cmd build
pnpm.cmd lint
```
