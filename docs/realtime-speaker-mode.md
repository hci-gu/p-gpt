# Realtime speaker mode

Speaker mode connects the browser to `GET /speaker/v1` with the
`p-gpt-speaker.v1` WebSocket subprotocol. Control events are JSON and audio is
raw mono PCM16: 16 kHz input frames and 24 kHz output chunks.

The EN/SV language preference is shared by text dictation and speaker mode. In
speaker mode it explicitly selects one of two endpoint ASR routes:

- `en` uses `nvidia/parakeet-tdt-0.6b-v3` through the NeMo-free
  `nano-parakeet` PyTorch runtime.
- `sv` uses the `standard` revision of `KBLab/kb-whisper-medium` through the
  Transformers speech-recognition pipeline. Utterances are processed in
  30-second chunks with a five-second overlap so the 60-second speaker limit is
  supported without truncation.

Both models load eagerly and remain resident. CUDA uses bfloat16 for
KB-Whisper when supported and float16 otherwise; CPU float32 remains a
development fallback. Startup transcribes `backend/assets/default-voice.mp3`
through both routes as a real inference warmup. A model load or inference error
therefore prevents startup instead of silently routing Swedish audio through a
different model.

The route settings are:

```dotenv
P_GPT_SPEAKER_ASR_MODEL=nvidia/parakeet-tdt-0.6b-v3
P_GPT_SPEAKER_ASR_MODEL_SV=KBLab/kb-whisper-medium
P_GPT_SPEAKER_ASR_REVISION_SV=standard
```

Parakeet TDT 0.6B v3 is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its
[model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) documents the
supported languages, intended use, and training data. The realtime architecture
is inspired by Hugging Face's
[speech-to-speech project](https://github.com/huggingface/speech-to-speech), but
P-GPT uses its own versioned protocol and does not depend on that package.

KB-Whisper Medium is distributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Its
[model card](https://huggingface.co/KBLab/kb-whisper-medium) documents its
Swedish training data, transcription revisions, evaluation, and supported
Transformers inference path.

Inference is connection-safe but intentionally serialized: all speaker sessions
share one ASR lock and the existing OmniVoice lock. A stale request waiting
for a lock is discarded before output; a PyTorch call already running cannot be
safely interrupted and retains its lock until it returns.

The reopen grace gate defaults to 1 second. Change it with

```dotenv
P_GPT_SPEAKER_REOPEN_GRACE_SECONDS=1.0
```

Values from 0.25 through 5 seconds are accepted. Increasing it makes short
pauses more likely to remain one user turn, but delays the earliest possible
TTS playback by the same amount. With the default 384 ms VAD soft-end, the
default minimum last-speech-to-commit delay is approximately 1.384 seconds.

## Client VAD tuning

Speaker capture uses 32 ms frames. By default, a turn begins when four frames
within a six-frame activation attempt reach a speech probability of 0.40. Once
active, the lower 0.30 continuation threshold keeps quiet syllables from
starting the silence counter. Twelve quiet frames (384 ms) soft-end the turn,
and the preceding 24 frames (768 ms) are retained as pre-roll.

These build-time settings can be placed in the frontend environment:

```dotenv
VITE_SPEAKER_VAD_START_THRESHOLD=0.40
VITE_SPEAKER_VAD_START_REQUIRED_FRAMES=4
VITE_SPEAKER_VAD_START_WINDOW_FRAMES=6
VITE_SPEAKER_VAD_CONTINUE_THRESHOLD=0.30
VITE_SPEAKER_VAD_PRE_ROLL_FRAMES=24
VITE_SPEAKER_VAD_SOFT_END_FRAMES=12
VITE_SPEAKER_VAD_BARGE_THRESHOLD=0.40
VITE_SPEAKER_VAD_BARGE_REQUIRED_FRAMES=4
VITE_SPEAKER_VAD_BARGE_WINDOW_FRAMES=6
```

The barge-in profile is independent so interruption sensitivity can be tuned
without changing ordinary speech starts. Invalid values or a required-frame
count larger than its activation window use the documented defaults and write
a browser warning. Vite embeds these settings at build time, so changing them
requires rebuilding the frontend. Changing the backend grace value requires a
backend restart.

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

`INFO` records pipeline timing and lifecycle milestones. It also warns when a
speaker TTS segment takes more than five seconds or synthesis is slower than
the produced audio in real time. `DEBUG` additionally records control-state
transitions, stale-generation drops, capture epochs and recoveries, effective
VAD configuration, candidate decisions, worker backlog and processing latency,
one-second VAD probability summaries, TTS lock/inference/encoding timing,
transcript commit delivery timing, and browser playback scheduling summaries.
Playback summaries include PCM chunk counts, minimum scheduling lead, and
underrun counts for each segment. Transcript diagnostics record only timing and
generation metadata, never transcript text. VAD and TTS diagnostics contain no
audio or transcript text. Restart the backend after changing the level.

The browser also writes speaker transport, PCM chunk scheduling, transcript
render timing, and VAD transitions to its developer console with the
`[speaker-session]`, `[speaker-playback]`, and `[speaker-vad]` prefixes.
Persistent copies of privacy-safe VAD, playback, underrun, and transcript
render-timing summaries are forwarded over the speaker socket to the backend
log when `DEBUG` is enabled.

An isolated capture or VAD-worker stall is recovered automatically. Three
stalls within 60 seconds mute capture and show an error instead of repeatedly
restarting. Manually unmuting begins a fresh recovery window.

For validation, run:

```powershell
cd backend
uv run --extra cpu python -m unittest discover -s tests

cd ..\frontend
pnpm.cmd test
pnpm.cmd build
pnpm.cmd lint
```
