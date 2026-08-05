# Realtime Speaker Mode Prototype

## Summary

Implement a dedicated realtime speaker mode while leaving the existing HTTP text-chat path unchanged.

Pipeline:

`microphone → 32 ms frames → client VAD → binary WebSocket PCM → endpoint ASR → speculative LLM → grace gate → sentence TTS → binary PCM playback`

Defaults:

- Input: mono PCM16 little-endian, 16 kHz, 512 samples/1,024 bytes per 32 ms frame.
- VAD threshold: `0.5`.
- Speech start: 5 consecutive positive frames, or 160 ms.
- Pre-roll: 16 frames, or 512 ms.
- Soft-end: 8 consecutive silent frames, or 256 ms.
- Reopen grace: 1,000 ms after soft-end.
- Barge-in: 160 ms confirmed speech.
- Maximum utterance: 60 seconds, approximately 1.9 MB.
- ASR: `nvidia/parakeet-tdt-0.6b-v3`, endpoint transcription only.
- TTS: one complete sentence at a time, with a 180-character word-boundary fallback.
- Output: mono PCM16 little-endian at OmniVoice’s 24 kHz rate, transported in 100 ms WebSocket chunks.
- Capture remains on the existing `ScriptProcessor` implementation for this prototype.
- Speaker mode displays the latest exchange; the user transcript appears after turn commitment and assistant text appears after playback completes.

## Protocol and Turn Semantics

### WebSocket interface

Add `/speaker/v1` using the `p-gpt-speaker.v1` WebSocket subprotocol.

JSON control messages use a versioned envelope containing `type`, `eventId`, and relevant `sessionId`, `turnId`, `turnRevision`, and `responseGeneration` fields. Audio uses binary WebSocket messages.

Client events:

- `session.configure`: persona identity/instructions, existing role/content history, LLM/TTS settings, and fixed input/output audio formats.
- `input.speech_candidate`: first positive frame during grace; temporarily holds the TTS gate.
- `input.speech_candidate_cancelled`: candidate failed to reach the 160 ms confirmation threshold.
- `input.speech_started`: confirmed speech and whether it is a reopened turn or barge-in.
- Binary input frames: exactly 1,024 bytes of 16 kHz PCM16 while capturing.
- `input.speech_soft_ended`: emitted after 256 ms silence.
- `input.limit_reached`: forced finalization at 60 seconds, ignoring further audio until silence rearms VAD.
- `response.cancel`: explicit mode exit, mute/reset, or client failure.
- `playback.segment_completed`: acknowledges a fully played TTS sentence.
- `playback.response_completed`: emitted after all scheduled audio actually finishes in the browser.

Server events:

- `session.ready`.
- `input.transcription.committed` or `input.transcription.empty`.
- `response.started`.
- `response.audio.segment_started`: includes generation, segment ID, text, sample rate, and encoding.
- Binary output chunks belonging to the current segment.
- `response.audio.segment_done` and `response.audio.done`.
- `response.completed` or `response.cancelled`.
- Typed recoverable/fatal `error` events.

WebSocket ordering associates binary output with the most recent `segment_started`; a single outbound send lock prevents interleaving. Both client and server validate generation metadata and discard stale audio.

### Turn lifecycle

- Before initial speech, the worker appends every frame to the 16-frame ring before VAD classification. After 5 positive frames, it emits `speech_started`, flushes the ring once in chronological order, then streams subsequent frames.
- Active turns include positive frames and the 8-frame trailing-silence hangover. After soft-end, the pre-roll ring starts fresh so reopened speech does not duplicate already-sent trailing audio.
- Soft-end immediately snapshots the entire accumulated utterance and launches endpoint Parakeet ASR. The LLM begins once ASR returns, while the 1-second grace timer continues.
- LLM tokens feed a sentence segmenter, but OmniVoice work and audio delivery remain gated until grace expires.
- A first positive frame during grace holds the gate. If it becomes 160 ms confirmed speech, increment `turnRevision`, invalidate the speculative `responseGeneration`, retain the accumulated utterance, append resumed audio, and rerun ASR/LLM after the next soft-end. If not confirmed, release the hold and preserve the original grace deadline.
- Each speculative assistant run receives a monotonically increasing `responseGeneration`. Every ASR, LLM, TTS, outbound audio, and playback event is checked against both the current turn revision and response generation.
- Confirmed speech during assistant playback is a new turn, not a revision. The browser immediately stops scheduled audio, marks the old generation stale, and sends the new speech event.
- In-flight PyTorch ASR/TTS calls cannot be safely killed. Cancellation removes queued work and drops results from already-running stale inference; locks remain held until those calls finish.

### History commitment

- The server resolves the trusted Persona system prompt and ignores client-supplied system messages, matching existing HTTP behavior.
- The speculative user transcript is added to session and PocketBase history only after grace commits the current revision.
- Assistant sentence text is acknowledged only after the corresponding audio finishes playing.
- Normal completion commits acknowledged segments as one assistant message.
- Barge-in commits only fully played segments. The currently playing sentence and all queued sentences are excluded.
- Persist an optional `finishReason: "interrupted"` alongside stored assistant messages so the UI can display an interruption marker without adding marker text to LLM context.
- On speaker-to-text mode switching, cancel the speaker generation, retain acknowledged content, close the socket/microphone, and render the same persisted conversation through the unchanged text UI.

## Implementation Changes

### Backend

- Add a speaker package containing:
  - A Transformers-based Parakeet adapter.
  - Pydantic protocol schemas and validation.
  - A connection-local session state machine.
  - Sentence segmentation, playback acknowledgement, cancellation, and generation filtering.
  - The `/speaker/v1` router.
- Load `AutoProcessor` and `AutoModelForTDT` eagerly during FastAPI lifespan. Use CUDA with bfloat16 when supported, otherwise float16; fall back to CPU float32. Run under `torch.inference_mode()` and serialize inference with a shared ASR lock.
- Warm up by decoding, downmixing, and resampling `backend/assets/default-voice.mp3` to 16 kHz, then performing one real transcription. Model load or inference exceptions fail backend startup; an empty warm-up transcript produces a warning.
- Keep OmniVoice’s existing internal ASR enabled because voice-clone prompt creation depends on it.
- Introduce `P_GPT_SPEAKER_ASR_MODEL`, defaulting to `nvidia/parakeet-tdt-0.6b-v3`; no speaker ASR selection UI is required.
- Reuse existing Persona prompt resolution, Ollama request construction, voice-clone prompt caches, and OmniVoice lock. Do not alter current HTTP endpoint behavior.
- Permit multiple WebSocket sessions with independent histories/state. Shared ASR and TTS locks serialize model computation; stale queued jobs check validity again after acquiring a lock.
- Reject binary input outside capture state, odd or incorrectly sized frames, unsupported formats, oversized history/configuration, and audio beyond the 60-second cap.
- Add structured timing logs keyed by session, turn, revision, and generation:
  - soft-end received
  - ASR start/end
  - first LLM token
  - first complete sentence
  - grace opened
  - TTS start/end
  - first audio sent
  - playback started/completed acknowledgements
- Avoid logging user transcripts or audio at normal log levels.
- Document the model’s CC BY 4.0 attribution and operational assumptions. The [Parakeet model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) confirms native Transformers support, 16 kHz mono input, automatic detection of 25 European languages, and the license. Hugging Face’s [speech-to-speech reference](https://github.com/huggingface/speech-to-speech) remains architectural guidance rather than a runtime dependency.

### Frontend

- Build on the existing uncommitted VAD groundwork; do not replace or discard it.
- Extract the VAD timing/ring logic into a deterministic state machine:
  - Append inactive frames before classification.
  - Flush pre-roll without duplication.
  - Emit candidate start/cancel events during grace.
  - Emit confirmed start, audio, soft-end, and limit events.
- Convert Float32 microphone frames to clamped little-endian PCM16 before sending.
- Keep the microphone active while the assistant is generating and playing. Pause only for explicit mute, mode exit, unsupported browser state, or fatal connection failure.
- Add a speaker-session hook responsible for:
  - Socket configuration and lifecycle.
  - Turn/revision/generation state.
  - Input backpressure.
  - Transcript and response buffers.
  - Idle-only reconnect.
  - Integration with the existing Zustand chat store and PocketBase persistence.
- Reconnect with capped exponential delays while speaker mode remains enabled. A mid-turn disconnect stops playback, commits locally known played segments as interrupted, fails any uncommitted input turn, and reconnects from persisted history; partial audio is never replayed.
- Add a dedicated PCM scheduler, sharing decoding utilities with `AudioMessage` where practical:
  - Maintain an 80 ms scheduling lead.
  - Track segment source nodes.
  - Acknowledge only after every source for a segment ends.
  - Stop all scheduled nodes synchronously on barge-in.
  - Drop binary data belonging to stale generations.
- Update the speaker UI with `connecting`, `listening`, `hearing`, `thinking`, `speaking`, `reconnecting`, and `error` states. Keep the avatar-focused layout, microphone/volume controls, and VAD indicator.
- Show the latest committed user transcript while processing. Buffer assistant text invisibly during playback and reveal it only after playback completion or interruption.
- Preserve the normal text input, browser Whisper transcription, HTTP generation, HTTP audio player, and their existing store actions unchanged.

## Test and Acceptance Plan

### Automated tests

- Add backend tests with fake ASR, LLM, TTS, and clocks:
  - Handshake and protocol validation.
  - Rejection of audio before configuration/start.
  - Exact PCM frame validation and 60-second cap.
  - Soft-end starts endpoint ASR once per revision.
  - Resume during grace invalidates stale ASR/LLM/TTS output.
  - Candidate speech holds and releases the grace gate correctly.
  - TTS never starts before grace commitment.
  - Sentence splitting and 180-character fallback.
  - Barge-in increments generation and drops stale queued/output audio.
  - Only acknowledged segments enter assistant history.
  - Disconnect cleanup and multiple sessions sharing serialized model locks.
  - Existing persona prompt and HTTP tests remain green.
- Add Vitest and test pure frontend logic:
  - Five-frame start and eight-frame soft-end thresholds.
  - Sixteen-frame pre-roll ordering and no duplication after reopen.
  - Candidate confirmation/cancellation near the grace deadline.
  - Float32-to-PCM16 clamping and little-endian encoding.
  - Protocol reducer rejection of stale generations.
  - Playback acknowledgements only after complete segments.
  - Immediate playback clearing on barge-in.
  - Backward-compatible parsing of history with optional interruption metadata.
- Verification commands:
  - Backend unit-test discovery using the selected `uv` CPU or CUDA extra.
  - `pnpm exec vitest run`.
  - `pnpm build`.
  - `pnpm lint`.

### Manual real-model scenarios

- Confirm startup downloads/loads Parakeet, successfully runs the default-voice warm-up, and surfaces failures before serving requests.
- Verify short words retain their beginnings through pre-roll.
- Verify brief noise and speech shorter than 160 ms do not start a turn.
- Verify pauses shorter than 256 ms remain within active speech.
- Verify a pause that soft-ends and resumes within grace becomes one combined user turn with no audible stale response.
- Verify English and Swedish transcription through automatic language detection.
- Verify first audio, stage timings, sentence boundaries, volume, mute, and latest-exchange rendering.
- Interrupt during the second sentence and confirm the browser stops within the 160 ms confirmation window, only the first fully played sentence persists, and the interruption marker appears.
- Switch between speaker and text modes and verify both operate on the same history without changing text-mode behavior.
- Drop the socket while idle and mid-turn to verify the selected reconnect policy.
- Run two speaker clients and confirm correctness while model work queues.
- Use captured timing data to establish a later end-of-speech-to-first-audio target; no numeric performance SLA is imposed on this prototype.

## Fresh-Context Handoff

- After leaving Plan Mode, save this plan verbatim as `docs/plans/realtime-speaker-prototype.md`.
- Start a fresh Codex task in the same workspace with this bootstrap instruction:

  > Read `AGENTS.md` and `docs/plans/realtime-speaker-prototype.md` completely. Inspect the dirty worktree before editing; the existing speaker VAD files and Chat page changes are intentional groundwork. Implement the plan in ordered milestones: protocol/state tests, backend Parakeet/session service, frontend transport/playback, Chat integration, then full verification. Preserve the existing HTTP text mode and unrelated user changes.

- The fresh task should use the same working tree so the current uncommitted VAD work is available; it should not create a clean worktree that omits those changes.
