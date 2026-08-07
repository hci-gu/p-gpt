from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTtsTiming:
    lock_wait_seconds: float
    inference_seconds: float
    encoding_seconds: float
    total_seconds: float
    audio_seconds: float
    real_time_factor: float


def calculate_speaker_tts_timing(
    *,
    total_started_at: float,
    lock_requested_at: float,
    lock_acquired_at: float,
    inference_finished_at: float,
    encoding_started_at: float,
    completed_at: float,
    output_bytes: int,
    sample_rate: int,
) -> SpeakerTtsTiming:
    audio_seconds = output_bytes / (sample_rate * 2)
    inference_seconds = inference_finished_at - lock_acquired_at
    return SpeakerTtsTiming(
        lock_wait_seconds=lock_acquired_at - lock_requested_at,
        inference_seconds=inference_seconds,
        encoding_seconds=completed_at - encoding_started_at,
        total_seconds=completed_at - total_started_at,
        audio_seconds=audio_seconds,
        real_time_factor=(
            inference_seconds / audio_seconds if audio_seconds > 0 else 0.0
        ),
    )
