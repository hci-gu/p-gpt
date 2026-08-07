from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    inference_host: str = "100.113.76.118"
    ollama_text_model: str = "gemma4:e4b"
    tts_model: str = "k2-fsa/OmniVoice"
    speaker_asr_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    speaker_asr_model_sv: str = "KBLab/kb-whisper-medium"
    speaker_asr_revision_sv: str = "standard"
    speaker_reopen_grace_seconds: float = Field(default=2.0, ge=0.25, le=5)
    vllm_tts_model: str = "mistralai/Voxtral-4B-TTS-2603"
    pocketbase_base_url: str = "http://127.0.0.1:8090"
    persona_prompt_cache_path: str = str(
        Path(__file__).with_name("persona_prompt_cache.sqlite3")
    )
    mlflow_prompt_name: str = "pgpt-prompt"
    mlflow_prompt_version: int | str = "latest"
    n_retries: int = Field(default=3, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_path: str = str(Path(__file__).with_name("logs") / "p-gpt.log")
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    log_backup_count: int = Field(default=5, ge=1, le=50)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="P_GPT_",
        extra="ignore",
    )

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.inference_host}:11434"


settings = Settings()
