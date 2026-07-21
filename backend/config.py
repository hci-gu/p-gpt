from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    inference_host: str = "100.113.76.118"
    ollama_text_model: str = "gemma4:e4b"
    tts_model: str = "k2-fsa/OmniVoice"
    vllm_tts_model: str = "mistralai/Voxtral-4B-TTS-2603"
    pocketbase_base_url: str = "http://127.0.0.1:8090"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="P_GPT_",
        extra="ignore",
    )

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.inference_host}:11434"

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.inference_host}:8001/v1"

settings = Settings()