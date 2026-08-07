from pydantic import BaseModel, Field


class PersonaProfile(BaseModel):
    problem: str = Field(description="Short description of the main problem that brings them to the therapist office")
    background: str = Field(description="Detailed background story of this persona. Includes behaviour, speaking patterns and emotional personality.")


class PersonaInput(BaseModel):
    id: str
    name: str
    instruction_prompt: str


class PersonaPreparationRequest(BaseModel):
    persona_id: str = Field(min_length=1)
    persona_name: str = Field(min_length=1)
    instruction_prompt: str = Field(min_length=1)
    audio_sample_url: str | None = None
    previous_audio_sample_url: str | None = None
    prepare_system_prompt: bool = False
    prepare_voice_clone_prompt: bool = False
