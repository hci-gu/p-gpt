from contextlib import contextmanager, nullcontext
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import soundfile as sf


QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human capable of perceiving auditory inputs "
    "and generating concise, natural text and speech responses."
)


@contextmanager
def _qwen2_5_config_load_logging():
    """Hide a known Transformers false positive for Qwen2.5-Omni.

    Qwen2.5-Omni's Talker config deliberately uses a small codec vocabulary
    while its ``tts_text_*`` IDs belong to the shared text vocabulary. Recent
    Transformers releases validate every ``*_token_id`` against the Talker
    vocabulary and log a warning, even though these IDs are valid for the
    model's split-vocabulary architecture.
    """
    config_logger = logging.getLogger("transformers.configuration_utils")
    previous_level = config_logger.level
    config_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        config_logger.setLevel(previous_level)


class QwenOmniModel:
    """Small adapter around the Transformers Qwen Omni checkpoints.

    The adapter intentionally keeps the model API behind one stable method so
    the HTTP layer can switch from Qwen2.5-Omni to Qwen3-Omni without changing
    the browser protocol.
    """

    def __init__(self, model_id: str) -> None:
        import torch

        self.model_id = model_id
        self.is_qwen3 = "Qwen3-Omni" in model_id

        if self.is_qwen3:
            from transformers import (  # type: ignore[import-not-found]
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )

            model_type: Any = Qwen3OmniMoeForConditionalGeneration
            processor_type: Any = Qwen3OmniMoeProcessor
        else:
            from transformers import (  # type: ignore[import-not-found]
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )

            model_type = Qwen2_5OmniForConditionalGeneration
            processor_type = Qwen2_5OmniProcessor

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": "auto",
        }
        if self.is_qwen3:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            model_kwargs["attn_implementation"] = "flash_attention_2"

        load_context = (
            _qwen2_5_config_load_logging()
            if not self.is_qwen3
            else nullcontext()
        )
        with load_context:
            self.model = model_type.from_pretrained(model_id, **model_kwargs)
            self.processor = processor_type.from_pretrained(model_id)
        self.torch = torch

    def generate(
        self,
        messages: list[dict[str, Any]],
        audio: Any,
        sample_rate: int,
        speaker: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, bytes]:
        from qwen_omni_utils import process_mm_info

        with NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            input_path = Path(temporary.name)

        try:
            sf.write(input_path, audio, sample_rate, format="WAV", subtype="PCM_16")
            conversation = [*messages]
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            audios, images, videos = process_mm_info(
                conversation,
                use_audio_in_video=False,
            )
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "speaker": speaker,
                "use_audio_in_video": False,
            }
            if self.is_qwen3:
                generation_kwargs["thinker_return_dict_in_generate"] = True

            with self.torch.inference_mode():
                text_ids, audio_output = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            if self.is_qwen3:
                text_ids = text_ids.sequences[:, inputs["input_ids"].shape[1] :]
            decoded = self.processor.batch_decode(
                text_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_text = decoded[0].strip() if decoded else ""
            if not generated_text:
                raise RuntimeError("Qwen Omni generated no text.")

            if audio_output is None:
                raise RuntimeError("Qwen Omni generated no audio.")

            audio_array = audio_output.reshape(-1).detach().float().cpu().numpy()
            output = NamedTemporaryFile(suffix=".pcm", delete=False)
            output_path = Path(output.name)
            output.close()
            try:
                sf.write(
                    output_path,
                    audio_array,
                    24_000,
                    format="RAW",
                    subtype="PCM_16",
                    endian="LITTLE",
                )
                return generated_text, output_path.read_bytes()
            finally:
                output_path.unlink(missing_ok=True)
        finally:
            input_path.unlink(missing_ok=True)


def build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    audio_path: str,
    input_text: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"{QWEN_SYSTEM_PROMPT}\n\n{system_prompt}"}],
        }
    ]
    messages.extend(
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in history
        if message["role"] in {"user", "assistant"}
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {
                    "type": "text",
                    "text": input_text
                    or "Respond to what the user said naturally and concisely.",
                },
            ],
        }
    )
    return messages
