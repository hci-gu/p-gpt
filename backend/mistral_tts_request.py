import io
import httpx
import soundfile as sf
from time import time

BASE_URL = "http://100.113.76.118:8000/v1"

payload = {
    "input": "Hello and welcome. I am your doctor dietistGPT. How can I help you today?",
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "response_format": "wav",
    "voice": "casual_male",
}

t = time()
response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=20)
response.raise_for_status()
t_tot = time() - t
print(f"Took {t_tot:.2f} s to generate {len(payload['input'].split(" "))} words")

# Save the WAV exactly as returned by the server
output_path = "voxtral_output.wav"
with open(output_path, "wb") as f:
    f.write(response.content)

audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
print(f"Got audio: {len(audio_array)} samples at {sr} Hz")
print(f"Saved audio to {output_path}")