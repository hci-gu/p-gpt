import io
import httpx
from time import time
import soundfile as sf

input_text = """Hello my name is Mr Omni Voice and would like to speak to you about diet and nutrition. My goal is to lose 5 kilograms.
"""
 
BASE_URL = "http://100.113.76.118:8001/v1"
FORMAT = "mp3" # mp3 or wav

 
payload = {
    "input": input_text,
    "model": "k2-fsa/OmniVoice",
    "response_format": FORMAT,
    "num_step": 22,
    "speed": 0.8,
}

t_start = time()
response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=300)
response.raise_for_status()
t_tot = time() - t_start
 
audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
sf.write("omnivoice."+FORMAT, audio_array, sr)
duration = len(audio_array) / sr

print("\n ### Statistics ###")
print(f"Got audio: {len(audio_array)} samples at {sr} Hz")
print(f"Request took {t_tot:.2f} s | RTF: {t_tot/duration:.3f} ({duration/t_tot:.1f}x real-time)")
print(response)
