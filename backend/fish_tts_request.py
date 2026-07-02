from pathlib import Path
from time import time

import requests

SPARK_URL = "100.113.76.118"
SERVER_URL = f"http://{SPARK_URL}:8080"

payload = {
    "text": "Hello and welcome. I am your doctor dietistGPT. How can I help you today?",
    "format": "wav",
    "references": [],
    "reference_id": None,
    "chunk_length": 200,
    "max_new_tokens": 1024,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "temperature": 0.8,
    "streaming": False,
    "use_memory_cache": "off",
    "seed": None,
}

t = time()
response = requests.post(
    f"{SERVER_URL}/v1/tts",
    json=payload,
    headers={"Accept": "audio/wav"},
    timeout=300,
)
response.raise_for_status()
t_tot = time() - t
print(payload)
print(f"Took {t_tot:.2f} s to generate {len(payload['text'].split(" "))} words")

Path("fish_speech.wav").write_bytes(response.content)
print("Saved fish_speech.wav")
