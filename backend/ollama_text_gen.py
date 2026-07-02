from openai import OpenAI

MODEL = "google/gemma-4-E4B-it"
IP = "100.113.76.118"
PORT = "11434/api/chat"
URL = f"http://{IP}:{PORT}"

client = OpenAI(api_key="EMPTY", base_url=URL)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are concise assistant. Answer helpfully"},
        {"role": "user", "content": "Give me a one-paragraph vLLM tuning checklist. Answer in bullet points"},
    ],
    reasoning_effort="none",
    temperature=1.0,
    top_p=0.95,
    max_tokens=256,
)