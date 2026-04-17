"""
Direct NVIDIA NIM call (same endpoint as the Gap Sheet app).
Usage: export NVIDIA_API_KEY=...  then  python example_direct_call.py
"""

import base64
import os

import requests

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

api_key = os.environ.get("NVIDIA_API_KEY", "")
if not api_key:
    raise SystemExit("Set NVIDIA_API_KEY")

# base64 — for embedding file bytes in JSON messages if you extend this script

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "text/event-stream" if stream else "application/json",
}

payload = {
    "model": "meta/llama-4-maverick-17b-128e-instruct",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 512,
    "temperature": 1.00,
    "top_p": 1.00,
    "frequency_penalty": 0.00,
    "presence_penalty": 0.00,
    "stream": stream,
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
