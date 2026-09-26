"""
Direct NVIDIA NIM call (same client as the Gap Sheet app).
Usage: export NVIDIA_API_KEY=...  then  python example_direct_call.py
"""

import os

from openai import OpenAI

api_key = os.environ.get("NVIDIA_API_KEY", "")
if not api_key:
    raise SystemExit("Set NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)
completion = client.chat.completions.create(
    model="moonshotai/kimi-k3",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    temperature=1,
    top_p=0.95,
    max_tokens=1024,
    stream=False,
    extra_body={"reasoning_effort": "low"},
)
print(completion.choices[0].message)
