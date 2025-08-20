import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
)

llama405="Meta-Llama-3.1-405B-Instruct"
llama70="Meta-Llama-3.3-70B-Instruct"
deepseek = 'DeepSeek-R1'
qwq = 'QwQ-32B'
qwen3 = 'Qwen3-32B'

def sambanova(system, user):
    completion = client.chat.completions.create(
        model=qwen3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return completion.choices[0].message.content
