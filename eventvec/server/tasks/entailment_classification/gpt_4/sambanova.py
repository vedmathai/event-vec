import os
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.sambanova.ai/v1",
)

llama405="Meta-Llama-3.1-405B-Instruct"
llama70="Meta-Llama-3.3-70B-Instruct"
llama8="Meta-Llama-3.1-8B-Instruct"

deepseek = 'DeepSeek-R1-0528'
qwq = 'QwQ-32B'
qwen3 = 'Qwen3-32B'
gpt = 'gpt-oss-120b'

model_map = {
    'llama405': llama405,
    'llama70': llama70,
    'llama8': llama8,
    'deepseek': deepseek,
    'qwq': qwq,
    'qwen3': qwen3,
    'gpt': gpt,
}

def sambanova(model, system, user):
    completion = client.chat.completions.create(
        model=model_map[model],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return completion.choices[0].message.content
