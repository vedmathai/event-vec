import os
from openai import OpenAI


def sambanova(system, user):
    completion = client.chat.completions.create(
        model='DeepSeek-R1',#"Meta-Llama-3.1-405B-Instruct",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return completion.choices[0].message.content
