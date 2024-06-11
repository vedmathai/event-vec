import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()


def gpt_4(system, user):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return completion.choices[0].message.content
