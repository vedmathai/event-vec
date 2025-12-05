
import os
from openai import OpenAI

os.environ["DASHSCOPE_API_KEY"] = ""


client = OpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

def qwen(system, user):
    completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
    return completion.choices[0].message.content
