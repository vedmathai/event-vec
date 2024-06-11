import requests
import time

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
#API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": "Bearer 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"}

def llama_3(system_prompt, user_prompt):
    payload = {"inputs": system_prompt + '\n\n' + user_prompt}
    
    tries = 5
    while tries > 0:
        response = requests.post(API_URL, headers=headers, json=payload)
        if 'error' in response.json():
            tries -= 1
            time.sleep(60)
            print('sleeping', tries)
        else:
            break
    return response.json()[0]['generated_text']
