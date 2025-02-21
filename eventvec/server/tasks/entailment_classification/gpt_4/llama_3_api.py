import requests
import time

#API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-405B-Instruct"
#API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-70B-Instruct"
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
#API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407"

def llama_3(system_prompt, user_prompt):
    payload = {"inputs": system_prompt + '\n\n' + user_prompt}
    
    tries = 5
    while tries > 0:
        response = requests.post(API_URL, headers=headers, json=payload)
        if 'error' in response.json():
            tries -= 1
            print(response.json())
            time.sleep(60)
            print(response.json())
            print('sleeping', tries)
        else:
            break
    return response.json()[0]['generated_text']


if __name__ == '__main__':
    system_prompt = """
    [INST] <<SYS>>
    The premise is a set of battles and their temporal relationships
    The hypothesis is a claim of the temporal relationship between two battles.

    There are three answer choices:
    1) True: The hypothesis is true given the premise
    2) False: The hypothesis is False given the premise
    3 Impossible: There is conflicting evidence in the premise regarding the events in the hypothesis. So no claim can be made.


    Provide your choice with an explanation.
     <</SYS>> 
    """

    user_prompt = """
Premise: start of Siege of Bloodthorn Keep happened after start of Conflict at Steelshade Valley. end of Siege of Bloodthorn Keep happened before start of Encounter at Misty Heights. end of Encounter at Misty Heights happened before start of Conflict at Steelshade Valley. start of Encounter at Misty Heights happened simultaneous end of Siege of Darkwater Keep

Hypothesis: Siege of Bloodthorn Keep happens after Siege of Darkwater Keep

    Do justify your answer along with an answer from one of the following ['True', 'False', 'Impossible]'.
    [/INST] 
    """
    print(llama_3(system_prompt, user_prompt))