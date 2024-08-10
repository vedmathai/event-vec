import requests
import time

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
#API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
#API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407"
headers = {"Authorization": "Bearer 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"}

def llama_3(system_prompt, user_prompt):
    payload = {"inputs": system_prompt + '\n\n' + user_prompt}
    
    tries = 5
    while tries > 0:
        response = requests.post(API_URL, headers=headers, json=payload)
        if 'error' in response.json():
            tries -= 1
            time.sleep(60)
            print(response.json())
            print('sleeping', tries)
        else:
            break
    return response.json()[0]['generated_text']


if __name__ == '__main__':
    system_prompt = """
    [INST] <<SYS>>
    The task is to Fill in the blanks with only the following words ["is", "were", 'weren't", "isn't", "was", "wasn't", "doesn't", "does", "can", "can't", "did", "didn't", "won't", "will", "not", "but", "and", "therefore", "because", "X"]. 
    If there is no appropriate word, then leave it unanswered. NO other word is allowed. Use the ADDITION as the culmination.

    Apply a negation such as ["doesn't, didn't, won't, not, can't"] to negate the event it is next to.
    Apply a conjunction such as ["and, but, therefore, because"] to connect the two events.
    Make sure "but" is used to contrast two events.
    "and" is used to connect two events that are similar but not contrasting.
    "therefore" is used to show a consequence.
    "because" is used to show a reason.

    Such as the following examples:
    PASSAGE: I MASK_1 want to drive the Lamborghini. MASK_2 I MASK_3 know how to.
    ADDITION: Therefore, I crashed the car.
    Answer:
    MASK_1: X
    MASK_2: but
    MASK_3: don't

    PASSAGE: I am MASK_1 thirsty. MASK_2 I MASK_3 get a drink."
    ADDITION: Therefore, I still want more water.
    Answer: 
    MASK_1: X
    MASK_2: but
    MASK_3: didn't

    
    PASSAGE: The win MASK_1 imminent. MASK_2 the team MASK_3 playing well.
    ADDITION: Therefore, the crowd is cheering.
    Answer:
    MASK_1: is
    MASK_2: because
    MASK_3: is

    Fill in the blanks strictly only with the following words ["is", "were", 'weren't", "isn't", "was", "wasn't", "doesn't", "does", "can", "can't", "did", "didn't", "won't", "will", "not", "but", "and", "therefore", "because", "X"].
    If a word doesn't apply then leave it blank. 
     <</SYS>> 
    """

    user_prompt = """

  
    PASSAGE :
    MASK_1 chris MASK_2 tell me to MASK_3 hold one side of the sofa . MASK_4 we MASK_5 start MASK_6 carrying it downstairs , MASK_7 chris MASK_8 lose his grip .
    ADDITION: It landed on the bottom floor.

    [/INST] 
    """
    print(llama_3(system_prompt, user_prompt))