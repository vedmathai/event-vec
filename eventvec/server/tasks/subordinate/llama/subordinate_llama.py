import numpy as np
import pprint
import re
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
import pprint
import random
from collections import defaultdict
from jadelogs import JadeLogger
import json
import time
import os


from eventvec.server.config import Config
from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova

from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader
from eventvec.server.tasks.subordinate.datareader.datareader import SubordinateTemporalDatareader



prompt_preamble = """
[INST] <<SYS>>

    Replace <Mask> with an answer from ['before', 'after', 'during'] for the following. Answer '-' if the sentence is not valid. Many answers are possible.
    Now is when the entire sentence is uttered.

    Example 1:
    Sentence: John said that Mary had eaten.
    Question: John saying happens <Mask> Mary eating. 
    Answer: after

    Example 2:
    Sentence: John said that Mary will eat.
    Question: John saying happens <Mask> Mary eating. 
    Answer: before

    The temporal relationships are: before, after, during, iduring.

    
    The query is in the form:
    Sentence: <sentence> 
    John saying: <tense> <aspect>
    Mary eating: <tense> <aspect>
    Question: <question>

    Response format:
    Answer: <Answer>

    Here are examples:

    """

prompt_preamble_cloze_dct_sub = """
[INST] <<SYS>>

    I have received an email with a message. I want to know the temporal relationship between the event mentioned in the email and the time when the email was written.
    Replace <Mask> in the sentence with an relationship from ['before', 'after', 'during'] for the following. Answer '-' if the sentence is not valid. Many answers are possible.

    Example 1:
    Email: John said that Mary had eaten.
    Sentence: Email is written <Mask> Mary eating. 
    Answer: ['after']

    Example 2:
    Email: John said that Mary will eat.
    Sentence: Email is written <Mask> Mary eating. 
    Answer: ['before', 'after', 'during']

    The query is in the form:
    Email: <email contents> 
    Sentence: <question>

    Response format:
    Answer: <Answer>

    Here are examples:

    """


prompt_preamble_question_matrix_sub = """
[INST] <<SYS>>

    This is a modified NLI task. You will be given a premise written by speaker A and read by listener B. You will also be given a hypothesis. 
    Answer with 'before', 'after' or 'during'. There can be multiple valid answers.

    Example 1:
    Speaker A: John said that Mary had eaten.'
    Question: When does John's utterance happen in relation to Mary eating? 
    Answer: ['after']

    key: 7624:
    Speaker A: John said that Mary will eat.
    Question: When does John's utterance happen in relation to Mary eating? 
    Answer: ['before']

    The query is in the form:
    Speaker A: <key>: 
    Question: <premise> 
    Answer: <answer>

    Response format:
    Answer: <Answer>

    """


prompt_preamble_question_dct_matrix = """
[INST] <<SYS>>

    This is a dialogue question answering task. You will be given a premise written by speaker A and read by listener B. You will then have to answer a question about the temporal relations in the dialogue.
    Answer with 'before', 'after' or 'during'. There can be multiple valid answers.

    Example 1:
    Speaker A: John said that Mary had eaten.'
    Question: When does John's utterance happen in relation to Mary eating? 
    Answer: ['after']

    key: 7624:
    Speaker A: John said that Mary will eat.
    Question: When does John's utterance happen in relation to Mary eating? 
    Answer: ['before']

    The query is in the form:
    Speaker A: <key>: 
    Question: <premise> 
    Answer: <answer>

    Response format:
    Answer: <Answer>

    """

prompt_preamble_question_dct_sub = """
[INST] <<SYS>>

    This is a dialogue question answering task. You will be given a premise written by speaker A and read by listener B. You will then have to answer a question about the temporal relations in the dialogue.
    Answer with 'before', 'after' or 'during'. There can be multiple valid answers.
    
    Example 1:
    Speaker A: John said that Mary will eat.
    Question: When does Speaker A's utterance happen in relation to Mary's eating?  
    Answer: ['after']

    Speaker A: John said, 'Mary will eat.'
    Question: When does Speaker A's utterance happen in relation to Mary's eating?  
    Answer: ['after']

    The query is in the form:
    Speaker A: <key>: 
    Question: <premise> 
    Answer: <answer>

    Response format:
    Answer: <Answer>

    """

prompt_preamble_question_dct_sub_unembeded = """
[INST] <<SYS>>

    This is a dialogue question answering task. You will be given a premise written by speaker A and read by listener B. You will then have to answer a question about the temporal relations in the dialogue.
    Answer with 'before', 'after' or 'during'. There can be multiple valid answers.
    
    Example 1:
    Speaker A: Mary will eat.
    Question: When does Speaker A's utterance happen in relation to Mary's eating?  
    Answer: ['before']

    Speaker A: Mary ate yesterday.
    Question: When does Speaker A's utterance happen in relation to Mary's eating?  
    Answer: ['after']

    The query is in the form:
    Speaker A: <key>: 
    Question: <premise> 
    Answer: <answer>

    Response format:
    Answer: <Answer>

    """

prompt_preamble_cloze_dct_sub_umembedded = """
[INST] <<SYS>>

    I have received an email with a message. I want to know the temporal relationship between the event mentioned in the email and the time when the email was written.
    Replace <Mask> in the sentence with an relationship from ['before', 'after', 'during'] for the following. Answer '-' if the sentence is not valid. Many answers are possible.

    Example 1:
    Email: Mary had eaten.
    Sentence: Email is written <Mask> Mary eating. 
    Answer: ['after']

    Example 2:
    Email: Mary will eat.
    Sentence: Email is written <Mask> Mary eating. 
    Answer: ['before']

    The query is in the form:
    Email: <email contents> 
    Sentence: <question>

    Response format:
    Answer: <Answer>

    Here are examples:

    """

class NLIDataPreparer():
    def __init__(self):
        self._data_readers = {
            'subordinate': SubordinateTemporalDatareader(),
        } 

    def load(self):
        k = 0
        file_name = 'subordinate/gpt_oss_subordinate_question_plain_dct_is_sub_2.json'
        jl = JadeLogger()
        gpt_answer = {}
        data_reader = self._data_readers['subordinate']
        data = data_reader.data('temporal_subordinate_said')[:4800]
        example_data = [data[256], data[140], data[158], data[100], data[0], data[2]]
        #example_data = [data[0], data[10], data[20], data[30]]
        data = [datum for datum in data if datum not in example_data]
        random.shuffle(data)
        system_prompt = str(prompt_preamble_question_dct_sub)
        location = jl.file_manager.data_filepath(file_name)
        if os.path.exists(location):
            with open(location, 'rt') as f:
                gpt_answer = json.load(f)
        for datumi, datum in enumerate(example_data, start=1):
            system_prompt += f'Speaker A: {datum.example()} \n'
            system_prompt += f"Question: When does Speaker A's utterance happen in relation to Mary's eating?"
            system_prompt += f'Sentence: {str(datum.dct_is_sub())} \n'

        for datumi, datum in enumerate(data):
            if str(datum.key()) in gpt_answer:
                continue

            user_prompt_normal = f"""
                <</SYS>>

                Provide the labels for the following sentences in the format of 'Answer: <A>.
            """
            user_prompt = user_prompt_normal
            user_prompt += f'{hash(datum.key())}: \n'
            user_prompt += f'Speaker A: {datum.example()} \n'
            user_prompt += f"Question: When does Speaker A's utterance happen in relation to Mary's eating?"
            print("prompting", hash(datum.key()), datum.dct_is_sub())
            print('sending prompt')
            answer = sambanova('gpt', system_prompt, user_prompt)
            print('received response')
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    print(line)
            
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    #_, index, premise_credence, hypothesis_credence, label = line.split(':')
                    if len(line.split(':')) == 2:
                        _, dct_is_sub = line.split(':')
                        gpt_answer[datum.key()] = [datum.example(), dct_is_sub, datum.dct_is_sub()]
                    #else:
                    #    raise ValueError

            k += 1
            with open(location, 'wt') as f:
                f.write(json.dumps(gpt_answer))
            time.sleep(2)

            
if __name__ == '__main__':
    Config.instance()
    data_preparer = NLIDataPreparer()
    data_preparer.load()