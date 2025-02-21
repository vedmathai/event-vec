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
from eventvec.server.tasks.subordinate.datareader.subordinate_datareader import SubordinateTemporalDatareader



prompt_preamble = """
[INST] <<SYS>>

    Replace <Mask> with an answer from ['before', 'after', 'during'] for the following. Answer '-' if the sentence is not valid. Many answers are possible.
    Now is when the entire sentence is uttered.

    Example 1:
    Key: 5647:
    Sentence: John said that Mary had eaten.
    Question: John saying happens <Mask> Mary eating. 
    Answer: 5647: after

    key: 7624:
    Sentence: John said that Mary will eat.
    Question: John saying happens <Mask> Mary eating. 
    Answer: 7624: before

    The temporal relationships are: before, after, during, iduring.

    
    The query is in the form:
    key: <key>: 
    Sentence: <sentence> 
    John saying: <tense> <aspect>
    Mary eating: <tense> <aspect>
    Question: <question>

    Response format:
    Answer: <key>: <Answer>

    Here are examples:



    """

prompt_preamble_2 = """
[INST] <<SYS>>

    Replace <Mask> with an answer from ['before', 'after', 'during'] for the following. Answer '-' if the sentence is not valid. Many answers are possible.
    Now is when the entire sentence is uttered.
    
    Example 1:
    Key: 5647:
    Sentence: John said that Mary had eaten.
    Question: Now is <mask> John saying.
    Answer: 5647: after

    key: 7624:
    Sentence: John will say that Mary will eat.
    Question: Now is <mask> John saying.
    Answer: 7624: before


    The temporal relationships are: before, after, during.
    
    The query is in the form:
    key: <key>: 
    Sentence: <sentence> 
    Question: <question>

    Response format:
    Answer: <key>: <Answer>

    Here are examples:


    """

prompt_preamble_3 = """
[INST] <<SYS>>

    Replace <Mask> with an answer from ['before', 'after', 'during'] for the following. Answer '-' if the sentence is not valid. Many answers are possible.
    Now is when the entire sentence is uttered.
    
    Example 1:
    Key: 5647:
    Sentence: John said that Mary had eaten.
    Question: Now is <MASK> Mary eating.
    Answer: 5647: after

    key: 7624:
    Sentence: John will say that Mary will eat.
    Question: Now is <MASK> Mary eating.
    Answer: 7624: after


    The temporal relationships are: before, after, during.
    
    The query is in the form:
    key: <key>: 
    Sentence: <sentence> 
    Question: <question>

    Response format:
    Answer: <key>: <Answer>

    Here are examples:


    """

prompt_preamble_helped = """
[INST] <<SYS>>

    The premise is a set of battles and their temporal relationships
    The hypothesis is a claim of the temporal relationship between two battles.

    There are three answer choices:
    1) True: The hypothesis is true given the premise
    2) False: The hypothesis is False given the premise
    3) Contradictory: There is contradictory evidence in the premise regarding the events in the hypothesis. So no claim can be made.

    Output all the paths in the premise between the events in the hypothesis.

    List all the connected paths between the two events in hypothesis.
    If Event A before Event B and Event B before Event C, then Event A before Event C.
    If Event A simultaneous Event B and Event B before Event C, then Event A before Event C.
    If Event A before Event B and Event B before Event C and Event C before Event A then impossible.
    If end of Event A before start of Event B then Event A before Event B.
    If start of Event A before start of Event B and end of Event A after start of Event B then Event A overlaps
    Event B.

    The first five are examples with the labels provided.

    Your task is to predict the label for the given examples. Do NOT provide reasoning and finally
    provide in the format of 'Answer: index: label'. 

    Examples:

    """



class NLIDataPreparer():
    def __init__(self):
        self._data_readers = {
            'subordinate': SubordinateTemporalDatareader(),
        } 

    def load(self):
        k = 0
        file_name = 'subordinate/llama_405B_subordinate_plain_dct_is_matrix.json'
        jl = JadeLogger()
        gpt_answer = {}
        true_answers = {}
        data_reader = self._data_readers['subordinate']
        data = data_reader.data()[:4800]
        example_data = [data[256], data[140], data[158], data[100], data[0], data[2]]
        data = [datum for datum in data if datum not in example_data]
        random.shuffle(data)
        system_prompt = str(prompt_preamble_2)
        location = jl.file_manager.data_filepath(file_name)
        if os.path.exists(location):
            with open(location, 'rt') as f:
                gpt_answer = json.load(f)
        for datumi, datum in enumerate(example_data, start=1):
            system_prompt += f'key: {hash(datum.key())}: \n'
            system_prompt += f'Sentence: {datum.example()} \n'
            system_prompt += f'Question: Now is <mask> John saying. \n'
            system_prompt += f'Answer: {hash(datum.key())}: {str(datum.dct_is_matrix())} \n'

        for datumi, datum in enumerate(data):
            if str(datum.key()) in gpt_answer:
                continue

            user_prompt_normal = f"""
                <</SYS>>

                Provide the labels for the following sentences in the format of 'Answer: <key>: <A>.
            """
            user_prompt = user_prompt_normal
            true_answers[datum.key()] = [datum.matrix_is_sub(), datum.dct_is_sub()]
            user_prompt += f'{hash(datum.key())}: \n'
            user_prompt += f'Sentence: {datum.example()} \n'
            user_prompt += f'Now is <mask> John saying. \n'
            print("prompting", hash(datum.key()), datum.dct_is_matrix())
            print('sending prompt')
            answer = sambanova(system_prompt, user_prompt)
            print('received response')
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    print(line)
            
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    #_, index, premise_credence, hypothesis_credence, label = line.split(':')
                    if len(line.split(':')) == 3:
                        _, index, dct_is_sub = line.split(':')
                        if index.strip() == str(hash(datum.key())):
                            #gpt_answer[index.strip()] = [label.strip(), premise_credence.strip(), hypothesis_credence.strip()]

                            gpt_answer[datum.key()] = [datum.example(), dct_is_sub, dct_is_sub]
                    #else:
                    #    raise ValueError

            #print(self.f1_score(true_answers, gpt_answer))
            k += 1
            with open(location, 'wt') as f:
                f.write(json.dumps(gpt_answer))
            time.sleep(15)

            
    def f1_score(self, true_answers, gpt_answers):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        f1s = []
        for uid, label in true_answers.items():

            if uid not in gpt_answers:
                fn[label] += 1
                continue
            gpt_answer = gpt_answers[uid][0]
            if label.lower() in gpt_answer.lower():
                tp[gpt_answer] += 1
            else:
                fp[label] += 1
                fn[gpt_answer] += 1
        for key in ['Contradictory', 'True', 'False']:
            f1 = 0
            precision = 0
            recall = 0
            if tp[key] + fp[key] != 0:
                precision = tp[key] / (tp[key] + fp[key])
            if tp[key] + fn[key] != 0:
                recall = tp[key] / (tp[key] + fn[key])
            if precision + recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            f1s.append(f1)
        return np.mean(f1s)

if __name__ == '__main__':
    Config.instance()
    data_preparer = NLIDataPreparer()
    data_preparer.load()