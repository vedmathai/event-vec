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
import os


from eventvec.server.config import Config
from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader


prompt_preamble = """
[INST] <<SYS>>

    The premise is a set of battles and their temporal relationships
    The hypothesis is a claim of the temporal relationship between two battles.

    
    There are three answer choices:
    1) True: The hypothesis is true given the premise
    2) False: The hypothesis is False given the premise
    3) Undefined: There is evidence for the hypothesis to be both true and false therefore the claim is undefined.

    The first five are examples with the labels provided.
    
    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 

    Examples:

    """

prompt_preamble_spatial = """
[INST] <<SYS>>

    The premise is a set of locations and their spatial relationships
    The hypothesis is a claim of the spatial relationship between two battles.

    
    There are three answer choices:
    1) True: The hypothesis is true given the premise
    2) False: The hypothesis is False given the premise
    3) Undefined: There is evidence for the hypothesis to be both true and false therefore the claim is undefined.

    The first five are examples with the labels provided.
    
    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 

    Examples:

    """

prompt_preamble_helped = """
[INST] <<SYS>>

    The premise is a set of battles and their temporal relationships
    The hypothesis is a claim of the temporal relationship between two battles.

    There are three answer choices:
    1) True: The hypothesis is true given the premise
    2) False: The hypothesis is False given the premise
    3) Undefined: There is evidence for the hypothesis to be both true and false therefore the claim is undefined.


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


label_map = {
    'True': 'True',
    'False': 'False',
    'Impossible': 'Undefined',
    'Contradictory': 'Undefined',
    'Undefined': 'Undefined',
    'Contrastive': 'Undefined',
}

class NLIDataPreparer():
    def __init__(self):
        self._data_readers = {
            'temporal': TemporalDatareader(),
        } 

    def load(self):
        k = 0
        file_name = 'temporal/gpt_o3_mini_temporal_nli_test.json'
        jl = JadeLogger()
        gpt_answer = {}
        true_answers = {}
        data_reader = self._data_readers['temporal']
        data = data_reader.data('temporal_nli_test')[:4800]
        example_data = [data[256], data[140], data[158], data[2627], data[2626], data[876]]
        data = [datum for datum in data if datum not in example_data]
        random.shuffle(data)
        data = data[:100]
        system_prompt = str(prompt_preamble)
        location = jl.file_manager.data_filepath(file_name)
        if os.path.exists(location):
            with open(location, 'rt') as f:
                gpt_answer = json.load(f)
        for datumi, datum in enumerate(example_data, start=1):
            system_prompt += f'{datum.uid()} \n Premise: ' + datum.premise() + '\n'
            system_prompt += 'Hypothesis: ' + datum.hypothesis() + '\n'
            system_prompt += f'Answer: {datum.uid()} :' + label_map.get(datum.label()) +  '\n\n'


        for datumi, datum in enumerate(data):
            if str(datum.uid()) in gpt_answer:
                continue
            print('premise:', datum.premise())
            print('hypothesis:', datum.hypothesis())
            user_prompt_normal = f"""
                <</SYS>>

                Provide the labels for the following sentences in the format of 'answer: index: label'.
            """
            user_prompt = user_prompt_normal
            true_answers[datum.uid()] = datum.label()
            user_prompt += f'{datum.uid()} Premise: ' + datum.premise() + '\n'
            user_prompt += 'Hypothesis: ' + datum.hypothesis() + '\n [/INST] \n'
            print("prompting", datum.uid(), datum.label())
            print('sending prompt')
            answer = gpt_4(system_prompt, user_prompt)
            print('received response')
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    print(line)
                    try: 
                        #_, index, premise_credence, hypothesis_credence, label = line.split(':')
                        _, index, label = line.split(':')
                        if index.strip() == datum.uid():
                            #gpt_answer[index.strip()] = [label.strip(), premise_credence.strip(), hypothesis_credence.strip()]

                            gpt_answer[index.strip()] = [label.strip(), '', '']
                        else:
                            raise ValueError

                    except ValueError:
                        #gpt_answer[datum.uid()] = ['', '', '']
                        continue
            print(self.f1_score(true_answers, gpt_answer))
            k += 1
            with open(location, 'wt') as f:
                f.write(json.dumps(gpt_answer))

            
    def f1_score(self, true_answers, gpt_answers):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        f1s = []
        for uid, label in true_answers.items():

            if uid not in gpt_answers:
                fn[label] += 1
                continue
            gpt_answer = label_map.get(gpt_answers[uid][0])
            if label.lower() in gpt_answer.lower():
                tp[gpt_answer] += 1
            else:
                fp[label] += 1
                fn[gpt_answer] += 1
        for key in ['Undefined', 'True', 'False']:
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