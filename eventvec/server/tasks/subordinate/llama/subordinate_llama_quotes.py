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
import csv


from eventvec.server.config import Config
from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova

from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader
from eventvec.server.tasks.subordinate.datareader.datareader import SubordinateTemporalDatareader



prompt_preamble_question_pronoun = """
[INST] <<SYS>>

    This is a question answering task. You will be given a premise written by speaker A and read by listener B. You will also be given a question. 
    Answer with one of ['Alice', 'Speaker A', 'Bob', 'Listener B', 'unknown third person'. 

    Example 1:
    Speaker A: I was talking to Alice earlier and Alice said, 'Alice’s house is in Oxford.''
    Question: According to Speaker A and Alice, whose house is in Oxford?
    Answer: Alice

    Example 2:
    Speaker A: I was talking to Alice earlier and Alice said that Alice’s house is in Oxford.
    Question: According to Speaker A and Alice, whose house is in Oxford?
    Answer: Alice

    The query is in the form:
    Speaker A: <premise>
    Question: <question>

    Response format:
    Answer: <Answer>

    """

model = 'gpt5'

class NLIDataPreparer():

    def read_data(self):
        with open('/home/lalady6977/oerc/projects/local_jade/jade_front/event-vec/data/temporal_subordinate/pronoun_subordinate.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            data = []
            for ri, r in enumerate(reader):
                if r[0] != '' and ri > 0:
                    data.append(r)
        return data

    def load(self):
        k = 0
        file_name = 'subordinate/pronoun_subordinate_{}_2.json'.format(model)
        jl = JadeLogger()
        gpt_answer = {}
        data = self.read_data()
        random.shuffle(data)
        system_prompt = str(prompt_preamble_question_pronoun)
        location = jl.file_manager.data_filepath(file_name)
        if os.path.exists(location):
            with open(location, 'rt') as f:
                gpt_answer = json.load(f)

        for datumi, datum in enumerate(data):
            key = '_'.join(datum[0:5])
            if key in gpt_answer:
                continue

            user_prompt_normal = f"""
                <</SYS>>

                Provide the labels for the following sentences in the format of 'Answer: <A>.
            """
            user_prompt = user_prompt_normal
            #user_prompt += f'{key}: \n'
            user_prompt += f'Speaker A: {datum[5]} \n'
            user_prompt += f"Question:  {datum[6]} "
            print('sending prompt')
            answer = gpt_4(model, system_prompt, user_prompt)
            print('received response')
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    print(line, 'expected:', datum[7])
            
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    #_, index, premise_credence, hypothesis_credence, label = line.split(':')
                    if len(line.split(':')) == 2:
                        _, referrent = line.split(':')
                        gpt_answer[key] = [datum[5], referrent, datum[7]]
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