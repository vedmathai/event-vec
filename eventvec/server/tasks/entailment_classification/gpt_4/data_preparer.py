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


from eventvec.server.config import Config
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4

from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.snli_data_reader import SNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.anli_data_reader import ANLIDataReader  # noqa

from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_data_reader import ChaosMNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_snli_data_reader import ChaosSNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_anli_data_reader import ChaosANLIDatareader  # noqa

prompt_preamble = """
    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to determine whether the hypothesis
    can be inferred from the premise.
     
    You will be given three options: entailment, neutral, and contradiction. 
    If the hypothesis can be inferred from the premise, you should select entailment.
    If the hypothesis is unrelated to the premise, you should select neutral.
    If the hypothesis contradicts the premise, you should select contradiction.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'index: label'. 

    Event credence is the degree of belief that an event will occur on a score between -3 and 3 where -3 is certainly won't happen and 3 is certainly did happen.
    The linguistic factors that effect event credence are:
    - The presence of modals
    - The presence of adverbs
    - The presence of negation
    - The presence of tense
    - The infinitive verb that is subordinate to the main verb such as expected to, intended to, etc.
    - The event is a subordinated clause of a speech verb such as said, told, etc. or a belief verb such as believe, think, etc.
    
    Credence has an effect on the entailment task. Because if the premise and hypothesis are talking about the same event then they may entail if the credences are the same.
    Use this reasoning to predict the label for the examples.
    The first five are examples with the labels provided.
    Examples:

    """

prompt_preamble = """
    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to determine whether the hypothesis
    can be inferred from the premise.
     
    You will be given three options: entailment, neutral, and contradiction. 
    If the hypothesis can be inferred from the premise, you should select entailment.
    If the hypothesis is unrelated to the premise, you should select neutral.
    If the hypothesis contradicts the premise, you should select contradiction.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'index: label'. 

    The first five are examples with the labels provided.
    Examples:

    """

credence_prompt_preamble = """
    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to determine whether the hypothesis
    can be inferred from the premise.
     
    You will be given three options: entailment, neutral, and contradiction. 
    If the hypothesis can be inferred from the premise, you should select entailment.
    If the hypothesis is unrelated to the premise, you should select neutral.
    If the hypothesis contradicts the premise, you should select contradiction.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'index : premise credence : hypothesis credence : label'. 

    Event credence is the degree of belief that an event will occur on a score between -3 and 3 where -3 is certainly won't happen and 3 is certainly did happen.
    The linguistic factors that effect event credence are:
    - The presence of modals
    - The presence of adverbs
    - The presence of negation
    - The presence of tense
    - The infinitive verb that is subordinate to the main verb such as expected to, intended to, etc.
    - The event is a subordinated clause of a speech verb such as said, told, etc. or a belief verb such as believe, think, etc.
    
    Credence has an effect on the entailment task. Because if the premise and hypothesis are talking about the same event then they entail if the credences are the same.
    Use this reasoning to predict the label for the examples.
    The first five are examples with the labels provided. The credences provided in the examples may be wrong.
    Examples:

    """

"""The first five are examples with the labels provided."""
cot_prompt = """
Event credence is the degree of belief that an event will occur.
    The linguistic factors that effect event credence are:
    - The presence of modals
    - The presence of adverbs
    - The presence of negation
    - The presence of tense
    - The infinitive verb that is subordinate to the main verb such as expected to, intended to, etc.
    - The event is a subordinated clause of a speech verb such as said, told, etc. or a belief verb such as believe, think, etc.

    Use this reasoning to predict the label for the examples.
"""

class NLIDataPreparer():
    def __init__(self):
        self._data_readers = {
            'mnli': MNLIDataReader(),
            'snli': SNLIDataReader(),
            'anli': ANLIDataReader(),
        } 

        self._chaos_data_readers = {
            'mnli': ChaosMNLIDatareader(),
            'snli': ChaosSNLIDatareader(),
            'anli': ChaosANLIDatareader(),
        }

    def load(self):
        k = 0
        jl = JadeLogger()
        gpt_answer = {}
        true_answers = {}
        chaos_data_reader = self._chaos_data_readers['mnli']
        data = chaos_data_reader.read_file('test').data()
        system_prompt = str(prompt_preamble)
        for datumi, datum in enumerate(data[:5], start=1):

            system_prompt += f'{datum.uid()} Premise: ' + datum.sentence_1() + '\n'
            system_prompt += 'Hypothesis: ' + datum.sentence_2() + '\n'
            
            #system_prompt += f'{datum.uid()} : 2 : 2 : ' + datum.label() +  '\n\n'
            system_prompt += f'{datum.uid()} : ' + datum.label() +  '\n\n'

        while k * 40 < len(data):
            print(k, len(data) // 40)
            user_prompt = ""
            user_prompt += 'Fill answer the following: \n\n'
            for datumi, datum in enumerate(data[5 + (k*40): (k+1) * 40 + 5], start=6):
                true_answers[datum.uid()] = datum.label()
                user_prompt += f'{datum.uid()} Premise: ' + datum.sentence_1() + '\n'
                user_prompt += 'Hypothesis: ' + datum.sentence_2() + '\n\n'
            print('sending prompt')
            answer = gpt_4(system_prompt, user_prompt)
            print('received response')
            print(answer)
            for line in answer.split('\n'):
                if ':' in line:
                    try:
                        #index, premise_credence, hypothesis_credence, label = line.split(':')
                        index, label = line.split(':')
                        #gpt_answer[index.strip()] = [label.strip(), premise_credence.strip(), hypothesis_credence.strip()]
                        gpt_answer[index.strip()] = [label.strip(), '', '']

                    except ValueError:
                        print(len(line.split(':')))
                        continue

            print(self.f1_score(true_answers, gpt_answer))
            k += 1
            location = jl.file_manager.data_filepath('gpt_answers_few_shot_no_explain.json')
            with open(location, 'wt') as f:
                f.write(json.dumps(gpt_answer))

    def f1_score(self, true_answers, gpt_answers):
        tp = 0
        fp = 0
        fn = 0
        for uid, label in true_answers.items():
            if uid not in gpt_answers:
                continue
            if label[0] == gpt_answers[uid][0][0]:
                tp += 1
            else:
                fp += 1
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

if __name__ == '__main__':
    Config.instance()
    data_preparer = NLIDataPreparer()
    data_preparer.load()