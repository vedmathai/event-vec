import re
import csv
from jadelogs import JadeLogger
import random
import numpy as np

from eventvec.server.featurizers.sentence_masker.sentence_masker import SentenceMasker
from eventvec.server.featurizers.sentence_masker.sentence_masker_simpler import SentenceMaskerSimpler

from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.data.alphanli.alphanli_reader import AlphaNLIDataReader



class ConnectorMLMDatum:
    def __init__(self):
        self._uid = None
        self._text = None
        self._label = None

"""
    Use 'and' like in first event 'and' second event, when the second event does not contrast first event. For example, I went to the store and bought some milk. Going to store does not contrast buying milk or vice versa.
    Use "but" like in first event but second event, when second event contrasts first event and highlights surprise given the first event. For example, I went to the store but I didn't buy anything. Going to store contrasts buying nothing.
    Use 'so' like in first event so second event, when second event follows as a natural continuation of first event. For example, I went to the store so I bought some milk. Buying milk is a natural continuation of going to the store.
    Use 'because' like  in first event because second event, when the second event is the reason the first event is happening. For example, I went to the store because I needed some milk. Going to store is happening because I needed milk.
    Use 'though' like in 'first event though second event', when first event contrasts second event and highlights surprise given the second event. For example, I went to the store though I didn't need anything. Going to store contrasts not needing anything.

    Hint:
    To fill in the mask analyse the following structure:
    A <mask> B

    And ask the following questions:
    A but B: if A then B is surprising.
    A so B: if A then B follows. A is the reason.
    A though B: if B then A is surprising.
    A because B: if A then B is the reason. 
    A and B: not contrasting not causal

        
"""

system_prompt_larger_2 = """
    [INST] <<SYS>>
    The task is to Fill in the blanks with only the following words ["because", "so", "but", "and", "though"]. 
    If there is no appropriate word, then leave it unanswered. NO other word is allowed.




    Hint:
        Hint:
    To fill in the mask analyse the following structure:
    A <mask> B

    And ask the following questions:
    A but B: if A then B is surprising.
    A so B: if A then B follows. A is the reason.
    A though B: if B then A is surprising.
    A because B: if A then B is the reason. 
    A and B: not contrasting not causal

    Such as the following examples:
    PASSAGE : 9870 : I want to drive the Lamborghini <mask> I don't know how to. Therefore, I crashed the car.
    Answer: 9870 : but

    PASSAGE : 9850 : Alex failed the test <mask> he studied hard. That was sad.
    Answer: 9850 : but

    PASSAGE: 10450 : I am thirsty. But I didn't get a drink <mask>, I still want more water.
    Answer: 10450 : so
    
    PASSAGE 56003: The win is imminent <mask> the team is playing well Therefore, the crowd is cheering.
    Answer: 56003 :  because

    PASSAGE 4542: He arrived and immediately started playing.
    Answer: 4542 :  and

    Fill in the blanks strictly only with the following words ["because", "so", "but", "and", "though"].
    If a word doesn't apply then leave it blank. 
    Use the form discovered to fill in the blanks.
    Reply in the form
    User Answer : <uid> : <connector>
     <</SYS>> 
"""


system_prompt_larger = """
    [INST] <<SYS>>
    The task is to Fill in the blanks with only the following words ["because", "so", "but", "and", "though"]. 
    If there is no appropriate word, then leave it unanswered. NO other word is allowed.




    Hint:
    First step: Identify the clauses being connected by the connector that is masked.
    To fill in the mask analyse the following structure:
    A <mask> B

    Compare following examples:

    Example 1: I went to the store but I didn't buy anything. 
    A: Going to store B: I didn't buy anything.
    Explanation: Not buying anything is surprising given going to the store. A but B.

    Example 2: I went to the store so I bought some milk. 
    A: Going to store B: Buying milk.
    Explanation: Going to the store was the cause of buying milk. A so B.

    Example 3: I went to the store because I needed some milk. 
    A: Going to store B: I needed some milk.
    Explanation: Needing milk was the cause of going to the store. A because B.

    Example 4: I went to the store though I didn't need anything.
    A: I went to the store B: I didn't need anything.
    Explanation: Going to store was surprising given not needing anything. A though B.

    Example 5: I went to the store and went to the palour. 
    A: I went to the store B: went to the palour
    Explanation: None of the above connections apply.

    Apply these explanations to choose the correct connector.

    Such as the following examples:
    PASSAGE : 9870 : I want to drive the Lamborghini <mask> I don't know how to. Therefore, I crashed the car.
    A: I want to drive the Lamborghini B: I don't know how to.
    Explanation: Not knowing how to drive is surprising given wanting to drive the Lamborghini. 
    Structure: B is surprising given A. 
    Form: A but B.
    Answer: 9870 : but

    PASSAGE : 9850 : Alex failed the test <mask> he studied hard. That was sad.
    A: Alex failed test B: He studied hard.
    Explanation: Alex failing the test is surprising given he studied hard. 
    Structure: A is surprising given B. 
    Form: A though B.
    Answer: 9850 : but

    PASSAGE: 10450 : I am thirsty. But I didn't get a drink <mask>, I still want more water.
    A: I didn't get a drink. B: I still want more water.
    Explanation: Not getting a drink causes I still want more water. 
    Structure: A causes B. 
    Form: A so B.
    Answer: 10450 : so
    
    PASSAGE 56003: The win is imminent <mask> the team is playing well Therefore, the crowd is cheering.
    A: The win is imminent B: the team is playing well
    Explanation: The team playing well causes the win being imminent.
    Structure: B causes A.
    Form: A because B.
    Answer: 56003 :  because

    PASSAGE 4542: He arrived and immediately started playing.
    A: He arrived B: immediately started playing
    Explanation: He arrived doesn't cause immediately starting playing. He started playing is not surprising given he arrived.
    Structure: A and B.
    Form: A and B.
    Answer: 4542 :  and

    Fill in the blanks strictly only with the following words ["because", "so", "but", "and", "though"].
    If a word doesn't apply then leave it blank. 
    Use the form discovered to fill in the blanks.
    Reply in the form
    A: <clause 1> B: <clause 2>
    Explanation: <explanation>
    Structure: <structure>
    Form: <form>
    User Answer : <uid> : <connector>
     <</SYS>> 
"""

system_prompt = """
    [INST] <<SYS>>
    The task is to Fill in the blanks with only the following words ["because", "so", "but", "and", "though"]. 
    If there is no appropriate word, then leave it unanswered. NO other word is allowed.

    Such as the following examples:
    PASSAGE : 9870 : I want to drive the Lamborghini <mask> I don't know how to. Therefore, I crashed the car.
    Answer: 9870 : but

    PASSAGE : 9850 : Alex failed the test <mask> he studied hard. That was sad.
    Answer: 9850 : but

    PASSAGE: 10450 : I am thirsty. But I didn't get a drink <mask>, I still want more water.
    Answer: 10450 : so
    
    PASSAGE 56003: The win is imminent <mask> the team is playing well Therefore, the crowd is cheering.
    Answer: 56003 :  because

    PASSAGE 4542: He arrived and immediately started playing.
    Answer: 4542 :  and

    Fill in the blanks strictly only with the following words ["because", "so", "but", "and", "though"].
    If a word doesn't apply then leave it blank. 
    User Answer : <uid> : <connector>
     <</SYS>> 
"""


user_prompt = """
    User Prompt:
    PASSAGE : {uid} : {passage}
    [/INST] 
"""

connectors = ['and', 'because', 'but', 'so', 'though']
class Llama_NLI:
    def __init__(self):
        self._alphanli_datareader = AlphaNLIDataReader()
        self._jade_logger = JadeLogger()
        self._filepath = self._jade_logger.file_manager.data_filepath('llama_mlm.csv')

    def read_data(self):
        data = []
        with open('/home/lalady6977/oerc/projects/data/roc_masked_connectors.csv') as f:
            reader = csv.reader(f, delimiter='\t')
            for r in reader:
                datum = ConnectorMLMDatum()
                datum._uid = r[0]
                datum._label = r[1]
                datum._text = r[2].replace('[MASK]', '<mask>')
                data.append(datum)
        return data

    def mask(self, train_test='train'):
        data = self.read_data()
        random.seed(0)
        random.shuffle(data)
        data = data[:500]
        csv_data = []
        seen = set()
        true_p = {'and': 0, 'but': 0, 'though':0, 'so': 0, 'because': 0}
        false_n = {'and': 0, 'but': 0, 'though':0, 'so': 0, 'because': 0}
        false_p = {'and': 0, 'but': 0, 'though':0, 'so': 0, 'because': 0}
        connector_counter = {i: 0 for i in connectors}
        required_answers = {}
        llm_answers = {}

        for datum_i, datum in enumerate(data):

            print(datum_i)
            required_answers[datum._uid] = datum._label
            llm_answers[datum._uid] = ''
            passage = f"{datum._text}."
            key = passage
            if key in seen:
                continue
            seen.add(key)
                
            user_prompt_formated = user_prompt.format(passage=passage, uid=datum._uid)
            use = False
            print('---' * 50)
            print('original_passage', passage)
            print('')
            print(connector_counter)
            response = ''
            response = llama_3(system_prompt, user_prompt_formated)
            for line in response.split('\n'):
                if any(i in line for i in ['Explanation', 'Form', 'Structure']):
                    print(line)
                if 'User Answer' in line and '<connector>' not in line:
                    split_line = line.split(':')
                    if len(split_line) == 3:
                        print(datum._label, split_line[2].strip())
                        llm_answers[split_line[1].strip()] = split_line[2].strip()
            true_p, false_p, false_n = self.check_metric(llm_answers, required_answers, true_p, false_p, false_n)
            f1s = []
            for key in connectors:
                f1 = 0
                precision = 0
                if true_p[key] + false_p[key] > 0:
                    precision = true_p[key] / (true_p[key] + false_p[key])
                recall = 0
                if true_p[key] + false_n[key] > 0:
                    recall = true_p[key] / (true_p[key] + false_n[key])
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                f1s.append(f1)
                print(key, precision, recall, f1)
            print(np.mean(f1s))
            
        with open(self._filepath, 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            random.shuffle(csv_data)
            for d in csv_data:
                writer.writerow(d)

    def check_metric(self, llm_answers, required_answers, true_p, false_p, false_n):
        for key in llm_answers.keys():
            if key in required_answers:
                if required_answers[key] == 'but':
                    if 'but' in llm_answers[key].split():
                        true_p['but'] += 1
                    else:
                        false_n['but'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] == 'and':
                    if 'and' in llm_answers[key].split():
                        true_p['and'] += 1
                    else:
                        false_n['and'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] == 'because':
                    if 'because' in llm_answers[key].split():
                        true_p['because'] += 1
                    else:
                        false_n['because'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] in ['so', 'therefore']:
                    if any(i in llm_answers[key].split() for i in ['so', 'therefore']):
                        true_p['so'] += 1
                    else:
                        false_n['so'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] in ['though']:
                    if 'though' in llm_answers[key].split():
                        true_p['though'] += 1
                    else:
                        false_n['though'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

        return true_p, false_p, false_n

if __name__ == '__main__':
    Llama_NLI().mask('train')