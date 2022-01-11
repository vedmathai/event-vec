import os
import random
import torch
from bs4 import BeautifulSoup
from timebank_embedding.srl import predict_srl
from collections import defaultdict

class DataHandler():
    def __init__(self, variables):
        self._variables = variables
        self._file_name2file = {}
        self._word_counter = defaultdict(int)
        self._file_name2event_set = {}
        self.word2index = {}
        self.index2word = {}

    def load_data(self):
        files = os.listdir(self._variables.data_folder)
        for file_name in files[:3]:
            with open(os.path.join(self._variables.data_folder, file_name), 'rt') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.text
                text = text.strip().lower()
                self._file_name2file[file_name] = text

    def get_srl(self, text):
        srl_events = []
        for sentence in text.split('.'):
            prediction = predict_srl(sentence)
            words = prediction['words']
            verbs = prediction['verbs']
            tagi = 0
            for verb_set in verbs:
                tags = verb_set['tags']
                event = defaultdict(list)
                while tagi < len(tags):
                    tag = tags[tagi]
                    if tag[0] in ['B', 'I']:
                        event[tag[2:]] += [words[tagi]]
                    tagi += 1
                srl_events += [event]
        return srl_events

    def generate_all_words(self):
        self._word_counter = defaultdict(int)
        for text in self._file_name2file.values():
            text = text.split()
            for word in text:
                self._word_counter[word] += 1
        self._variables.vocabulary_size = len(self._word_counter.keys())

    def generate_word2index(self):
        self.generate_all_words()
        for wordi, (word, count) in enumerate(sorted(self._word_counter.items(), key=lambda x: x[1])):
            self.word2index[word] = wordi
            self.index2word[wordi] = word

    def generate_event_sets(self):
        for file_name, text in self._file_name2file.items():
            text = ' '.join(text.split())
            event_set = self.get_srl(text)
            self._file_name2event_set[file_name] = event_set

    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingPair(self):
        file_name = self.randomChoice(self._file_name2file)
        events = self._file_name2event_set[file_name]
        event_1_idx = random.randint(0, len(events) - 1)
        event_2_idx = event_1_idx + 1
        return (event_1_idx, event_2_idx)

    def categoryTensor(self, category):
        li = all_categories.index(category)
        tensor = torch.zeros(1, n_categories)
        tensor[0][li] = 1
        return tensor

    def inputTensor(self, phrase):
        tensor = torch.zeros(len(phrase.split()), 1, self._variables.vocabulary_size + 2)
        for li in range(len(phrase.split())):
            word = phrase.split()[li]
            tensor[li][0][self.word2index[word]] = 1
        return tensor

    def targetTensor(self, line):
        letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(n_letters - 1)
        return torch.LongTensor(letter_indexes)

    def randomTrainingExample(self):
        category, line = randomTrainingPair()
        category_tensor = categoryTensor(category)
        input_line_tensor = inputTensor(line)
        target_line_tensor = targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor