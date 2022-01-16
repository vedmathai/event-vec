import os
import json
import re


class FerverousDataset:
    def __init__(self):
        self._folder = '/media/vedmathai/The Big One/python/temporal/FeverousWikiv1'
        self._files = []
        self._current_file_index = 0
        self._current_index = 0

    def load(self):
        self._files = os.listdir(self._folder)
        file_name = self._files[0]
        self._contents = self.read_file(file_name)

    def get_next_article(self):
        if self._current_index+1 >= len(self._contents) and self._current_file_index < len(self._files) - 1:
            self._contents = self.read_file(self._files[self._current_file_index+1])
            self._current_index = 0
            self._current_file_index += 1
        elif self._current_index+1 >= len(self._contents) and self._current_file_index >= len(self._files) - 1: 
            return None
        self._current_index += 1
        return self._contents[self._current_index]

    def read_file(self, file_name):
        contents = []
        file_path = os.path.join(self._folder, file_name)
        with open(file_path, encoding='utf-8') as f:
            try:
                for line_i, line in enumerate(f):
                    sentence_content = []
                    content = json.loads(line)
                    for key in content:
                        if key[0:8] == 'sentence':
                            sentence_content += [self.fix_sentence(content[key])]
                    sentence_content = ' '.join(sentence_content)
                    contents += [sentence_content]
                    #if 'cicero' in sentence_content.lower():
                    #    print(content['title'])
            except UnicodeDecodeError:
                print('unicode_error:', file_name)
                return []
        return contents


    def fix_sentence(self, sentence):
        sentence = re.sub('(\[\[[^\|]*\||\]\])',  '', sentence)
        return sentence

    def contents(self):
        return self._contents
