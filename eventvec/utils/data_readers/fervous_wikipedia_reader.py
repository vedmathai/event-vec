import os
import json
import re


class FerverousDataset:
    def __init__(self):
        self._contents = []

    def load(self):
        folder = '/media/vedmathai/The Big One/python/temporal/FeverousWikiv1'
        for file_name in os.listdir(folder)[0:10]:
            file_path = os.path.join(folder, file_name)
            with open(file_path, encoding='utf-8') as f:
                try:
                    for line_i, line in enumerate(f):
                        sentence_content = []
                        content = json.loads(line)
                        for key in content:
                            if key[0:8] == 'sentence':
                                sentence_content += [self.fix_sentence(content[key])]
                        sentence_content = ' '.join(sentence_content)
                        self._contents += [sentence_content]
                        if 'cicero' in sentence_content.lower():
                            print(content['title'])
                except UnicodeDecodeError:
                    continue
            print(len(self._contents))

    def fix_sentence(self, sentence):
        sentence = re.sub('(\[\[[^\|]*\||\]\])',  '', sentence)
        return sentence

    def contents(self):
        return self._contents
