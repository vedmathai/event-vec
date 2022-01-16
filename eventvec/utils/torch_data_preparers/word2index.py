from collections import defaultdict
import re

from eventvec.utils.data_readers.fervous_wikipedia_reader import FerverousDataset

MIN_WORD_COUNT = 5


class Word2Index:
    def __init__(self):
        pass

    def word2index(self):
        dataset = FerverousDataset()
        dataset.load()
        words_counter = defaultdict(int)
        count = 0
        while True:
            text = dataset.get_next_article()
            if text is None:
                break
            text = re.sub('[\.,":;]',  '', text)
            for token in text.lower().split():
                if True or token.isalpha():
                    words_counter[token] += 1  # Another source of improvement is the way the tokens are chosen. eg numbers, years, apostrophes, other special characters.
            if count % 1000 == 0:
                above_x = len([word for word, count in words_counter.items() if count > MIN_WORD_COUNT])
                total = len(words_counter.keys())
                ratio = int(float(above_x) / total * 100)
                print(count, above_x, total, ratio)
            count += 1
        with open('local/data/word2index.txt', 'wt') as f:
            words = [word for word, count in words_counter.items() if count > MIN_WORD_COUNT]
            words.sort()
            for word in words:
                f.write(f'{word}\n')


if __name__ == '__main__':
    word2index = Word2Index()
    word2index.word2index()
