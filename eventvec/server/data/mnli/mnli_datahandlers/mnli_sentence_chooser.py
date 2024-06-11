from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader  # noqa
from collections import defaultdict

contrast_words = [
    'But', 'However', 'Yet', 'Although', 'Though',
    'Even though', 'Despite', 'In spite of', 'Nevertheless', 'Nonetheless',
    'On the other hand', 'Whereas', 'While', 'Still', 'Conversely',
    'Instead', 'Alternatively', 'In contrast', 'Rather',
    'Though', 'Nonetheless', 'On the contrary', 'Notwithstanding',
    'Even so', 'For all that', 'Be that as it may', 'Despite that',
    'Albeit', 'Except that', 'Admittedly', 'All the same', 'After all',
    'Otherwise', 'Regardless', 'Conversely', 'Apart from', 'Anyhow', 'And yet',
    'At the same time', 'Even if', 'In any case', 'In any event', 'Except',
    'Though', 'Still and all', 'Differently', 'Else',
    'Alternatively', 'If not',
]
contrast_words = [word.lower() for word in contrast_words][9:10]

class NLISentenceChooser():
    def __init__(self):
        self._data_readers = {
            'mnli': MNLIDataReader(),
            
        } 

    def load(self, run_config):
        data_reader = self._data_readers['mnli']
        data = data_reader.read_file('train').data()
        sentence_counter = 0
        word_counter = defaultdict(int)
        contrast_sentence_counter = 0
        count = 0
        for d in data:
            sentence1 = d.sentence_1().lower()
            sentence2 = d.sentence_2().lower()
            sentence_counter += 2
            for word in contrast_words:
                if word in sentence1:
                    word_counter[word] += 1
                    contrast_sentence_counter += 1
                    print(count, d.sentence_1())
                    print('-' * 50)
                    count += 1
                if word in sentence2:
                    word_counter[word] += 1
                    contrast_sentence_counter += 1
                    print(count, d.sentence_2())
                    print('-' * 50)
                    count += 1
            if count >= 10:
                    break
        print('Total sentences:', contrast_sentence_counter, sentence_counter, contrast_sentence_counter/sentence_counter)
        print(word_counter)

if __name__ == '__main__':
    NLISentenceChooser().load('mnli')