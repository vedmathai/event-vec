import json

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer  # noqa
from eventvec.server.data.book_corpus.book_corpus_datahandlers.book_corpus_llm_datahandler import BookCorpusDatahandler  # noqa
from eventvec.server.data.book_corpus.book_corpus_datahandlers.torque_noun_events import torque_noun_events
from eventvec.server.data.book_corpus.book_corpus_datahandlers.torque_all_words import all_torque_words

prep_set = set(['after', 'during', 'before', "between", "by", "following", "for", "from", "on", "since", "till", "to", "until", "within", "while", "except"])
prep_set = set(['after', 'during', 'before', "since", "until", "while"])

class BookCorpusLLMDatahandler():
    def __init__(self):
        self._book_corpus_data_handler = BookCorpusDatahandler()
        self._linguistic_featurizer = LinguisticFeaturizer()

    def load(self):
        filenames = self._book_corpus_data_handler.book_corpus_file_list()
        end = 20000
        noun_events = set()
        count = {i: 0 for i in prep_set}
        for filenamei, filename in enumerate(filenames[:end]):
            print(filenamei)
            file_contents = self._book_corpus_data_handler.read_file(filename)  # noqa
            file_contents = file_contents[:1000000-1]
            sentences = file_contents.split('.')
            for sentence in sentences:
                for i in prep_set:
                    if i in sentence:
                        count[i.lower()] += 1
            print(count)

        
if __name__ == "__main__":
    bcld = BookCorpusLLMDatahandler()
    bcld.load()