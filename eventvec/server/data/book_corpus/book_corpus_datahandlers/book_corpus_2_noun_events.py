import json

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer  # noqa
from eventvec.server.data_handlers.book_corpus_datahandlers.bookcorpus_datahandler import BookCorpusDatahandler  # noqa
from eventvec.server.data_handlers.book_corpus_datahandlers.torque_noun_events import torque_noun_events
from eventvec.server.data_handlers.book_corpus_datahandlers.torque_all_words import all_torque_words

prep_set = set(['after', 'during', 'before', "between", "by", "following", "for", "from", "on", "since", "till", "to", "until", "within", "while", "except"])

class BookCorpusLLMDatahandler():
    def __init__(self):
        self._book_corpus_data_handler = BookCorpusDatahandler()
        self._linguistic_featurizer = LinguisticFeaturizer()

    def load(self):
        filenames = self._book_corpus_data_handler.book_corpus_file_list()
        end = 20000
        noun_events = set()
        for filenamei, filename in enumerate(filenames[:end]):
            file_contents = self._book_corpus_data_handler.read_file(filename)  # noqa
            file_contents = file_contents[:1000000-1]
            featurized = self._linguistic_featurizer.featurize_document(file_contents)
            for sentence in featurized.sentences():
                for token in sentence.tokens():
                    if token.text().lower() in prep_set:
                        for child in token.all_children():
                            if child.pos() == 'NOUN':
                                noun_events.add(child.text().lower())
            jaccard = len(noun_events & torque_noun_events) / float(len(noun_events | torque_noun_events))
            recall = len(noun_events & torque_noun_events) / float(len(torque_noun_events))
            precision = len(noun_events & torque_noun_events) / float(len(noun_events))
            f1 = 2 * (precision * recall) / (precision + recall)
            print('round: {}, jaccard: {}, recall: {}, precision: {}, f1: {}'.format(filenamei, jaccard, recall, precision, f1))
            with open('eventvec/server/data_handlers/book_corpus_datahandlers/book_corpus_noun_events.json', 'wt') as f:
                json.dump(sorted(list(noun_events)), f)

        
if __name__ == "__main__":
    bcld = BookCorpusLLMDatahandler()
    bcld.load()