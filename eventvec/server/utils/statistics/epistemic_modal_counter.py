import json
from collections import defaultdict


from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer  # noqa
from eventvec.server.data.book_corpus.book_corpus_datahandlers.book_corpus_llm_datahandler import BookCorpusDatahandler  # noqa
from eventvec.server.data.wikipedia.datahandlers.wiki_datahandler import WikiDatahandler
from eventvec.server.data.nyt.nyt_datahandlers.nyt_datahandler import NYTDatahandler
from eventvec.server.data.hansard.hansard_datahandlers.hansard_datahandler import HansardDatahandler
from eventvec.server.data.maec.maec_datahandlers.maec_datahandler import MAECDatahandler
from eventvec.server.data.politosphere.politosphere_datahandlers.politosphere_datahandler import PolitosphereDatahandler
from eventvec.server.data.timebank.timebank_reader.timebank_reader import TimeMLDataReader 
from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader
from eventvec.server.data.torque.readers.torque_datareader import TorqueDataReader
from eventvec.server.data.timebank.timebank_reader.te3_silver_reader import TE3SilverDatareader
from eventvec.server.data.timebank.timebank_reader.te3_gold_reader import TE3GoldDatareader
from eventvec.server.common.lists.said_verbs import said_verbs, future_said_verbs
from eventvec.server.utils.general import token2parent, token2tense
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer




class EpistemicModalCounter():
    def __init__(self):
        self._book_corpus_data_handler = BookCorpusDatahandler()
        self._wiki_data_handler = WikiDatahandler()
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._nyt_corpus_handler = NYTDatahandler()
        self._hansard_handler = HansardDatahandler()
        self._maec_handler = MAECDatahandler()
        self._politosphere_handler = PolitosphereDatahandler()
        self._mnli_handler = MNLIDataReader()
        self._time_ml_handler = TimeMLDataReader()
        self._torque_handler = TorqueDataReader()
        self._te3_silver_handler = TE3SilverDatareader()
        self._te3_gold_handler = TE3GoldDatareader()
        self._corpus = 'wikidata'
        self._factuality_categorizer = FactualityCategorizer()


    def filenames(self):
        filename2list = {
            'wikidata': self._wiki_data_handler.wiki_file_list,
            'bookcorpus': self._book_corpus_data_handler.book_corpus_file_list,
            'nytcorpus': self._nyt_corpus_handler.nyt_file_list,
            'hansard': self._hansard_handler.hansard_file_list,
            'mnli': self._mnli_handler.mnli_file_list,
            'timebank': self._time_ml_handler.list_extra,
            'torque': self._torque_handler.file_list,
            'te3_silver': self._te3_silver_handler.list_folder,
            'te3_gold': self._te3_gold_handler.list_folder,
            'maec': self._maec_handler.maec_file_list,
            'politosphere': self._politosphere_handler.politosphere_file_list,
        }
        return filename2list[self._corpus]()
    
    def read_file(self, filename):
        filename2file = {
            'wikidata': self._wiki_data_handler.read_file,
            'bookcorpus': self._book_corpus_data_handler.read_file,
            'nytcorpus': self._nyt_corpus_handler.read_file,
            'hansard': self._hansard_handler.read_file,
            'mnli': self._mnli_handler.read_file,
            'timebank': self._time_ml_handler.read_file_text,
            'torque': self._torque_handler.torque_sentences,
            'te3_silver': self._te3_silver_handler.timebank_documents_contents,
            'te3_gold': self._te3_gold_handler.timebank_documents_contents,
            'maec': self._maec_handler.read_file,
            'politosphere': self._politosphere_handler.read_file
        }
        fn = filename2file[self._corpus]
        return fn(filename)
    
    def load(self):
        filenames = self.filenames()
        end = 20000
        sentence_counter = 0
        counter = defaultdict(int)
        stop = 100000

        for filenamei, filename in enumerate(filenames[:end]):
            print(sentence_counter)
            file_contents_list = self.read_file(filename)  # noqa
            if sentence_counter > stop: #100000
                    break
            if self._corpus == 'mnli':
                file_contents_list = file_contents_list.data()
            for file_contentsi, file_contents in enumerate(file_contents_list):
                if self._corpus == 'mnli':
                    file_contents = ' '.join([file_contents.sentence_1(), file_contents.sentence_2()])
                if sentence_counter > stop: #100000
                    break
                file_contents = file_contents[:1000000-1]
                fdoc = self._linguistic_featurizer.featurize_document(file_contents)
                for fsent in fdoc.sentences():
                    sentence_counter += 1
                    if sentence_counter > stop: #100000
                        break
                    if sentence_counter % 1000 == 0:
                        print(sentence_counter)
                    for token in fsent.tokens():
                        if token.pos() in ['VERB', 'AUX']:
                            features_array = self._factuality_categorizer.categorize(fsent.text(), token.text())
                            features_array_dict = features_array.to_dict()
                            for key in features_array.to_dict():
                                if features_array_dict[key] is True:
                                    counter[key] += 1
                            if features_array_dict['has_modal'] is True and features_array_dict['is_subordinate_of_said'] is True:
                                counter['has_modal_is_subordinate_of_said'] += 1
        for key in counter:
            print(key, counter[key]/sentence_counter)


if __name__ == "__main__":
    emc = EpistemicModalCounter()
    emc.load()
