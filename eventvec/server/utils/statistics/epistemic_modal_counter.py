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
from eventvec.server.data.mnli.mnli_datahandlers.mnli_datahandler import MNLIDatahandler
from eventvec.server.data.torque.readers.torque_datareader import TorqueDataReader
from eventvec.server.data.timebank.timebank_reader.te3_silver_reader import TE3SilverDatareader
from eventvec.server.data.timebank.timebank_reader.te3_gold_reader import TE3GoldDatareader
from eventvec.server.common.lists.said_verbs import said_verbs, future_said_verbs
from eventvec.server.utils.general import token2parent, token2tense

said_verbs = said_verbs | future_said_verbs
auxs = {'can', 'could', 'would', 'will', 'wo', 'must', 'll', 'should', 'shall', 'might', 'may', "'d"}
                
adverbs = set(["probably", "possibly", "clearly", "obviously",
               "presumably", "evidently", "apparently", "supposedly",
               "conceivably", "undoubtedly", "allegedly", "reportedly",
               "arguably", "unquestionably", "seemingly", "certainly",
               "likely", "credibly", "feasibly", "plausably", "reasonably"
               "believably", "seemingly", "surely", "precisely", "plainly",
               "obviously", "definitely", "distinctly", "undeniably",
               "unmistakably", "undeniably", "transparently", "recognizably",
               "perceptibly", "patently", "overtly", "noticeably", "markedly", "manifestly",
               "lucidly", "indubitably", "incontroverbly", "incotestably", "discernibly", "conspicously",
               ])

adverb_adjectives = set(["probable", "possible", "clear", "obvious",
               "presumable", "evident", "apparent", "supposed",
               "conceivable", "alleged", "doubtless", "reported",
               "arguable", "unquestionable", "seeming", "certain",
               "credible", "feasible", "plausable", "reasonable", "likely",
               "believable", "seeming", "sure", "precise", "plain",
               "obvious", "definite", "distinct", "undeniable",
               "unmistakable", "undeniable", "transparent", "recognizable",
               "perceptible", "patent", "overt", "noticeable", "marked", "manifest",
               "lucid", "incontroverble", "discernible", "conspicous",
               ])


class EpistemicModalCounter():
    def __init__(self):
        self._book_corpus_data_handler = BookCorpusDatahandler()
        self._wiki_data_handler = WikiDatahandler()
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._nyt_corpus_handler = NYTDatahandler()
        self._hansard_handler = HansardDatahandler()
        self._maec_handler = MAECDatahandler()
        self._politosphere_handler = PolitosphereDatahandler()
        self._mnli_handler = MNLIDatahandler()
        self._time_ml_handler = TimeMLDataReader()
        self._torque_handler = TorqueDataReader()
        self._te3_silver_handler = TE3SilverDatareader()
        self._te3_gold_handler = TE3GoldDatareader()
        self._corpus = 'politosphere'

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
        aux_set = set()
        said_counter = 0
        aux_counter = 0
        verb_counter = 0
        aux_in_said = 0
        verb_in_said = 0
        adverb_counter = 0
        adverb_in_said = 0
        sentence_counter = 0
        verb_in_said_quotes = 0
        tense_counter = defaultdict(int)

        for filenamei, filename in enumerate(filenames[:end]):
            file_contents_list = self.read_file(filename)  # noqa
            for file_contentsi, file_contents in enumerate(file_contents_list):
                if sentence_counter > 100000:
                    break
                file_contents = file_contents[:1000000-1]
                fdoc = self._linguistic_featurizer.featurize_document(file_contents)
                for fsent in fdoc.sentences():
                    sentence_counter += 1
                    for token in fsent.tokens():
                        if token.pos() in ['VERB', 'AUX'] and token.dep() != 'aux':
                            if token.text() in said_verbs:
                                said_counter += 1
                            verb_counter += 1
                            for childk, childv in token.children().items():
                                if childk == 'aux':
                                    for v in childv:
                                        if v.text() in auxs:
                                            aux_counter += 1
                                            verb_counter -= 1
                                            parent = v.parent()
                                            while parent is not None:
                                                if parent.dep() in ['ccomp', 'xcomp'] and parent.parent().text() in said_verbs:
                                                    aux_in_said += 1
                                                parent = parent.parent()
                                if childk == 'advmod':
                                    for v in childv:
                                        if v.text() in adverbs:
                                            adverb_counter += 1
                                            parent = v.parent()
                                            while parent is not None:
                                                if parent.dep() in ['ccomp', 'xcomp'] and parent.parent().text() in said_verbs:
                                                    adverb_in_said += 1
                                                parent = parent.parent()
                                if childk == 'acomp':
                                    for v in childv:
                                        if v.text() in adverb_adjectives:
                                            adverb_counter += 1
                                            parent = v.parent()
                                            while parent is not None:
                                                if parent.dep() in ['ccomp', 'xcomp'] and parent.parent().text() in said_verbs:
                                                    adverb_in_said += 1
                                                parent = parent.parent()
                            parent = token.parent()
                            while parent is not None:
                                if parent.dep() in ['ccomp', 'xcomp'] and parent.parent().text() in said_verbs:
                                    for dep, tokens in parent.parent().children().items():
                                        for t in tokens:
                                            if t.text() == '"' or t.text() == "'":
                                                verb_in_said_quotes += 1
                                    verb_in_said += 1
                                    parent_tense, parent_aspect = token2tense(fsent.text(), parent.parent())
                                    token_tense, token_aspect = token2tense(fsent.text(), token)
                                    tense_tuple = (parent_tense, parent_aspect, token_tense, token_aspect)
                                    tense_counter[tense_tuple] += 1
                                parent = parent.parent()
                if file_contentsi % 1000 == 0 or sentence_counter > 1000:
                    try:
                        print(
                            'aux_counter', aux_counter,
                            'verb_counter', verb_counter,
                            'sentence_counter', sentence_counter,
                            'aux_counter/verb_counter', aux_counter/verb_counter,
                            'aux_in_said/aux_counter', aux_in_said/aux_counter, 
                            'aux_in_said/verb_counter', aux_in_said/verb_counter,
                            'aux_in_said/verb_in_said', aux_in_said/(verb_in_said - aux_in_said),
                            'adverb_counter/verb_counter', adverb_counter/verb_counter,
                            'adverb_in_said/adverb_counter', adverb_in_said/adverb_counter,
                            'adverb_in_said/verb_counter', adverb_in_said/verb_counter,
                            'adverb_in_said/verb_in_said', adverb_in_said/verb_in_said,
                            'said_counter/verb_counter', said_counter/verb_counter,
                            'verb_in_said/verb_counter', verb_in_said/verb_counter,
                            'verb_in_said_quotes/verb_in_said', verb_in_said_quotes/verb_in_said,
                        )
                        for k, v in sorted(tense_counter.items(), key=lambda x: x[1]):
                            print(k, v / verb_in_said)
                        print(sum([i[1]/verb_in_said for i in tense_counter.items()]))
                    except:
                        pass


if __name__ == "__main__":
    emc = EpistemicModalCounter()
    emc.load()
