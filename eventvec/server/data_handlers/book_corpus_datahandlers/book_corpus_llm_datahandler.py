from transformers import RobertaTokenizer


from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer  # noqa
from eventvec.server.data_handlers.book_corpus_datahandlers.bookcorpus_datahandler import BookCorpusDatahandler  # noqa
from eventvec.server.data_handlers.book_corpus_input_datum.book_corpus_model_input_data import BookCorpusModelInputData  # noqa
from eventvec.server.data_handlers.book_corpus_input_datum.book_corpus_model_input_datum import BookCorpusModelInputDatum  # noqa

verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

TRAIN_NUMBER = 3
TEST_NUMBER = 1

tag2id = {
    "VB": 0,
    "VBD": 1,
    "VBG": 2,
    "VBN": 3,
    "VBP": 4,
    "VBZ": 5,
    "Future": 6,
}

future_modals = [
    'will',
    'going to',
    'would',
    'could',
    'might',
    'may',
    'can',
    'going to',
]


class BookCorpusLLMDatahandler():
    def __init__(self):
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._book_corpus_data_handler = BookCorpusDatahandler()
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._model_input_data = BookCorpusModelInputData()

    def load(self):
        filenames = self._book_corpus_data_handler.book_corpus_file_list()
        end = TRAIN_NUMBER + TEST_NUMBER
        for filenamei, filename in enumerate(filenames[:end]):
            data_division_type = self.data_division_type(filenamei)
            if data_division_type in ['train', 'test']:
                file_contents = self._book_corpus_data_handler.read_file(filename)  # noqa
                file_contents = file_contents[:1000000-1]
                self.featurize(file_contents, data_division_type)

    def data_division_type(self, i):
        if i < TRAIN_NUMBER:
            return 'train'
        if TRAIN_NUMBER <= i < TRAIN_NUMBER + TEST_NUMBER:
            return 'test'

    def featurize(self, file_contents, data_division_type):
        featurized = self._linguistic_featurizer.featurize_document(file_contents)  # noqa
        for sentence in featurized.sentences():
            for token in sentence.tokens():
                if token.tag() in tag2id:
                    datum = self.create_datum(sentence, token)
                    if data_division_type == 'train':
                        self._model_input_data.add_train_datum(datum)
                    if data_division_type == 'test':
                        self._model_input_data.add_test_datum(datum)

    def create_datum(self, sentence, token):
        datum = BookCorpusModelInputDatum()
        datum.set_entity_token_i(token.i_in_sentence())
        datum.set_aspect(token.aspect())
        datum.set_tense(token.tense())
        datum.set_pos(token.pos())
        datum.set_tag(token.tag())
        datum.set_original_sentence(sentence.text())
        datum.set_target(tag2id[token.tag()])
        is_future = self.check_future_tense(sentence, token.i_in_sentence())
        if is_future is True:
            datum.set_tag('Future')
            datum.set_target(tag2id['Future'])
        encoded_sentence = self.llm_encode_sentence(sentence.text())
        datum.set_sentence_encoded(encoded_sentence)
        datum.set_is_trainable(True)
        if token.i_in_sentence() > 199:
            datum.set_is_trainable(False)
        return datum

    def llm_encode_sentence(self, sentence):
        sentence_encoding = self._tokenizer(
            [sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        return sentence_encoding

    def model_input_data(self):
        return self._model_input_data

    def check_future_tense(self, sentence, entity_idx):
        window_start = entity_idx - 6
        window_end = entity_idx
        window = sentence.tokens()[window_start: window_end]
        window = ' '.join(token.text() for token in window)
        if any(i in window for i in future_modals):
            return True
        return False
