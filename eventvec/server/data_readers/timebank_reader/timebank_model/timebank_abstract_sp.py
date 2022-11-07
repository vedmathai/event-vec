class AbstractSentencePart():
    def __init__(self):
        self._sentence_token_i = None
        self._start_token_i = None
        self._end_token_i = None

    def sentence_token_i(self):
        return self._sentence_token_i

    def start_token_i(self):
        return self._start_token_i

    def end_token_i(self):
        return self._end_token_i

    def set_sentence_token_i(self, sentence_token_i):
        self._sentence_token_i = sentence_token_i

    def set_start_token_i(self, start_token_i):
        self._start_token_i = start_token_i

    def set_end_token_i(self, end_token_i):
        self._end_token_i = end_token_i
