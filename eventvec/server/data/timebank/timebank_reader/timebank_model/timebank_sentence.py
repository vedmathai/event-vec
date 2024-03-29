
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_enamex import TimebankEnamex  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_numex import TimebankNumex  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_signal import TimebankSignal  # noqa 
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_text_segment import TimebankTextSegment  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_timex import TimebankTimex  # noqa


class TimebankSentence:
    def __init__(self):
        self._sentence_i = None
        self._sequence = []
        self._sentence_start_token_global_i = None
        self._sentence_end_token_global_i = None

    def sequence(self):
        return self._sequence

    def sentence_i(self):
        return self._sentence_i

    def sentence_start_token_global_i(self):
        return self._sentence_start_token_global_i

    def sentence_end_token_global_i(self):
        return self._sentence_end_token_global_i

    def set_sentence_i(self, sentence_i):
        self._sentence_i = sentence_i

    def text(self):
        return ' '.join([i.text() for i in self._sequence])

    def set_sentence_start_token_global_i(self, sentence_start_token_global_i):
        self._sentence_start_token_global_i = sentence_start_token_global_i

    def set_sentence_end_token_global_i(self, sentence_end_token_global_i):
        self._sentence_end_token_global_i = sentence_end_token_global_i

    def append(self, item):
        self._sequence.append(item)

    @staticmethod
    def from_bs_obj(sentence, sentence_i, sentence_start_token_global_i,
                    timebank_document):
        timebank_sentence = TimebankSentence()
        timebank_sentence.set_sentence_start_token_global_i(sentence_start_token_global_i)
        timebank_sentence.set_sentence_end_token_global_i(sentence_start_token_global_i)
        children = list(sentence.children)
        creators = {
            'event': TimebankEvent,
            'enamex': TimebankEnamex,
            'numex': TimebankNumex,
            'signal': TimebankSignal,
            'timex3': TimebankTimex,
            'text_segment': TimebankTextSegment,
        }
        timebank_sentence.set_sentence_i(sentence_i)
        last_token = 0
        for c_i, c in enumerate(children):
            creator = creators.get(c.name, creators['text_segment'])
            obj, last_token = creator.from_bs_obj(c, c_i, last_token)
            timebank_sentence.append(obj)
            if c.name == 'event':
                timebank_document.add_eid2event(obj.eid(), obj)
                timebank_document.add_eid2sentence(
                    obj.eid(), timebank_sentence
                )
            if c.name == 'timex3':
                timebank_document.add_time_id2timex3(obj.tid(), obj)
                timebank_document.add_time_id2sentence(
                    obj.tid(), timebank_sentence
                )
            end_token_global_i = obj.end_token_i() + sentence_start_token_global_i
            timebank_sentence.set_sentence_end_token_global_i(end_token_global_i)
        return timebank_sentence

    def to_dict(self):
        return [
            i.to_dict() for i in self.sequence()
        ]
