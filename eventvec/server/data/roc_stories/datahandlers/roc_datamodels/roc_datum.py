class ROCDatum():
    def __init__(self):
        self._uid = None
        self._sentence_1 = None
        self._sentence_2 = None
        self._sentence_3 = None
        self._sentence_4 = None
        self._sentence_5 = None
        self._title = None

    def set_uid(self, uid):
        self._uid = uid

    def set_sentence_1(self, sentence_1):
        self._sentence_1 = sentence_1

    def set_sentence_2(self, sentence_2):
        self._sentence_2 = sentence_2

    def set_sentence_3(self, sentence_3):
        self._sentence_3 = sentence_3

    def set_sentence_4(self, sentence_4):
        self._sentence_4 = sentence_4

    def set_sentence_5(self, sentence_5):
        self._sentence_5 = sentence_5

    def set_title(self, title):
        self._title = title

    def uid(self):
        return self._uid

    def sentence_1(self):
        return self._sentence_1
    
    def sentence_2(self):
        return self._sentence_2

    def sentence_3(self):
        return self._sentence_3
    
    def sentence_4(self):
        return self._sentence_4
    
    def sentence_5(self):
        return self._sentence_5
    
    def sentences(self):
        return [
            self._sentence_1,
            self._sentence_2,
            self._sentence_3,
            self._sentence_4,
            self._sentence_5,
        ]
    
    def text(self):
        return ' '.join(self.sentences())
    
    def title(self):
        return self._title
    
    def to_dict(self):
        return {
            'uid': self.uid(),
            'sentence_1': self._sentence_1,
            'sentence_2': self._sentence_2,
            'sentence_3': self._sentence_3,
            'sentence_4': self._sentence_4,
            'sentence_5': self._sentence_5,
            'title': self._title,
        }
    
    @staticmethod
    def from_dict(datum_dict):
        datum = MNLIDatum()
        datum.set_uid(datum_dict['uid'])
        datum.set_sentence_1(datum_dict['sentence_1'])
        datum.set_sentence_2(datum_dict['sentence_2'])
        datum.set_sentence_3(datum_dict['sentence_3'])
        datum.set_sentence_4(datum_dict['sentence_4'])
        datum.set_sentence_5(datum_dict['sentence_5'])
        datum.set_title(datum_dict['title'])
        return datum
